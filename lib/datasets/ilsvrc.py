# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import datasets
import datasets.ilsvrc #import datasets.pascal_voc
import os
import datasets.imdb
import xml.dom.minidom as minidom
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import time

from PIL import Image

import random 

class ilsvrc(datasets.imdb):
    def __init__(self, 
                 image_set, 
                 year, 
                 devkit_path=None, # where ILSVRC2015 is in 
                 include_negative=False, # whether include negative examples
                 include_exhaustive_search_in_test=True, # whether include exhaustive search in test (val or test)
                 ):
        datasets.imdb.__init__(self, 'ilsvrc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        if self._image_set is 'trainval': 
            self._data_path = os.path.join(self._devkit_path, 'Data', 'DET')
        else: 
            self._data_path = os.path.join(self._devkit_path, 'Data', 'DET', self._image_set)
        self._include_negative = include_negative
        self._include_exhaustive_search_in_test = include_exhaustive_search_in_test

        # load classes
        #self._classes = ('__background__', # always index 0
        #                 'aeroplane', 'bicycle', 'bird', 'boat',
        #                 'bottle', 'bus', 'car', 'cat', 'chair',
        #                 'cow', 'diningtable', 'dog', 'horse',
        #                 'motorbike', 'person', 'pottedplant',
        #                 'sheep', 'sofa', 'train', 'tvmonitor')
        try: 
            f = open(os.path.join(self._devkit_path, 'devkit/data/map_det.txt'), 'r'); 
        except: 
            raise ValueError('Failed to open map_det.txt file in devkit. Tried to open {}'.format( os.path.join(self._devkit_path, 'devkit/data/map_det.txt') ))
        fread = f.read()
        classes = []; classes.append('__background__')
        for line in fread.split('\n')[:-1]:
            linesplit = line.split()
            classes.append(linesplit[0])
            assert classes[int(linesplit[1])] == linesplit[0], "classes[{}] should be matched with {}".format(int(linesplit[1]), linesplit[0])       
        classes = tuple(classes)
        self._classes = classes
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.JPEG' #'.jpg'
        self._image_index = self._load_image_set_index()

        if self._include_negative is False: # not include negative examples
            self._image_index = [img_index for img_index in self._image_index if 'extra' not in img_index] # only positive images
        else: # include negative examgle
            pass # do nothing
        #self._image_index = [self._image_index[i] for i in [230911, 230961, 231345, 332698]]
        #cache_file = 'aa.pkl'
        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as fid:
        #        self._image_index = cPickle.load(fid)
        #    print 'image_index loaded from {}'.format(cache_file)
        #else:
        #    aa = np.arange(len(self._image_index))
        #    random.shuffle(aa)
        #    self._image_index = [self._image_index[i] for i in aa[:10000]]
        #    with open(cache_file, 'wb') as fid:
        #        cPickle.dump(self._image_index, fid, cPickle.HIGHEST_PROTOCOL)
        #    print 'wrote image_index to {}'.format(cache_file)

        # filter out the images has no ground truth (only when training)
        if self._image_set in {'train', 'trainval'}:
          start_time = time.time()        
          def num_gt_roi_in_index(index):
              if self._image_set is 'trainval':
                  filename = os.path.join(self._devkit_path, 'Annotations', 'DET', index + '.xml')
              else:
                  filename = os.path.join(self._devkit_path, 'Annotations', 'DET', self._image_set, index + '.xml')

              with open(filename) as f:
                  data = minidom.parseString(f.read())

              objs = data.getElementsByTagName('object')
              num_objs = len(objs)
              return num_objs

          self._image_index = [img_index for img_index in self._image_index if num_gt_roi_in_index(img_index)] # only images having ground truth
          end_time = time.time()
          print 'filter out images w/o ground truth: %.3f (sec)', (end_time - start_time)
    
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # PASCAL specific config options
        self.config = {'cleanup'  : True,
                       'use_salt' : True,
                       'top_k'    : 2000}

        assert os.path.exists(self._devkit_path), \
                'VOCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path,
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /ImageSets/DET/train/train.txt
        image_set_file = os.path.join(self._devkit_path, 'ImageSets', 'DET',
                                      self._image_set + '.txt')

        # When you try to use train and val (both), trainval.txt has to be made manually. So, re run it.
        if self._image_set is 'trainval':
            if not os.path.exists(image_set_file):
                # cat train.txt val.txt > trainval.txt
                train_filename = os.path.join(self._devkit_path, 'ImageSets', 'DET', "train" + '.txt')
                tmp_train_filename = os.path.join(self._devkit_path, 'ImageSets', 'DET', "tmp_train" + '.txt')
                val_filename = os.path.join(self._devkit_path, 'ImageSets', 'DET', "val" + '.txt')
                tmp_val_filename = os.path.join(self._devkit_path, 'ImageSets', 'DET', "tmp_val" + '.txt')
                trainval_filename = os.path.join(self._devkit_path, 'ImageSets', 'DET', self._image_set + '.txt')
                os.system(" ".join(["sed","-e","'s/^/train\//'", train_filename, ">", tmp_train_filename]))
                time.sleep(1) # wait 1 sec
                os.system(" ".join(["sed","-e","'s/^/val\//'", val_filename, ">", tmp_val_filename]))
                time.sleep(1) # wait 1 sec
                os.system(" ".join(["cat", tmp_train_filename, tmp_val_filename, ">", trainval_filename]))
                time.sleep(1) # wait 1 sec
                os.system(" ".join(["rm", tmp_train_filename]))
                time.sleep(1) # wait 1 sec
                os.system(" ".join(["rm", tmp_val_filename]))
                time.sleep(1) # wait 1 sec
                print "Done to execute a command for writing trainval.txt in ", \
                      os.path.join(self._devkit_path, 'ImageSets', 'DET')
                print "If you still have error with 'Path does not exist:' message, manually run the command"

        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip().split()[0] for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where ILSVRC is expected to be installed.
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'ILSVRCdevkit' + self._year)

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self._include_negative is False: # without negative example data
            cache_file = os.path.join(self.cache_path, self.name + '_wo_neg_gt_roidb.pkl')
        else: # with negative example data
            cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
 
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self._include_negative is False: # without negative example data
            if self._include_exhaustive_search_in_test:
                cache_file = os.path.join(self.cache_path,
                                          self.name + '_wo_neg_selective_search_w_exhaustive_search_roidb.pkl')
            else:
                cache_file = os.path.join(self.cache_path,
                                          self.name + '_wo_neg_selective_search_roidb.pkl')
        else: # with negative example data
            if self._include_exhaustive_search_in_test:
                cache_file = os.path.join(self.cache_path,
                                          self.name + '_selective_search_w_exhaustive_search_roidb.pkl')
            else:
                cache_file = os.path.join(self.cache_path,
                                          self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set not in {'test', 'val'}: # for training
            gt_roidb = self.gt_roidb()
            ##jhlim
            #print 'cccccccccccccccccccccc'
            #boxes = gt_roidb[67]['boxes']
            #for i in xrange(boxes.shape[0]):
            #  print boxes[i,:]
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            ##jhlim
            #print 'dddddddddddddddddddddd'
            #boxes = ss_roidb[67]['boxes']
            #for i in xrange(10):
            #  print boxes[i,:]
            #print 'eeeeeeeeeeeeeeeeeeeee'
            #print 'gt_roidb: ', gt_roidb
            #print 'ss_roidb: ', ss_roidb
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else: # if self._image_set in {'test', 'val'} # for evaluation
            if self._include_exhaustive_search_in_test:
                es_roidb = self._load_exhaustive_search_roidb(None)
                ss_roidb = self._load_selective_search_roidb(None)
                roidb = datasets.imdb.merge_roidbs(ss_roidb, es_roidb)
            else: 
                roidb = self._load_selective_search_roidb(None)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    # This function is called only when self._image_set = 'val' or 'test'
    def _load_exhaustive_search_roidb(self, gt_roidb):

        def read_exhaustive_search(index): 
            ##juhyeon
	    #if self._image_set is 'val':
	    #    filename = os.path.join(self._devkit_path, 'Annotations', 'DET', self._image_set, index +'.xml')
	    #with open(filename) as f:
	    #    data = minidom.parseString(f.read())
	    #def get_data_from_tag(node, tag):
            #        return node.getElementsByTagName(tag)[0].childNodes[0].data	    
	    #width = int(get_data_from_tag(data, 'width'))
	    #height = int(get_data_from_tag(data, 'height'))
	    filename = self.image_path_from_index(index)
            img = Image.open(filename)
            width, height = img.size
	    window_scale = 1.5
	    step_size = 32
	    win_width= 64
	    win_height = 64

            boxes = []
	    while((win_width < width) or (win_height < height)):
	        for y in xrange(0, height - win_height, step_size):
		    for x in xrange(0, width - win_width, step_size):
	                boxes.append([x, y, x+win_width, y+win_height]) #boxes.append([y, x, y+win_height, x+win_width])
                win_width = int(win_width * window_scale)
	        win_height = int(win_height * window_scale)

	    #print len(boxes)								 
            boxes = np.array(boxes)
            #print index
            #print boxes
            #print type(boxes)
            #print type(boxes[0])
            return boxes

        #box_list = list with len num_images
        #           each box_list[i] is numpy.ndarray with size (num_selective_search_results, 4) where each row xmin, ymin, xmax, ymax
        box_list = [read_exhaustive_search(index) for index in self.image_index]

        return self.create_roidb_from_box_list(box_list, gt_roidb)


    def _load_selective_search_roidb(self, gt_roidb):
        #filename = os.path.abspath(os.path.join(self.cache_path, '..',
        #                                        'selective_search_data',
        #                                        self.name + '.mat'))
        #assert os.path.exists(filename), \
        #       'Selective search data not found at: {}'.format(filename)
        #raw_data = sio.loadmat(filename)['boxes'].ravel()

        if self._image_set is 'trainval':
            foldername = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                'ilsvrc_'+self._year))
        else: 
            foldername = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                'ilsvrc_'+self._year, self._image_set))
        #print foldername
        assert os.path.exists(foldername)

        def read_selective_search_txt(index): 
            filename = os.path.join(foldername, index + '.txt')
            with open(filename, 'rb') as f:
                boxes_str = f.read().split('\n')[:-1]
            #boxes = [np.array(box_str.split(), dtype=np.uint16) for box_str in boxes_str]
            boxes = [np.array(box_str.split(), dtype=np.float).astype(np.uint16) for box_str in boxes_str]

            # if boxes is empty (i.e. selective search result is empty)
            if not boxes:
                print 'index: ', index
                print 'boxes: ', boxes

                if self._image_set is 'trainval':
                    filename = os.path.join(self._devkit_path, 'Annotations', 'DET', index + '.xml')
                else: 
                    filename = os.path.join(self._devkit_path, 'Annotations', 'DET', self._image_set, index + '.xml')

                def get_data_from_tag(node, tag):
                    return node.getElementsByTagName(tag)[0].childNodes[0].data

                with open(filename) as f:
                    data = minidom.parseString(f.read())

                ##jhlim
                box = np.zeros((4,), dtype=np.uint16)
                x1 = float(1)                                 #xmin
                y1 = float(1)                                 #ymin
                x2 = float(get_data_from_tag(data, 'width'))  #xmax
                y2 = float(get_data_from_tag(data, 'height')) #ymax
                box[:] = [y1, x1, y2, x2] # selecive search outputs come out with ymin xmin ymax xmax order
                boxes.append(box)
                print 'boxes is empty (i.e. selective search result is empty). [1 1 width height] box added'

	    #print len(boxes)								 
            boxes = np.array(boxes)
            #print index
            #print boxes
            #print type(boxes)
            #print type(boxes[0])
            return boxes

        #print 'aaaaaaaaaaaaaaaaaaaaaaaa'
        #for i in [230911, 230961, 231345, 332698]:
        #  print self.image_index[i]
        #  print read_selective_search_txt(self.image_index[i])
          
        #raw_data = numpy.ndarray with size (num_images,)
        #           each raw_data[i] is numpy.ndarray with size (num_selective_search_results, 4) where each row xmin, ymin, xmax, ymax
        raw_data = [read_selective_search_txt(index) for index in self.image_index]
        #raw_data = [read_selective_search_txt(self.image_index[i]) for i in xrange(10)]
        raw_data = np.asarray(raw_data)

        #print type(raw_data)
        #print raw_data.dtype
        #print raw_data.shape
        #print '-----'
        #print raw_data[0]
        #print type(raw_data[0])
        #print raw_data[0].dtype
        #print raw_data[0].shape
         
        ##jhlim 
        #boxes = raw_data[297]
        #print 'aaaaaaaaaaaaaaaaaaaaaaaa'
        #print self.image_index[297]
        #for i in xrange(10):
        #    print boxes[i,:]
        #raise NotImplementedError('from selective search results folders to extract boxes')

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

        ##jhlim
        #print 'bbbbbbbbbbbbbbbbbbbbbbbbb'
        #boxes = box_list[297]
        #for i in xrange(10):
        #    print boxes[i,:]

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def selective_search_IJCV_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        if self._include_negative is False: # without negative example data
            cache_file = os.path.join(self.cache_path,
                    '{:s}_wo_neg_selective_search_IJCV_top_{:d}_roidb.pkl'.
                    format(self.name, self.config['top_k'])) 
        else: # with negative example data
            cache_file = os.path.join(self.cache_path,
                    '{:s}_selective_search_IJCV_top_{:d}_roidb.pkl'.
                    format(self.name, self.config['top_k'])) 

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_IJCV_roidb(gt_roidb)
        roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_IJCV_roidb(self, gt_roidb):
        raise NotImplementedError('_load_selective_search_IJCV_roidb')
        ''' 
        IJCV_path = os.path.abspath(os.path.join(self.cache_path, '..',
                                                 'selective_search_IJCV_data',
                                                 'voc_' + self._year))
        assert os.path.exists(IJCV_path), \
               'Selective search IJCV data not found at: {}'.format(IJCV_path)

        top_k = self.config['top_k']
        box_list = []
        for i in xrange(self.num_images):
            filename = os.path.join(IJCV_path, self.image_index[i] + '.mat')
            raw_data = sio.loadmat(filename)
            box_list.append((raw_data['boxes'][:top_k, :]-1).astype(np.uint16))

        return self.create_roidb_from_box_list(box_list, gt_roidb)
        '''

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        if self._image_set is 'trainval':
            filename = os.path.join(self._devkit_path, 'Annotations', 'DET', index + '.xml')
        else:
            filename = os.path.join(self._devkit_path, 'Annotations', 'DET', self._image_set, index + '.xml')
        # print 'Loading: {}'.format(filename)

        def get_data_from_tag(node, tag):
            return node.getElementsByTagName(tag)[0].childNodes[0].data

        if 'extra' not in filename: # positive dataset 
            with open(filename) as f:
                data = minidom.parseString(f.read())
    
            objs = data.getElementsByTagName('object')
            num_objs = len(objs)
    
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

            ##jhlim
            width = float(get_data_from_tag(data, 'width'))
            height =float(get_data_from_tag(data, 'height'))
    
            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                # ILSVRC2015's DET annotation is provided with 0-based indexes
                # Do not need to make pixel indexes 0-based
                # Note: ILSVRC2015's DET annotation has weird annotations, such as some xmax or ymax equals to width or height, respectively. Thus, we manually limit the xmax and ymax to be less than width and height, respectively.
                x1 = float(get_data_from_tag(obj, 'xmin')) #- 1
                y1 = float(get_data_from_tag(obj, 'ymin')) #- 1
                x2 = min(float(get_data_from_tag(obj, 'xmax')), width-1) #- 1
                y2 = min(float(get_data_from_tag(obj, 'ymax')), height-1) #- 1
                cls = self._class_to_ind[
                        str(get_data_from_tag(obj, "name")).lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
               
                ##jhlim
                #if 'n00141669_27.xml' in filename:
                #  print filename
                #  print 'x1: ', x1, ', x2: ', x2, ', y1: ', y1, ', y2: ', x2
                #assert x1 <= width, 'x1: %d' % {x1}
                #assert x2 <= width, 'x1: %d' % {x2}
                #assert y1 <= height, 'y1: %d' % {y1}
                #assert y2 <= height, 'y2: %d' % {y2}
    
            overlaps = scipy.sparse.csr_matrix(overlaps)
        else: # There are images which were queried specifically for the DET
              # dataset to serve as negative training data. These images are
              # packaged as 11 folders: ILSVRC2013_DET_train_extra0 ... See, readme.txt at devkit
            num_objs = 0
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            raise NotImplementedError('gt_boxses are empty for negative training examples')
   
        # filter out wrong ground truth annotation, esp. ILSVRC2014_train_0006/ILSVRC2014_train_00060036.JPEG
        num_objs = len(boxes)
        ixs_width = np.where(boxes[:, 2] > boxes[:, 0])[0]
        boxes = boxes[ixs_width,:]
        assert (boxes[:, 2] >= boxes[:, 0]).all() # width
        ixs_height = np.where(boxes[:, 3] > boxes[:, 1])[0]
        boxes = boxes[ixs_height,:]
        assert (boxes[:, 3] >= boxes[:, 1]).all() # height 
        if (num_objs is not len(ixs_width)) or (num_objs is not len(ixs_height)):
            print 'There is wrong ground truth(s) in index: ', index
            print '.. ignored the wrong indices'
 
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

    def _write_voc_results_file(self, all_boxes):
        use_salt = self.config['use_salt']
        comp_id = 'comp4'
        if use_salt:
            comp_id += '-{}'.format(os.getpid())

        # VOCdevkit/results/VOC2007/Main/comp4-44503_det_test_aeroplane.txt
        path = os.path.join(self._devkit_path, 'results', comp_id + '_')
	filename = path + 'det_' + self._image_set + '.txt'
        with open(filename, 'wt') as f:
	    for cls_ind, cls in enumerate(self.classes):
            	if cls == '__background__':
                    continue
		for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
		    # but ilsvrc expects 0-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:d} {:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n'. \
                                format(im_ind+1, cls_ind, dets[k, -1], \
                                       dets[k, 0], dets[k, 1],  \
                                       dets[k, 2], dets[k, 3]))
        return comp_id

    def _do_matlab_eval(self, comp_id, output_dir='output'):
        rm_results = self.config['cleanup']

        path = os.path.join(os.path.dirname(__file__),
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(datasets.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\',{:d}); quit;"' \
               .format(self._devkit_path, comp_id,
                       self._image_set, output_dir, int(rm_results))
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        comp_id = self._write_voc_results_file(all_boxes)
        self._do_matlab_eval(comp_id, output_dir)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    d = datasets.ilsvrc('train', '2015')
    res = d.roidb
    from IPython import embed; embed()
