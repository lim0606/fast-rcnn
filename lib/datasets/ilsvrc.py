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

class ilsvrc(datasets.imdb):
    def __init__(self, 
                 image_set, 
                 year, 
                 devkit_path=None, # where ILSVRC2015 is in 
                 include_negative=False, # whether include negative examples
                 ):
        datasets.imdb.__init__(self, 'ilsvrc_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'Data', 'DET', self._image_set)
        self._include_negative = include_negative
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
        image_path = os.path.join(self._data_path, 'JPEGImages',
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
            cache_file = os.path.join(self.cache_path,
                                      self.name + '_wo_neg_selective_search_roidb.pkl')
        else: # with negative example data
            cache_file = os.path.join(self.cache_path,
                                      self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        if self._image_set != 'test': #int(self._year) == 2015 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = datasets.imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb

    def _load_selective_search_roidb(self, gt_roidb):
        #filename = os.path.abspath(os.path.join(self.cache_path, '..',
        #                                        'selective_search_data',
        #                                        self.name + '.mat'))
        #assert os.path.exists(filename), \
        #       'Selective search data not found at: {}'.format(filename)
        #raw_data = sio.loadmat(filename)['boxes'].ravel()

        foldername = os.path.abspath(os.path.join(self.cache_path, '..',
                                                'selective_search_data',
                                                'ilsvrc_'+self._year, self._image_set))
        #print foldername
        assert os.path.exists(foldername)

        def read_selective_search_txt(index): 
            filename = os.path.join(foldername, index + '.txt')
            with open(filename, 'rb') as f:
                boxes_str = f.read().split('\n')[:-1]
            boxes = [np.array(box_str.split(), dtype=np.int16) for box_str in boxes_str]
            print index
            return boxes
          
        #raw_data = numpy.ndarray with size (num_images,)
        #           each raw_data[i] is numpy.ndarray with size (num_selective_search_results, 4) where each row xmin, ymin, xmax, ymax
        raw_data = [read_selective_search_txt(index) for index in self.image_index]
        raw_data = np.asarray(raw_data)

        print type(raw_data)
        print raw_data.dtype
        print raw_data.shape
        print '-----'
        print raw_data[0]
        print type(raw_data[0])
        print raw_data[0].dtype
        print raw_data[0].shape

        raise NotImplementedError('from selective search results folders to extract boxes')

        box_list = []
        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, (1, 0, 3, 2)] - 1)

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
        raise NotImplementedError('hi')
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
    
            # Load object bounding boxes into a data frame.
            for ix, obj in enumerate(objs):
                # Make pixel indexes 0-based
                x1 = float(get_data_from_tag(obj, 'xmin')) - 1
                y1 = float(get_data_from_tag(obj, 'ymin')) - 1
                x2 = float(get_data_from_tag(obj, 'xmax')) - 1
                y2 = float(get_data_from_tag(obj, 'ymax')) - 1
                cls = self._class_to_ind[
                        str(get_data_from_tag(obj, "name")).lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0
    
            overlaps = scipy.sparse.csr_matrix(overlaps)
        else: # There are images which were queried specifically for the DET
              # dataset to serve as negative training data. These images are
              # packaged as 11 folders: ILSVRC2013_DET_train_extra0 ... See, readme.txt at devkit
            num_objs = 0
            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
    
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
        path = os.path.join(self._devkit_path, 'results', 'VOC' + self._year,
                            'Main', comp_id + '_')
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = path + 'det_' + self._image_set + '_' + cls + '.txt'
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
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
