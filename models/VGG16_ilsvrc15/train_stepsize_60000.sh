#!/bin/bash

log_file_name='VGG16_ilsvrc15_stepsize_60000_train_'$(date +%Y%m%d_%H%M%S_%N)'.log'

echo $log_file_name

GLOG_logtostderr=1 python -u tools/train_net.py --gpu 1 --solver models/VGG16_ilsvrc15/solver_stepsize_60000.prototxt --weights data/imagenet_models/VGG16.v2.caffemodel --imdb ilsvrc_2015_train --cfg models/VGG16_ilsvrc15/config.yml --iters 2400000 2>&1 | tee $log_file_name
