#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NGPUS=8

CFG_DIR=cfgs/kitti_models

CFG_NAME=hh3d_rcnn_car

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file $CFG_DIR/$CFG_NAME.yaml --workers 8
