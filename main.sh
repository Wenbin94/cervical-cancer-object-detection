#!/bin/bash

# 推理复现脚本
export PROJECT_ROOT=`pwd`
cd data
mkdir train_pos
mkdir train_pos_1024
mkdir roi_tiles
mkdir train_neg
mkdir train_neg_1024
mkdir train_all_1024
mkdir test_tiles
mkdir test_1024
cd ..
mkdir -p models/tile_classify
mkdir -p models/posneg

cp data/train/pos*/* data/train_pos
cp data/train/neg*/* data/train_neg
mv data/test data/test_raw
mkdir data/test
cp data/test_raw/test_*/* data/test
cp -r data/train/labels data/

cp -r user_data/tmp_data/analyses data/
cp -r user_data/tmp_data/annotations data/
cp user_data/tmp_data/misc/* data/

python code/scripts/image_tiling.py
python code/scripts/image_tiling_test.py

cd code/mmdetection/
./tools/dist_test.sh cervical_config/tile_libra_faster_rcnn_x101_64x4d_fpn_1x_new.py ../../user_data/model_data/20191115_234658_libra_x101_epoch_15.pth 1 --json_out work_dirs/tile_libra_faster_rcnn_x101_64x4d_fpn_1x_new/20191115_234658_libra_x101_epoch_15_only_pos.json

cd $PROJECT_ROOT
python code/scripts/filter_neg_image.py
