# Cervical Cancer Detection(初赛24/2359代码)
——————————team：NoGameNoLife 


This code is for the competition of ['Digitized Human Body' Visual Challenge - Intelligent Diagnosis of Cervical Cancer Risk](https://tianchi.aliyun.com/competition/entrance/231757/introduction). The purpose of the competition is to provide large-scale thin-layer cell data of cervical cancer labeled by professional doctors. The competitors can propose and comprehensively use methods such as object detection and deep learning to locate abnormal squamous epithelial cells (i.e., ASC) of cervical cancer cytology and classify cervical cancer cells through images, which improve the speed and accuracy of model detection, and assist doctors in real diagnosis.  
![image](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/156976273635179161569762735242.jpeg)
Note: Data and kfbreader is not allowed to be published, but [experiment details](https://mp.weixin.qq.com/s?__biz=MzUyNzA1OTcxNg==&mid=2247483668&idx=1&sn=e9c3d6afd96ebdd5c330825b6e5d5188&chksm=fa041f7fcd739669af9cc181ffcd9bf1bd3ed32c156d7c8adb860104ef4ac0a87cb5e8420140&token=1035786795&lang=zh_CN#rd) are pulished..  

## 特别声明
感谢您的查看，由于脚本内路径全部需要调整来符合代码规范，如遇路径问题无法运行代码请及时与我们联系。谢谢！

## 方案概述
由于预测结果中假阳性较多，我们使用了多级分类检测的方法，首先利用所有整图数据训练一个resnext101_32x8d分类整图的阴性和阳性，再利用ROI中不含有目标的切块训练一个同样结构的resnext101_32x8d来分类切块是否含有目标。最后将经过两轮过滤的切块用来目标检测推理。此方法除了过滤假阳性之外，还有效缩短了约60%的推理时间。

## 模型实现
整图分类和小图切块分类使用torchivision的预训练模型，训练脚本参考PyTorch官方教程。
目标检测使用[mmdetection](https://github.com/open-mmlab/mmdetection)实现的目标检测算法。最佳单模型为Libra faster R-CNN resnext_101_64x8d.

## Dependency
默认环境已成功安装PyTorch, OpenCV以及mmdetection
PyTorch >= 1.3.0
CUDA = 10.1


## 预处理
### 数据预处理
创建所需文件夹
```
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
mkdir submissions
mkdir -p models/tile_classify
mkdir -p models/posneg
```

将所有图片合并到同一文件夹，方便处理
```{shell}
cp data/train/pos*/* data/train_pos
cp data/train/neg*/* data/train_neg
mv data/test data/test_raw
mkdir data/test
cp data/test_raw/test_*/* data/test
```

移动用户数据到指定位置
```
cp -r user_data/tmp_data/analyses data/
cp -r user_data/tmp_data/annotations data/
cp user_data/tmp_data/misc/* data/
```

### 图片切块 (tiling)
```{shell}
python code/scripts/image_tiling.py
python code/scripts/image_tiling_test.py
```

## 复现推理
在此步骤前请确保所有切块已正确切割并放置在相应目录下。

### 推理
执行目标检测算法，检测小图中的阳性异常细胞
```
cd code/mmdetection/
./tools/dist_test.sh cervical_config/tile_libra_faster_rcnn_x101_64x4d_fpn_1x_new.py ../../user_data/model_data/20191115_234658_libra_x101_epoch_15.pth 1 --json_out work_dirs/tile_libra_faster_rcnn_x101_64x4d_fpn_1x_new/20191115_234658_libra_x101_epoch_15_only_pos.json
```

进一步过滤整图
```
cd $PROJECT_ROOT
python code/scripts/filter_neg_image.py
```

最终结果存于`./prediction_result/result/`

## 复现训练
### 训练整图分类网络
通过ReadRoi以5000x5000的大小依次读取，拼接成全分辨率的整图，然后再将整图的短边resize至1024。
```
python code/scripts/resize_all_pos.py
python code/scripts/resize_all_neg.py
python code/scripts/resize_all_test.py
```

训练网络
```
python code/scripts/train_pos_neg.py
```

推理测试集整图
```
python code/scripts/pos_neg_predict.py
```
结果存于`./data/posneg_predict_0.6.csv`

### 训练tile小图分类网络
将所有含有目标的切块作为正样本与随机等量不含目标的切块作为负样本一起训练分类网络。
```
python code/scripts/tile_posneg_classify.py
```

推理测试切块是否含有目标
```
python code/scripts/tile_classify_predict.py
```
结果存于`./data/tile_classify_train.csv`

### 通过mmdetection训练Libra faster RCNN
```
cd code/mmdetection
./tools/dist_train.sh cervical_config/tile_libra_faster_rcnn_x101_64x4d_fpn_1x_new.py 4 --validate
```
