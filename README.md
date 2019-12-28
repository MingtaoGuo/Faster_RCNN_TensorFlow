# Faster_RCNN_TensorFlow
Simple implementation of faster rcnn by TensorFlow, containing 4 step training and easy to understand

# Introduction
![](https://github.com/MingtaoGuo/Faster_RCNN_TensorFlow/blob/master/IMGS/faster rcnn.jpg)
# Dataset
Pascal Voc 2007

# Requirements
1. tensorflow 1.4.0
2. numpy
3. scipy
4. pillow

# How to use
1. Download the pretrained vgg16 model, and then put vgg16 model into the folder 'pretrained_VGG'.
   download address: http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz (please unzip it)
2. Download the Pascal VOC 2007 dataset.
   download address: http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar (please unzip it)
3. Execute step1_train_rpn.py, step2_train_fast_rcnn.py, step3_finetune_rpn.py, step4_finetune_fast_rcnn.py in order.
4. Testing please execute inference.py

# Results


# Author
Mingtao Guo
             
