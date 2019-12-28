CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle",
         "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
RATIO = [0.5, 1.0, 2.0]
SCALE = [128, 256, 512]

IMG_H = 600
IMG_W = 800
NMS_THRESHOLD = 0.7
NUMS_PROPOSAL = 2000

MINIBATCH = 256
BATCHSIZE = 2
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

EPSILON = 1e-10

XML_PATH = "../VOCdevkit/VOC2007/Annotations/"
IMG_PATH = "../VOCdevkit/VOC2007/JPEGImages/"
