import tensorflow as tf
import scipy.io as sio
import numpy as np
from PIL import Image
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from fast_rcnn.networks import network
from fast_rcnn.ops import xywh2x1y1x2y2, offset2bbox
from fast_rcnn.utils import read_batch, offset2bbox_np, draw_box
from fast_rcnn.config import IMG_H, IMG_W, BATCHSIZE, MINIBATCH, LEARNING_RATE

# batch_imgs, batch_proposal, target_bboxes, target_bboxes_idx, target_classes, masks
proposals = sio.loadmat("../proposal.mat")
def train():
    imgs = tf.placeholder(tf.float32, [BATCHSIZE, IMG_H, IMG_W, 3])
    batch_proposal = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH, 4])
    target_bboxes = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH, 4])
    target_bboxes_idx = tf.placeholder(tf.int32, [BATCHSIZE * MINIBATCH])#for roi pooling
    target_classes = tf.placeholder(tf.int32, [BATCHSIZE * MINIBATCH])
    masks = tf.placeholder(tf.float32, [BATCHSIZE * MINIBATCH])
    learning_rate = tf.placeholder(tf.float32)

    batch_proposal_ = xywh2x1y1x2y2(batch_proposal)#for roi pooling
    cls, reg = network(imgs, batch_proposal_, target_bboxes_idx)
    scores = 1 - tf.nn.softmax(cls)[:, -1]
    boxes, _ = offset2bbox(reg, batch_proposal)
    box_idx = tf.image.non_max_suppression(boxes, scores, 300)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./model/model.ckpt")

    LR = LEARNING_RATE
    for i in range(40001):
        BATCH_IMGS, BATCH_PROPOSAL, TARGET_BBOXES, TARGET_BBOXES_IDX, TARGET_CLASSES, MASKS = read_batch(proposals)
        [CLS, REG, SCORES, BOX_IDX] = sess.run([cls, reg, scores, box_idx], feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS,
                                 target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: LR})
        a = 0

if __name__ == "__main__":
    train()