import tensorflow as tf
import scipy.io as sio
import numpy as np

from fast_rcnn.networks import network
from fast_rcnn.ops import smooth_l1, xywh2x1y1x2y2
from fast_rcnn.utils import read_batch
from fast_rcnn.config import IMG_H, IMG_W, BATCHSIZE, MINIBATCH, EPSILON, WEIGHT_DECAY, LEARNING_RATE, MOMENTUM, CLASSES

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

    one_hot = tf.one_hot(target_classes, len(CLASSES) + 1)
    pos_nums = tf.reduce_sum(tf.cast(masks, dtype=tf.float32))
    loss_cls = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.nn.softmax(cls) * one_hot, axis=-1) + EPSILON)) / pos_nums
    loss_reg = tf.reduce_sum(tf.reduce_sum(smooth_l1(reg, target_bboxes), axis=-1) * masks) / pos_nums
    regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = loss_cls + loss_reg + regular * WEIGHT_DECAY
    with tf.variable_scope("Opt"):
        Opt = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM).minimize(total_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
    saver.restore(sess, "../pretrained_VGG/vgg_16.ckpt")
    saver = tf.train.Saver()
    LR = LEARNING_RATE
    for i in range(40001):
        if i > 30000:
            LR = 0.0001
        BATCH_IMGS, BATCH_PROPOSAL, TARGET_BBOXES, TARGET_BBOXES_IDX, TARGET_CLASSES, MASKS = read_batch(proposals)
        sess.run(Opt, feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS,
                                 target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: LR})
        if i % 10 == 0:
            [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss], feed_dict={imgs: BATCH_IMGS, batch_proposal: BATCH_PROPOSAL, masks: MASKS,
                                 target_bboxes: TARGET_BBOXES, target_bboxes_idx: TARGET_BBOXES_IDX,target_classes: TARGET_CLASSES, learning_rate: LR})
            print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG))
        if i % 100 == 0:
            saver.save(sess, "./model/model.ckpt")

if __name__ == "__main__":
    train()