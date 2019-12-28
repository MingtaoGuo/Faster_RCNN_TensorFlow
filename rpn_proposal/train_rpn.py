import tensorflow as tf
import numpy as np

from rpn_proposal.vggnet import vgg_16
from rpn_proposal.ops import smooth_l1
from rpn_proposal.networks import rpn
from rpn_proposal.utils import read_batch, generate_anchors
from rpn_proposal.config import IMG_H, IMG_W, BATCHSIZE, MINIBATCH, EPSILON, WEIGHT_DECAY, LEARNING_RATE, MOMENTUM


def train():
    imgs = tf.placeholder(tf.float32, [BATCHSIZE, IMG_H, IMG_W, 3])
    bbox_indxs = tf.placeholder(tf.int32, [BATCHSIZE, MINIBATCH])
    masks = tf.placeholder(tf.int32, [BATCHSIZE, MINIBATCH])
    target_bboxes = tf.placeholder(tf.float32, [BATCHSIZE, MINIBATCH, 4])
    learning_rate = tf.placeholder(tf.float32)

    vgg_logits = vgg_16(imgs)
    cls, reg = rpn(vgg_logits)
    cls_logits = tf.concat([tf.nn.embedding_lookup(cls[i], bbox_indxs[i])[tf.newaxis] for i in range(BATCHSIZE)], axis=0)
    reg_logits = tf.concat([tf.nn.embedding_lookup(reg[i], bbox_indxs[i])[tf.newaxis] for i in range(BATCHSIZE)], axis=0)

    one_hot = tf.one_hot(masks, 2)
    pos_nums = tf.reduce_sum(tf.cast(masks, dtype=tf.float32))
    loss_cls = tf.reduce_sum(-tf.log(tf.reduce_sum(tf.nn.softmax(cls_logits) * one_hot, axis=-1) + EPSILON)) / pos_nums
    loss_reg = tf.reduce_sum(tf.reduce_sum(smooth_l1(reg_logits, target_bboxes), axis=-1) * tf.cast(masks, dtype=tf.float32)) / pos_nums
    regular = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    total_loss = loss_cls + loss_reg + regular * WEIGHT_DECAY
    with tf.variable_scope("Opt"):
        Opt = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM).minimize(total_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
    saver.restore(sess, "../pretrained_VGG/vgg_16.ckpt")
    saver = tf.train.Saver()
    anchors = generate_anchors()
    LR = LEARNING_RATE
    for i in range(80001):
        if i > 60000:
            LR = 0.0001
        BATCH_IMGS, BATCH_IDXS, TARGET_BBOXS, MASKS = read_batch(anchors)
        sess.run(Opt, feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS, learning_rate: LR})
        if i % 10 == 0:
            [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss],
            feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS})
            print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG))
        if i % 100 == 0:
            saver.save(sess, "./model/model.ckpt")

if __name__ == "__main__":
    train()