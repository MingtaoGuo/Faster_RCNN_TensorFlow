import tensorflow as tf
import numpy as np
import scipy.io as sio
from PIL import Image
import os
import time

from rpn_proposal.vggnet import vgg_16
from rpn_proposal.ops import smooth_l1, offset2bbox
from rpn_proposal.networks import rpn
from rpn_proposal.utils import read_batch, generate_anchors
from rpn_proposal.config import IMG_H, IMG_W, BATCHSIZE, MINIBATCH, EPSILON, WEIGHT_DECAY, LEARNING_RATE, MOMENTUM, XML_PATH, NMS_THRESHOLD, NUMS_PROPOSAL, IMG_PATH


def train():
    imgs = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, 3])
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
    trainable_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn")
    with tf.variable_scope("Opt"):
        Opt = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM).minimize(total_loss, var_list=trainable_var)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16"))
    saver.restore(sess, "./fast_rcnn/model/model.ckpt")
    saver = tf.train.Saver()
    anchors = generate_anchors()
    LR = LEARNING_RATE
    for i in range(80001):
        if i > 60000:
            LR = 0.0001
        s = time.time()
        BATCH_IMGS, BATCH_IDXS, TARGET_BBOXS, MASKS = read_batch(anchors)
        e = time.time()
        read_time = e - s
        s = time.time()
        sess.run(Opt, feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS, learning_rate: LR})
        e = time.time()
        update_time = e - s
        if i % 100 == 0:
            [LOSS_CLS, LOSS_REG, TOTAL_LOSS] = sess.run([loss_cls, loss_reg, total_loss],
            feed_dict={imgs: BATCH_IMGS, bbox_indxs: BATCH_IDXS, masks: MASKS, target_bboxes: TARGET_BBOXS})
            print("Iteration: %d, total_loss: %f, loss_cls: %f, loss_reg: %f, read_time: %f, update_time: %f" % (i, TOTAL_LOSS, LOSS_CLS, LOSS_REG, read_time, update_time))
        if i % 1000 == 0:
            saver.save(sess, "./rpn_proposal/model/model.ckpt")

    #--------------------PROPOSAL-----------------------#
    cls, reg = cls[0], reg[0]
    scores = tf.nn.softmax(cls)[:, 1]
    anchors = tf.constant(anchors, dtype=tf.float32)
    normal_bbox, reverse_bbox = offset2bbox(reg, anchors)
    nms_idxs = tf.image.non_max_suppression(reverse_bbox, scores, max_output_size=2000, iou_threshold=NMS_THRESHOLD)
    bboxes = tf.nn.embedding_lookup(normal_bbox, nms_idxs)[:NUMS_PROPOSAL]
    saver = tf.train.Saver()
    saver.restore(sess, "./rpn_proposal/model/model.ckpt")

    xml_files = os.listdir(XML_PATH)
    proposal_data = {}
    for idx, filename in enumerate(xml_files):
        img = np.array(Image.open(IMG_PATH + filename[:-3] + "jpg").resize([IMG_W, IMG_H]))
        BBOX = sess.run(bboxes, feed_dict={imgs: img[np.newaxis]})
        x, y = (BBOX[:, 0:1] + BBOX[:, 2:3]) / 2, (BBOX[:, 1:2] + BBOX[:, 3:4]) / 2
        w, h = BBOX[:, 2:3] - BBOX[:, 0:1], BBOX[:, 3:4] - BBOX[:, 1:2]
        BBOX = np.concatenate((x, y, w, h), axis=-1)
        proposal_data[filename] = BBOX
        print("Total: %d, Current: %d"%(len(xml_files), idx))
    sio.savemat("./proposal.mat", proposal_data)

if __name__ == "__main__":
    train()


