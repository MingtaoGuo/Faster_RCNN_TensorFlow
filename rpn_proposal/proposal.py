import tensorflow as tf
import numpy as np
import scipy.io as sio
import os

from PIL import Image
from rpn_proposal.ops import offset2bbox
from rpn_proposal.vggnet import vgg_16
from rpn_proposal.networks import rpn
from rpn_proposal.utils import generate_anchors
from rpn_proposal.config import IMG_H, IMG_W, NMS_THRESHOLD, NUMS_PROPOSAL, XML_PATH, IMG_PATH


def proposal():
    anchors = generate_anchors()
    inputs = tf.placeholder(tf.float32, [1, IMG_H, IMG_W, 3])

    vgg_logits = vgg_16(inputs)
    cls, reg = rpn(vgg_logits)
    cls, reg = cls[0], reg[0]
    scores = tf.nn.softmax(cls)[:, 1]
    anchors = tf.constant(anchors, dtype=tf.float32)
    normal_bbox, reverse_bbox = offset2bbox(reg, anchors)
    nms_idxs = tf.image.non_max_suppression(reverse_bbox, scores, max_output_size=2000, iou_threshold=NMS_THRESHOLD)
    bboxes = tf.nn.embedding_lookup(normal_bbox, nms_idxs)[:NUMS_PROPOSAL]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, "./model/model.ckpt")

    xml_files = os.listdir(XML_PATH)
    proposal_data = {}
    for idx, filename in enumerate(xml_files):
        img = np.array(Image.open(IMG_PATH + filename[:-3] + "jpg").resize([IMG_W, IMG_H]))
        BBOX = sess.run(bboxes, feed_dict={inputs: img[np.newaxis]})
        x, y = (BBOX[:, 0:1] + BBOX[:, 2:3]) / 2, (BBOX[:, 1:2] + BBOX[:, 3:4]) / 2
        w, h = BBOX[:, 2:3] - BBOX[:, 0:1], BBOX[:, 3:4] - BBOX[:, 1:2]
        BBOX = np.concatenate((x, y, w, h), axis=-1)
        proposal_data[filename] = BBOX
        print(idx)
    sio.savemat("../proposal.mat", proposal_data)


if __name__ == "__main__":
    proposal()