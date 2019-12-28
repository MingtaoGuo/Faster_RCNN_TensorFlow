import tensorflow as tf
import numpy as np
from PIL import Image

from unified_network.networks import unified_net
from unified_network.ops import offset2bbox, non_max_suppression
from rpn_proposal.utils import generate_anchors, draw_box
from unified_network.config import IMG_H, IMG_W


anchors = generate_anchors()
def inference():
    imgs = tf.placeholder(tf.float32, [1, IMG_H, IMG_W, 3])
    cls, reg, proposal = unified_net(imgs, anchors)
    x0, y0, x1, y1 = proposal[:, 0:1], proposal[:, 1:2], proposal[:, 2:3], proposal[:, 3:4]
    x, y, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
    proposal = tf.concat([x, y, w, h], axis=1)
    normal_bbox, reverse_bbox = offset2bbox(reg, proposal)
    cls = tf.nn.softmax(cls)
    boxes, score, classes = non_max_suppression(cls, reverse_bbox)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    fast_rcnn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="vgg_16") + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="classification") + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="regression")
    rpn_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rpn")
    saver = tf.train.Saver(fast_rcnn_var)
    saver.restore(sess, "./fast_rcnn/model/model.ckpt")
    saver = tf.train.Saver(rpn_var)
    saver.restore(sess, "./rpn_proposal/model/model.ckpt")

    IMGS = np.array(Image.open("C:/Users/gmt/Desktop/cats/15.jpg").resize([IMG_W, IMG_H]))
    [BBOX, SCORE, CLS] = sess.run([boxes, score, classes], feed_dict={imgs: IMGS[np.newaxis]})
    X0, Y0, X1, Y1 = BBOX[:, 0:1], BBOX[:, 1:2], BBOX[:, 2:3], BBOX[:, 3:4]
    X, Y, W, H = (X0 + X1) / 2, (Y0 + Y1) / 2, X1 - X0, Y1 - Y0
    BBOX = np.concatenate((X, Y, W, H), axis=-1)
    Image.fromarray(np.uint8(draw_box(IMGS, BBOX, CLS))).show()


if __name__ == "__main__":
    inference()