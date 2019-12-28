import tensorflow as tf


from unified_network.vggnet import vgg_16, roi_fc
from unified_network.config import SCALE, RATIO, CLASSES
from unified_network.ops import relu, conv, rpn2proposal, fully_connected

def rpn(inputs):
    num_anchors = len(SCALE) * len(RATIO)
    with tf.variable_scope("rpn"):
        inputs = relu(conv("conv1", inputs, 512, 3, 1))
        cls = conv("cls", inputs, num_anchors * 2, 1, 1)
        reg = conv("reg", inputs, num_anchors * 4, 1, 1)
    first_dim = tf.shape(cls)[0]
    cls = tf.reshape(cls, [first_dim, -1, 2])
    reg = tf.reshape(reg, [first_dim, -1, 4])
    return cls, reg

def unified_net(inputs, anchors):
    inputs = vgg_16(inputs)
    rpn_cls, rpn_reg = rpn(inputs)
    normal_bbox, reverse_bbox, bbox_idx = rpn2proposal(rpn_cls, rpn_reg, anchors)
    inputs = roi_fc(inputs, reverse_bbox, bbox_idx)
    inputs = tf.squeeze(inputs, axis=[1, 2])
    cls = fully_connected("classification", inputs, len(CLASSES)+1)
    reg = fully_connected("regression", inputs, 4)
    return cls, reg, normal_bbox

