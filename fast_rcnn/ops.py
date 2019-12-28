import tensorflow as tf
from fast_rcnn.config import POOLED_H, POOLED_W, IMG_H, IMG_W

def conv(name, inputs, nums_out, k_size, strides):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [k_size, k_size, nums_in, nums_out], initializer=tf.random_normal_initializer(mean=0, stddev=0.01))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], "SAME")
        inputs = tf.nn.bias_add(inputs, b)
    return inputs

def relu(inputs):
    return tf.nn.relu(inputs)

def max_pooling(inputs):
    return tf.nn.max_pool(inputs, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def fully_connected(name, inputs, nums_out):
    nums_in = inputs.shape[-1]
    with tf.variable_scope(name):
        W = tf.get_variable("W", [nums_in, nums_out], initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        b = tf.get_variable("b", [nums_out], initializer=tf.constant_initializer([0.]))
        inputs = tf.matmul(inputs, W) + b
    return inputs

def bbox2offset(gt_bbox, anchor_bbox):
    t_x = (gt_bbox[:, 0:1] - anchor_bbox[:, 0:1])/anchor_bbox[:, 2:3]
    t_y = (gt_bbox[:, 1:2] - anchor_bbox[:, 1:2])/anchor_bbox[:, 3:4]
    t_w = tf.log(gt_bbox[:, 2:3] / anchor_bbox[:, 2:3])
    t_h = tf.log(gt_bbox[:, 3:4] / anchor_bbox[:, 3:4])
    return tf.concat([t_x, t_y, t_w, t_h], axis=-1)

def offset2bbox(pred, anchors):
    t_x, t_y, t_w, t_h = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    a_x, a_y, a_w, a_h = anchors[:, 0:1], anchors[:, 1:2], anchors[:, 2:3], anchors[:, 3:4]
    x = t_x * a_w + a_x
    y = t_y * a_h + a_y
    w = tf.exp(t_w) * a_w
    h = tf.exp(t_h) * a_h
    x0, y0 = x - w/2, y - h/2
    x1, y1 = x + w/2, y + h/2
    reverse_bbox = tf.concat([y0, x0, y1, x1], axis=-1)
    normal_bbox = tf.concat([x0, y0, x1, y1], axis=-1)
    return normal_bbox, reverse_bbox

def xywh2x1y1x2y2(xywh):
    x, y, w, h = xywh[:, 0:1], xywh[:, 1:2], xywh[:, 2:3], xywh[:, 3:4]
    x1, y1 = x - w / 2, y - h / 2
    x2, y2 = x + w / 2, y + h / 2
    new_h, new_w = IMG_H // 16, IMG_W // 16
    return tf.concat([y1 / 16 / new_h, x1 / 16 / new_w, y2 / 16 / new_h, x2 / 16 / new_w], axis=1)

def smooth_l1(logits, targets):
    x = logits - targets
    mask = tf.cast(tf.less(tf.abs(x), 1.0), dtype=tf.float32)
    return 0.5 * tf.square(x) * mask + (tf.abs(x) - 0.5) * (1.0 - mask)

def roi_pooling(inputs, boxes, box_idx):
    inputs = tf.image.crop_and_resize(inputs, boxes, box_idx, [POOLED_H, POOLED_W])
    return inputs


