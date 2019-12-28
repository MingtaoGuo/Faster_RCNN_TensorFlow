import tensorflow as tf
from unified_network.config import POOLED_W, POOLED_H, NUMS_PROPOSAL, NMS_THRESHOLD, IMG_H, IMG_W, CLASSES

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

def smooth_l1(logits, targets):
    x = logits - targets
    mask = tf.cast(tf.less(tf.abs(x), 1.0), dtype=tf.float32)
    return 0.5 * tf.square(x) * mask + (tf.abs(x) - 0.5) * (1.0 - mask)

def roi_pooling(inputs, boxes, box_idx):
    inputs = tf.image.crop_and_resize(inputs, boxes, box_idx, [POOLED_H, POOLED_W])
    return inputs

def rpn2proposal(cls, reg, anchors):
    cls, reg = cls[0], reg[0]
    anchors = tf.constant(anchors, dtype=tf.float32)
    normal_bbox, reverse_bbox = offset2bbox(reg, anchors)
    score = tf.nn.softmax(cls)[:, 1]
    box_idx = tf.image.non_max_suppression(reverse_bbox, score, max_output_size=NUMS_PROPOSAL, iou_threshold=NMS_THRESHOLD)
    reverse_bbox = tf.nn.embedding_lookup(reverse_bbox, box_idx)
    normal_bbox = tf.nn.embedding_lookup(normal_bbox, box_idx)
    temp = tf.constant([[IMG_H, IMG_W, IMG_H, IMG_W]], dtype=tf.float32)
    reverse_bbox = reverse_bbox / temp
    bbox_idx = tf.zeros([NUMS_PROPOSAL], dtype=tf.int32)
    return normal_bbox, reverse_bbox, bbox_idx

def non_max_suppression(cls, boxes, nms_threshold=0.1):
    box_list = []
    score_list = []
    class_list = []
    for i in range(len(CLASSES)):
        score = cls[:, i]
        box_idx = tf.image.non_max_suppression(boxes, score, 150, nms_threshold)
        score_ = tf.nn.embedding_lookup(score, box_idx)
        boxes_ = tf.nn.embedding_lookup(boxes, box_idx)
        box_idx = tf.where(score_>0.5)[:, 0]
        boxes_ = tf.nn.embedding_lookup(boxes_, box_idx)
        score_ = tf.nn.embedding_lookup(score_, box_idx)
        box_list.append(boxes_)
        score_list.append(score_)
        class_list.append(tf.ones_like(score_) * i)
    boxes = tf.concat(box_list, axis=0)
    score = tf.concat(score_list, axis=0)
    cls = tf.concat(class_list, axis=0)
    y0, x0, y1, x1 = boxes[:, 0:1], boxes[:, 1:2], boxes[:, 2:3], boxes[:, 3:4]
    return tf.concat([x0, y0, x1, y1], axis=1), score, cls