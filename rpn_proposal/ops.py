import tensorflow as tf

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


