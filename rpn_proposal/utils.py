import numpy as np
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET
import os

from rpn_proposal.config import CLASSES, RATIO, SCALE, IMG_PATH, XML_PATH, MINIBATCH, BATCHSIZE, IMG_H, IMG_W


def draw_box(img, bboxes, cls):
    #bbox: [x, y, w, h]
    x1 = np.int32(bboxes[:, 0] - bboxes[:, 2]/2)
    x2 = np.int32(bboxes[:, 0] + bboxes[:, 2]/2)
    y1 = np.int32(bboxes[:, 1] - bboxes[:, 3]/2)
    y2 = np.int32(bboxes[:, 1] + bboxes[:, 3]/2)
    nums = bboxes.shape[0]
    x1 = np.maximum(0, x1)
    x1 = np.minimum(IMG_W-1, x1)
    x2 = np.maximum(0, x2)
    x2 = np.minimum(IMG_W-1, x2)
    y1 = np.maximum(0, y1)
    y1 = np.minimum(IMG_H - 1, y1)
    y2 = np.maximum(0, y2)
    y2 = np.minimum(IMG_H-1, y2)
    for i in range(nums):
        green = np.zeros([y2[i] - y1[i], 3], dtype=np.uint8)
        green[:, 1] = np.ones([y2[i] - y1[i]], dtype=np.uint8) * 255
        img[y1[i]:y2[i], x1[i], :] = green
        img[y1[i]:y2[i], x2[i], :] = green
        green = np.zeros([x2[i] - x1[i], 3], dtype=np.uint8)
        green[:, 1] = np.ones([x2[i] - x1[i]], dtype=np.uint8) * 255
        img[y1[i], x1[i]:x2[i], :] = green
        img[y2[i], x1[i]:x2[i], :] = green
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        draw.text((x1[i], y1[i]), CLASSES[int(cls[i])])
        img = np.array(img)
    return img

def read_data(xml_path, img_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall("object")
    names = []
    gtbboxes = np.zeros([len(objects), 4], dtype=np.int32)
    for idx, obj in enumerate(objects):
        names.append(obj.find("name").text)
        xmin = int(obj.find("bndbox").find("xmin").text)
        xmax = int(obj.find("bndbox").find("xmax").text)
        ymin = int(obj.find("bndbox").find("ymin").text)
        ymax = int(obj.find("bndbox").find("ymax").text)
        gtbboxes[idx, 0] = (xmin + xmax)//2
        gtbboxes[idx, 1] = (ymin + ymax)//2
        gtbboxes[idx, 2] = xmax - xmin
        gtbboxes[idx, 3] = ymax - ymin
    img = np.array(Image.open(img_path))
    one_hot_labels = np.zeros([len(objects), len(CLASSES)])
    for idx, name in enumerate(names):
        one_hot_labels[idx, CLASSES.index(name)] = 1
    return img, gtbboxes, one_hot_labels

def generate_anchors():
    #anchors: [x, y, w, h]
    # IMG_H, IMG_W = img.shape[0], img.shape[1]
    h = int(np.ceil(IMG_H/16))
    w = int(np.ceil(IMG_W/16))
    anchor_wh = np.zeros([9, 2])
    count = 0
    for ratio in RATIO:
        for scale in SCALE:
            #ratio = h / w ---> h = ratio * w, area = h * w ---> area = ratio * w ** 2 ---> w = sqrt(area/ratio)
            area = scale ** 2
            anchor_wh[count, 0] = np.sqrt(area / ratio)
            anchor_wh[count, 1] = ratio * np.sqrt(area / ratio)
            count += 1
    anchors = np.zeros([h, w, 9, 4])
    for i in range(h):
        for j in range(w):
            anchor_xy = np.ones([9, 2])
            anchor_xy[:, 0] = anchor_xy[:, 0] * (j * 16)
            anchor_xy[:, 1] = anchor_xy[:, 1] * (i * 16)
            anchors[i, j, :, 2:4] = anchor_wh
            anchors[i, j, :, 0:2] = anchor_xy
    anchors = np.reshape(anchors, [-1, 4])
    return anchors

def resize_img_bbox(img, bboxes):
    img_h, img_w = img.shape[0], img.shape[1]
    resized_bboxes = np.zeros_like(bboxes)
    resized_bboxes[:, 0] = IMG_W * bboxes[:, 0] / img_w
    resized_bboxes[:, 1] = IMG_H * bboxes[:, 1] / img_h
    resized_bboxes[:, 2] = IMG_W * bboxes[:, 2] / img_w
    resized_bboxes[:, 3] = IMG_H * bboxes[:, 3] / img_h
    resized_img = np.array(Image.fromarray(img).resize([IMG_W, IMG_H]))
    return resized_img, resized_bboxes

def cal_ious(anchors, gtbboxes):
    anchors = anchors[np.newaxis, :, :]
    gtbboxes = gtbboxes[:, np.newaxis, :]
    anchors_x1 = anchors[:, :, 0] - anchors[:, :, 2] / 2
    anchors_x2 = anchors[:, :, 0] + anchors[:, :, 2] / 2
    anchors_y1 = anchors[:, :, 1] - anchors[:, :, 3] / 2
    anchors_y2 = anchors[:, :, 1] + anchors[:, :, 3] / 2
    gtbboxes_x1 = gtbboxes[:, :, 0] - gtbboxes[:, :, 2] / 2
    gtbboxes_x2 = gtbboxes[:, :, 0] + gtbboxes[:, :, 2] / 2
    gtbboxes_y1 = gtbboxes[:, :, 1] - gtbboxes[:, :, 3] / 2
    gtbboxes_y2 = gtbboxes[:, :, 1] + gtbboxes[:, :, 3] / 2
    inter_x1 = np.maximum(anchors_x1, gtbboxes_x1)
    inter_x2 = np.minimum(anchors_x2, gtbboxes_x2)
    inter_y1 = np.maximum(anchors_y1, gtbboxes_y1)
    inter_y2 = np.minimum(anchors_y2, gtbboxes_y2)
    inter_area = np.maximum(0., inter_x2 - inter_x1) * np.maximum(0., inter_y2 - inter_y1)
    union_area = anchors[:, :, 2] * anchors[:, :, 3] + gtbboxes[:, :, 2] * gtbboxes[:, :, 3] - inter_area
    ious = inter_area / union_area
    return ious


def generate_minibatch(anchors, gtbboxes):
    #gtbboxes: [None, 4]
    nums = anchors.shape[0]
    anchors_x1 = anchors[:, 0] - anchors[:, 2]/2
    anchors_x2 = anchors[:, 0] + anchors[:, 2]/2
    anchors_y1 = anchors[:, 1] - anchors[:, 3]/2
    anchors_y2 = anchors[:, 1] + anchors[:, 3]/2
    illegal_idx0 = np.union1d(np.where(anchors_x1<0)[0], np.where(anchors_x2>=IMG_W)[0])
    illegal_idx1 = np.union1d(np.where(anchors_y1<0)[0], np.where(anchors_y2>=IMG_H)[0])
    illegal_idx = np.union1d(illegal_idx0, illegal_idx1)
    legal_idx = np.setdiff1d(np.array(range(nums)), illegal_idx)
    legal_anchors = anchors[legal_idx]
    ious = cal_ious(legal_anchors, gtbboxes)#[nums_obj, nums_anchor]
    max_iou_idx = np.where(np.abs(ious - np.max(ious, axis=1, keepdims=True)) < 1e-3)[1]
    ious = np.max(ious, axis=0)
    iou_greater_7_idx = np.where(ious >= 0.7)[0]
    pos_idx = np.union1d(max_iou_idx, iou_greater_7_idx)
    neg_idx = np.where(ious < 0.3)[0]
    neg_idx = np.setdiff1d(neg_idx, max_iou_idx)#remove some bboxes that may be iou < 0.3, but they are the maxest overlapping

    pos_nums = pos_idx.shape[0]
    neg_nums = neg_idx.shape[0]
    if pos_nums < MINIBATCH//2:
        remain_nums = MINIBATCH - pos_nums
        rand_idx = np.random.randint(0, neg_nums, [remain_nums])
        neg_idx = neg_idx[rand_idx]
        batch_idx = np.concatenate((pos_idx, neg_idx), axis=0)
        batch_idx = legal_idx[batch_idx]
        labels = np.concatenate((np.ones([pos_nums]), np.zeros([remain_nums])))
        pos_anchor_bbox = legal_anchors[pos_idx]
        pos_iou = cal_ious(pos_anchor_bbox, gtbboxes)
        pos_gt_idx = np.argmax(pos_iou, axis=0)
        pos_gt_bbox = gtbboxes[pos_gt_idx]
    else:
        rand_pos_idx = np.random.randint(0, pos_nums, [MINIBATCH//2])
        rand_neg_idx = np.random.randint(0, neg_nums, [MINIBATCH//2])
        batch_idx = np.concatenate((pos_idx[rand_pos_idx], neg_idx[rand_neg_idx]), axis=0)
        batch_idx = legal_idx[batch_idx]
        labels = np.concatenate((np.ones([MINIBATCH//2]), np.zeros([MINIBATCH//2])), axis=0)
        pos_anchor_bbox = legal_anchors[pos_idx[rand_pos_idx]]
        pos_iou = cal_ious(pos_anchor_bbox, gtbboxes)
        pos_gt_idx = np.argmax(pos_iou, axis=0)
        pos_gt_bbox = gtbboxes[pos_gt_idx]
    target_bbox = bbox2offset(pos_anchor_bbox, pos_gt_bbox)
    init_target_bbox = np.zeros([MINIBATCH, 4])

    init_target_bbox[:target_bbox.shape[0]] = target_bbox
    return batch_idx, labels, init_target_bbox

def offset2bbox(pred_t, anchor_idx, anchors):
    anchors = anchors[np.int32(anchor_idx)]
    pred_t = pred_t[:anchor_idx.shape[0]]
    pred_bbox_x = pred_t[:, 0:1] * anchors[:, 2:3] + anchors[:, 0:1]
    pred_bbox_y = pred_t[:, 1:2] * anchors[:, 3:4] + anchors[:, 1:2]
    pred_bbox_w = np.exp(pred_t[:, 2:3]) * anchors[:, 2:3]
    pred_bbox_h = np.exp(pred_t[:, 3:4]) * anchors[:, 3:4]
    return np.concatenate((pred_bbox_x, pred_bbox_y, pred_bbox_w, pred_bbox_h), axis=-1)

def bbox2offset(anchor_bbox, gt_bbox):
    t_x = (gt_bbox[:, 0:1] - anchor_bbox[:, 0:1])/anchor_bbox[:, 2:3]
    t_y = (gt_bbox[:, 1:2] - anchor_bbox[:, 1:2])/anchor_bbox[:, 3:4]
    t_w = np.log(gt_bbox[:, 2:3] / anchor_bbox[:, 2:3])
    t_h = np.log(gt_bbox[:, 3:4] / anchor_bbox[:, 3:4])
    return np.concatenate([t_x, t_y, t_w, t_h], axis=-1)

def read_batch(anchors):
    xml_names = os.listdir(XML_PATH)
    rand_idx = np.random.randint(0, len(xml_names), [BATCHSIZE])
    batch_imgs = np.zeros([BATCHSIZE, IMG_H, IMG_W, 3])
    batch_idxs = np.zeros([BATCHSIZE, MINIBATCH])
    masks = np.zeros([BATCHSIZE, MINIBATCH])
    target_bboxes = np.zeros([BATCHSIZE, MINIBATCH, 4])
    for i in range(BATCHSIZE):
        filename = xml_names[rand_idx[i]]
        img, gtbboxes, class_labels = read_data(XML_PATH + filename, IMG_PATH + filename[:-4] + ".jpg")
        img, gtbboxes = resize_img_bbox(img, gtbboxes)
        batch_idx, labels, target_bbox = generate_minibatch(anchors, gtbboxes)
        batch_idxs[i] = batch_idx
        masks[i] = labels
        target_bboxes[i] = target_bbox
        batch_imgs[i] = img
    return batch_imgs, batch_idxs, target_bboxes, masks


# import time
# xml_path = "E:/数据集/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/"
# img_path = "E:/数据集/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/"
# xml_names = os.listdir(xml_path)
# anchors = generate_anchors()
# for idx, filename in enumerate(xml_names):
#     s = time.time()
#     # img, gtbboxes, class_labels = read_data(xml_path + filename, img_path + filename[:-4] + ".jpg")
#     # img, gtbboxes = resize_img_bbox(img, gtbboxes)
#     # batch_idx, labels, target_bbox = generate_minibatch(anchors, gtbboxes)
#     batch_imgs, batch_idxs, target_bboxes, masks = read_batch(anchors)
#     # pos_gt_bbox = offset2bbox(target_bboxes[0], batch_idxs[0, :int(sum(masks[0]))], anchors)
#     e = time.time()
#     if idx == 8:
#         a = 0
#     print(e-s)
# #
#     Image.fromarray(draw_box(batch_imgs[0], pos_gt_bbox)).save("C:/Users/gmt/Desktop/anchors/"+str(idx)+".jpg")
#     img, gtbboxes, class_labels = read_data(xml_path + filename, img_path + filename[:-4] + ".jpg")
#     img, gtbboxes = resize_img_bbox(img, gtbboxes)
#     Image.fromarray(draw_box(img, anchors[batch_idx[:int(sum(labels))]])).save("C:/Users/gmt/Desktop/anchors/" + str(idx) + "_.jpg")
#     print("idx: %d, nums_pos: %d, time: %f"%(idx, pos_gt_bbox.shape[0], e - s))
#     print(e - s)
# cal_ious(anchors, gtbboxes)
# img = draw_box(img, gtbboxes)
#
# pass