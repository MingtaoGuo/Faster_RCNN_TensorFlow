from fast_rcnn.vggnet import vgg_16
from fast_rcnn.ops import fully_connected
from fast_rcnn.config import CLASSES


def network(inputs, boxes, box_idx):
    inputs = vgg_16(inputs, boxes, box_idx)
    cls = fully_connected("classification", inputs, len(CLASSES) + 1)
    reg = fully_connected("regression", inputs, 4)
    return cls, reg

