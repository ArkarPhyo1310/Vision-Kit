import time
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision
from vision_kit.utils.bboxes import clip_coords, cxcywh_to_xyxy
from vision_kit.utils.logging_utils import logger


def resize(
    img_sz: Tuple[int, int],
    img: np.ndarray,
    stride: int = 32,
    letterbox: bool = True,
    scaleup: bool = True,
    auto: bool = False,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    # Resize and pad image while meeting stride-multiple constraints
    shape: Tuple[int, ...] = img.shape[:2]  # current shape [height, width]

    img_sz = (img_sz, img_sz) if isinstance(img_sz, int) else img_sz

    # Scale ratio (new / old)
    ratio: float = min(
        img_sz[0] / shape[0], img_sz[1] / shape[1])
    # only scale down, do not scale up (for better val mAP)
    if not scaleup:
        ratio = min(ratio, 1.0)

    # Compute padding
    new_unpad: Tuple[int, int] = int(
        round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw: int = img_sz[1] - new_unpad[0]
    dh: int = img_sz[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw: int = np.mod(dw, stride)
        dh: int = np.mod(dh, stride)  # wh padding

    if shape[::-1] != new_unpad:  # resize
        img: np.ndarray = cv2.resize(
            img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(0)), int(round(dh))
    left, right = int(round(0)), int(round(dw))

    if letterbox:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
        left, right = int(round(dw-0.1)), int(round(dw+0.1))

    pad = (dw, dh)

    img: np.ndarray = cv2.copyMakeBorder(img, top, bottom, left, right,
                                         cv2.BORDER_CONSTANT, value=color)  # add border

    return img, (ratio, pad)


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    converted_bboxes = coords.clone() if isinstance(
        coords, torch.Tensor) else np.copy(coords)
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return converted_bboxes


def nms(prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = cxcywh_to_xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = torchvision.ops.box_iou(
                boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            logger.info(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded

    return output
