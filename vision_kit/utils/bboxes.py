import math
from typing import Union

import numpy as np
import torch
from torch import nn


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xywhn_to_xyxy(bboxes, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else np.copy(bboxes)
    converted_bboxes[:, 0] = w * \
        (bboxes[:, 0] - bboxes[:, 2] / 2) + padw  # top left x
    converted_bboxes[:, 1] = h * \
        (bboxes[:, 1] - bboxes[:, 3] / 2) + padh  # top left y
    converted_bboxes[:, 2] = w * \
        (bboxes[:, 0] + bboxes[:, 2] / 2) + padw  # bottom right x
    converted_bboxes[:, 3] = h * \
        (bboxes[:, 1] + bboxes[:, 3] / 2) + padh  # bottom right y
    return converted_bboxes


def xyxy_to_xywhn(bboxes, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_coords(bboxes, (h - eps, w - eps))  # warning: inplace clip
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else np.copy(bboxes)

    converted_bboxes[:, 0] = (
        (bboxes[:, 0] + bboxes[:, 2]) / 2) / w  # x center
    converted_bboxes[:, 1] = (
        (bboxes[:, 1] + bboxes[:, 3]) / 2) / h  # y center
    converted_bboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) / w  # width
    converted_bboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) / h  # height

    return converted_bboxes


def xywh_to_xyxy(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else bboxes.copy()
    converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    return converted_bboxes


def cxcywh_to_xyxy(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else bboxes.copy()
    converted_bboxes[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
    converted_bboxes[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
    converted_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
    converted_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2

    return converted_bboxes


def xyxy_to_xywh(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else bboxes.copy()
    converted_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    converted_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return converted_bboxes


def xyxy_to_cxcywh(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else bboxes.copy()
    converted_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    converted_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    converted_bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    converted_bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return converted_bboxes


def xywh_to_cxcywh(bboxes: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    converted_bboxes = bboxes.clone() if isinstance(
        bboxes, torch.Tensor) else bboxes.copy()
    converted_bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    converted_bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return converted_bboxes


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def box_area(bboxes: torch.Tensor) -> torch.Tensor:
    return (bboxes[..., 2] - bboxes[..., 0]) * (bboxes[..., 3] - bboxes[..., 1])


def bbox_overlaps(
    bboxes1: torch.Tensor,
    bboxes2: torch.Tensor,
    box_format: str = "xyxy",
    mode: str = "iou",
    is_aligned: bool = False,
    eps: float = 1e-6
) -> torch.Tensor:
    """Calculate overlap between two set of bboxes.

    If ``is_aligned `` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.
    """

    assert mode in ["iou", "iof", "giou", "ciou"], f"Unsupported mode {mode}"
    assert box_format in ["xyxy", "xywh",
                          "cxcywh"], f"Unsupported box format {box_format}"
    # Either the boxes are empty or the length of boxes's last dimenstion is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]

    if box_format == "xywh":
        bboxes1 = xywh_to_xyxy(bboxes1)
        bboxes2 = xywh_to_xyxy(bboxes2)
    elif box_format == "cxcywh":
        bboxes1 = cxcywh_to_xyxy(bboxes1)
        bboxes2 = cxcywh_to_xyxy(bboxes2)
    else:
        bboxes1 = bboxes1
        bboxes2 = bboxes2

    batch_shape = bboxes1.shape[:-2]
    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows,))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = box_area(bboxes1)
    area2 = box_area(bboxes2)

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1 + area2 - overlap
        else:
            ap = (bboxes1[..., 2] - bboxes1[..., 0]) * \
                (bboxes1[..., 3] - bboxes1[..., 1])
            ag = (bboxes2[..., 2] - bboxes2[..., 0]) * \
                (bboxes2[..., 3] - bboxes2[..., 1])
            union = ap + ag - overlap + eps
        if mode in ["giou", "ciou"]:
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(
            bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
        )  # [B, rows, cols, 2]
        rb = torch.min(
            bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
        )  # [B, rows, cols, 2]

        wh = (rb - lt).clamp(min=0)  # [B, rows, cols, 2]
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ["iou", "giou"]:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            ap = (bboxes1[..., :, None, 2] - bboxes1[..., :, None, 0]) * \
                (bboxes1[..., :, None, 3] - bboxes1[..., :, None, 1])
            ag = (bboxes2[..., :, None, 2] - bboxes2[..., :, None, 0]) * \
                (bboxes2[..., :, None, 3] - bboxes2[..., :, None, 1])
            union = ap + ag - overlap + eps
        if mode in ["giou", "ciou"]:
            enclosed_lt = torch.min(
                bboxes1[..., :, None, :2], bboxes2[..., None, :, :2]
            )
            enclosed_rb = torch.max(
                bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]
            )

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ["iou", "iof"]:
        return ious

    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0)

    if mode == "ciou":
        cw = enclose_wh[..., 0]
        ch = enclose_wh[..., 1]

        c2 = cw ** 2 + ch ** 2 + eps

        b1_x1, b1_y1 = bboxes1[..., 0], bboxes1[..., 1]
        b1_x2, b1_y2 = bboxes1[..., 2], bboxes1[..., 3]
        b2_x1, b2_y1 = bboxes2[..., 0], bboxes2[..., 1]
        b2_x2, b2_y2 = bboxes2[..., 2], bboxes2[..., 3]

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
        rho2 = left + right

        factor = 4 / math.pi**2
        v = factor * torch.pow(torch.atan(w2 / (h2 + eps)) -
                               torch.atan(w1 / (h1 + eps)), 2)

        with torch.no_grad():
            # alpha = v / (1 - ious + v)
            alpha = v / (v - ious + (1 + eps))

        # CIoU
        cious = ious - (rho2 / c2 + v * alpha)
        return cious

    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area

    return gious
