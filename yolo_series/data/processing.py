from typing import Tuple, Union

import cv2
import numpy as np
import torch
import torchvision
from yolo_series.utils.bboxes import cxcywh_to_xyxy


class ImageProcessor:
    def __init__(
        self,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        filtered_classes: tuple = None,
        labels: tuple = (),
        img_sz: list = (640, 640),
        color: list = (114, 114, 114),
        letterbox: bool = True,
        auto: bool = False,
        scaleup: bool = False,
        agnostic: bool = False,
        multi_label: bool = False,
        max_det: int = 300,
        stride: int = 32,
    ) -> None:
        # Preprocessing config
        self.img_sz: list = img_sz
        self.color: list = color
        self.stride: int = stride
        self.letterbox: bool = letterbox
        self.auto: bool = auto
        self.scaleup: bool = scaleup

        # NMS config
        self.conf_thres: float = conf_thres
        self.iou_thres: float = iou_thres
        self.filtered_classes = filtered_classes
        self.agnostic: bool = agnostic
        self.multi_label: bool = multi_label
        self.labels = labels
        self.max_det: int = max_det

    def preprocess(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.resize(img)
        img = np.ascontiguousarray(np.array(img).transpose((2, 0, 1)))
        img_tensor = torch.from_numpy(img).unsqueeze(0) / 255
        return img_tensor

    def postprocess(self, prediction: torch.Tensor) -> torch.Tensor:
        outputs: torch.Tensor = self.nms(prediction)
        outputs: torch.Tensor = self.scale_coords(outputs[0])
        return outputs

    def resize(self, img: np.ndarray) -> np.ndarray:
        # Resize and pad image while meeting stride-multiple constraints
        shape: Tuple[int, ...] = img.shape[:2]  # current shape [height, width]
        if isinstance(self.img_sz, int):
            self.img_sz = (self.img_sz, self.img_sz)

        # Scale ratio (new / old)
        self.ratio: float = min(
            self.img_sz[0] / shape[0], self.img_sz[1] / shape[1])
        # only scale down, do not scale up (for better val mAP)
        if not self.scaleup:
            self.ratio = min(self.ratio, 1.0)

        # Compute padding
        new_unpad: Tuple[int, int] = int(
            round(shape[1] * self.ratio)), int(round(shape[0] * self.ratio))
        dw: int = self.img_sz[1] - new_unpad[0]
        dh: int = self.img_sz[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw: int = np.mod(dw, self.stride)
            dh: int = np.mod(dh, self.stride)  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img: np.ndarray = cv2.resize(
                img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(0)), int(round(dh))
        left, right = int(round(0)), int(round(dw))

        if self.letterbox:
            dw /= 2  # divide padding into 2 sides
            dh /= 2
            top, bottom = int(round(dh)), int(round(dh))
            left, right = int(round(dw)), int(round(dw))

        self.pad = (dw, dh)

        img: np.ndarray = cv2.copyMakeBorder(img, top, bottom, left, right,
                                             cv2.BORDER_CONSTANT, value=self.color)  # add border

        return img

    def scale_coords(self, outputs: torch.Tensor) -> torch.Tensor:
        # Rescale coords (xyxy)
        if self.letterbox:
            outputs[:, [0, 2]] -= self.pad[0]  # x padding
            outputs[:, [1, 3]] -= self.pad[1]  # y padding
        outputs[:, : 4] /= self.ratio
        return outputs

    def nms(self, prediction: torch.Tensor):
        bs: int = prediction.shape[0]  # batch size
        nc: int = prediction.shape[2] - 5  # number of classes
        xc: torch.Tensor = prediction[..., 4] > self.conf_thres  # candidates

        # Checks
        assert 0 <= self.conf_thres <= 1, f'Invalid Confidence threshold {self.conf_thres}, valid values are between 0.0 and 1.0'
        assert 0 <= self.iou_thres <= 1, f'Invalid IoU {self.iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 10000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        self.multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        output: list[torch.Tensor] = [torch.zeros(
            (0, 6), device=prediction.device)] * bs

        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence

            # Cat apriori labels if autolabelling
            if self.labels and len(self.labels[xi]):
                lb = self.labels[xi]
                v: torch.Tensor = torch.zeros(
                    (len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # self
                x: torch.Tensor = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue
            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box: torch.Tensor = cxcywh_to_xyxy(x[:, :4])

            # Detections matrix nx6 (xyxy, conf, self)
            if self.multi_label:
                i, j = (x[:, 5:] > self.conf_thres).nonzero(as_tuple=False).T
                x: torch.Tensor = torch.cat(
                    (box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x: torch.Tensor = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > self.conf_thres]

            # Filter by class
            if self.filtered_classes is not None:
                x: torch.Tensor = x[(x[:, 5:6] == torch.tensor(
                    self.filtered_classes, device=x.device)).any(1)]

            # Apply finite constraint
            # if not torch.isfinite(x).all():
            #     x = x[torch.isfinite(x).all(1)]

            # Check shape
            n: int = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                # sort by confidence
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]

            # Batched NMS
            c = x[:, 5: 6] * (0 if self.agnostic else max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, : 4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, self.iou_thres)  # NMS
            if i.shape[0] > self.max_det:  # limit detections
                i = i[: self.max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                # iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                iou = torchvision.ops.box_iou(
                    boxes[i], boxes) > self.iou_thres  # iou matrix

                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]

        return output
