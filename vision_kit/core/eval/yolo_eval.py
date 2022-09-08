from typing import Tuple

import numpy as np
import torch
from torchvision.ops.boxes import box_iou
from vision_kit.utils.bboxes import xywh_to_xyxy, xywhn_to_xyxy
from vision_kit.utils.dataset_utils import coco80_to_coco91_class
from vision_kit.utils.image_proc import scale_coords


class YOLOEvaluator:
    def __init__(
        self,
        class_ids: list = None,
        img_size: Tuple[int, int] = (640, 640),
        per_class_AP: bool = False,
        per_class_AR: bool = False
    ) -> None:
        self.class_ids = class_ids if class_ids else coco80_to_coco91_class()
        self.img_sz = img_size
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR

        self.p, self.r, self.f1, self.mp, self.mr, self.map50, self.map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        self.stats = []
        # iou vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.num_iou = self.iouv.numel()
        self.seen = 0

    def evaluate(
        self, img: torch.Tensor, img_infos: list,
        preds: torch.Tensor, targets: torch.Tensor
    ):
        device = targets.device
        b, c, h, w = img.shape
        # targets[:, 2:] *= torch.tensor((w, h, w, h), device=device)

        predictions = []
        detections = []

        for idx, pred in enumerate(preds):
            # Targets : (idx, cls, xn, yn, wn, hn)
            labels = targets[targets[:, 0] == idx, 1:]
            num_lbl, num_pred = labels.shape[0], pred.shape[0]
            img_orig_shape = img_infos[idx]
            correct = torch.zeros(num_pred, self.num_iou,
                                  dtype=torch.bool, device=device)

            self.seen += 1

            predn = pred.clone()
            scale_coords(
                img[idx].shape[1:], predn[:, :4], img_orig_shape
            )
            if num_pred == 0:
                if num_lbl:
                    self.stats.append(
                        (correct, *torch.zeros((2, 0),
                         device=device), labels[:, 0])
                    )
                continue

            if num_lbl:
                target_box = xywhn_to_xyxy(labels[:, 1:5])
                target_boxes = scale_coords(
                    img[idx].shape[1:], target_box, img_orig_shape)
                targetn = torch.cat((labels[:, 0:1], target_box), 1)
                # correct = self.process_batch(predn, targetn, self.iouv)

            # import cv2
            # import numpy as np
            # imgn = img[idx].cpu().numpy()
            # imgn = np.transpose(imgn, (1, 2, 0))
            # for bbox in target_boxes:
            #     bbox = bbox.cpu().numpy()
            #     bbox = list(map(int, bbox.tolist()))
            #     cv2.rectangle(
            #         imgn, bbox[:2], bbox[2:], (255, 0, 0), 2
            #     )
            # cv2.imshow("S", imgn)
            # cv2.waitKey(0)

            # Correct, conf, pred_cls, target_cls
            # self.stats.append((correct, pred[:4], pred[:, 5], labels[:, 0]))
            predictions.append(predn)
            detections.append(targetn)

        return torch.vstack(predictions), torch.vstack(detections)

    @staticmethod
    def process_batch(preds, labels, iouv):
        """
        Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = np.zeros((preds.shape[0], iouv.shape[0])).astype(bool)
        iou = box_iou(labels[:, 1:], preds[:, :4])
        correct_class = labels[:, 0:1] == preds[:, 5]
        for i in range(len(iouv)):
            # IoU > threshold and classes match
            x = torch.where((iou >= iouv[i]) & correct_class)
            if x[0].shape[0]:
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu(
                ).numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(
                        matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(
                        matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)
