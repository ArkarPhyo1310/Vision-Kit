from typing import Tuple

import numpy as np
import torch
from torchvision.ops.boxes import box_iou

from vision_kit.utils.bboxes import cxcywh_to_xyxy, xyxy_to_xywh
from vision_kit.utils.image_proc import scale_coords
from vision_kit.utils.metrics import smooth
from vision_kit.utils.table import RichTable


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px = np.linspace(0, 1, 1000)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros(
        (nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = np.interp(-px, -conf[i], precision[:, 0],
                          left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # points where x axis (recall) changes
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


class DetEvaluator:
    def __init__(
        self,
        class_labels: list,
        img_size: Tuple[int, int] = (640, 640),
        gt_json: str = None,
        label_format: str = "yolo"
    ) -> None:
        self.class_labels = class_labels
        self.img_sz = img_size
        self.class_ids = [i + 1 for i in range(len(self.class_labels))]
        self.gt_json = gt_json
        self.label_format = label_format

        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0
        self.mp = 0.0
        self.mr = 0.0
        self.map50 = 0.0
        self.map95 = 0.0
        self.stats = []

        # iou vector for mAP@0.5:0.95
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.num_iou = self.iouv.numel()
        self.seen = 0
        self.coco_data = []

    def evaluate(
        self, img: torch.Tensor, img_infos: list, idxs: list,
        preds: torch.Tensor, targets: torch.Tensor
    ):
        self.device = targets.device
        self.iouv = self.iouv.to(self.device)
        b, c, h, w = img.shape
        if self.label_format == "yolo":
            targets[:, 2:] *= torch.tensor((w, h, w, h), device=self.device)

        predictions = []
        detections = []

        for idx, pred in enumerate(preds):
            # Targets : (idx, cls, xn, yn, wn, hn)
            labels = targets[targets[:, 0] == idx, 1:]
            num_lbl, num_pred = labels.shape[0], pred.shape[0]
            img_orig_shape = img_infos[idx]
            img_id = idxs[idx]
            correct = torch.zeros(num_pred, self.num_iou,
                                  dtype=torch.bool, device=self.device)

            self.seen += 1

            predn = pred.clone()
            scale_coords(
                img[idx].shape[1:], predn[:, :4], img_orig_shape
            )
            if num_pred == 0:
                if num_lbl:
                    self.stats.append(
                        (correct, *torch.zeros((2, 0),
                         device=self.device), labels[:, 0])
                    )
                continue

            if num_lbl:
                target_box = cxcywh_to_xyxy(labels[:, 1:5])
                scale_coords(
                    img[idx].shape[1:], target_box, img_orig_shape)
                targetn = torch.cat((labels[:, 0:1], target_box), 1)
                correct = self.process_batch(predn, targetn, self.iouv)

            # Correct, conf, pred_cls, target_cls
            self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

            if self.gt_json is not None:
                # For COCO evaluation
                self.convert_to_coco(predn, img_id)

            predictions.append(predn)
            detections.append(targetn)

        return torch.vstack(predictions), torch.vstack(detections)

    def summarize(self, details_per_class: bool = False, do_coco_eval: bool = False):
        self.stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]
        rtable = None
        coco_eval_info = None
        num_classes = len(self.class_labels)
        if len(self.stats) and self.stats[0].any():
            true_pos, false_pos, self.precision, self.recall, self.f1, ap, ap_class = ap_per_class(
                *self.stats)
            ap50, ap = ap[:, 0], ap.mean(1)
            self.mp, self.mr, self.map50, self.map95 = self.precision.mean(
            ), self.recall.mean(), ap50.mean(), ap.mean()

        if details_per_class:
            rtable = RichTable(title="Details Per Class")
            # number of targets per class
            num_targets = np.bincount(
                self.stats[3].astype(int), minlength=num_classes)
            table_content = []
            for i, c in enumerate(ap_class):
                table_content.append(
                    [
                        self.class_labels[int(c)],
                        self.seen,
                        num_targets[c],
                        round(self.precision[i], 3),
                        round(self.recall[i], 3),
                        round(ap50[i], 3),
                        round(ap[i], 3)
                    ]
                )

            rtable.add_headers(
                ["Class", "Images", "Num_Targets", "Precision", "Recall", "mAP@.5", "mAP@.5:.95"])
            rtable.add_content(table_content)

        if self.gt_json is not None and do_coco_eval:
            coco_eval_info = self.coco_evaluate()

        self.seen = 0
        self.stats.clear()
        self.coco_data.clear()

        return self.map50, self.map95, rtable, coco_eval_info

    def convert_to_coco(self, pred, img_id):
        pred_output = pred.cpu()
        bboxes = pred_output[:, 0:4]
        scores = pred_output[:, 4]
        cls = pred_output[:, 5]
        bboxes = xyxy_to_xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            label = self.class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
            }  # COCO json format
            self.coco_data.append(pred_data)

    def coco_evaluate(self):
        import contextlib
        import io
        import json
        import tempfile

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        info = ""
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(self.coco_data) > 0:
            cocoGt = COCO(self.gt_json)
            _, tmp = tempfile.mkstemp()
            json.dump(self.coco_data, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)

            cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info = redirect_string.getvalue()

        return info

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
                matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).detach().cpu(
                ).numpy()  # [label, detect, iou]
                if x[0].shape[0] > 1:
                    matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(
                        matches[:, 1], return_index=True)[1]]
                    # matches = matches[matches[:, 2].argsort()[::-1]]
                    matches = matches[np.unique(
                        matches[:, 0], return_index=True)[1]]
                correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=preds.device)
