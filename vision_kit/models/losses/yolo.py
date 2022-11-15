from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import box_iou

from vision_kit.utils.bboxes import bbox_iou, bbox_overlaps, xywh_to_xyxy
from vision_kit.utils.metrics import smooth_BCE


class YoloLoss:
    def __init__(self, num_classes: int, hyp: dict = None) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes: int = num_classes
        self.hyp: dict = hyp

        self.gr: float = 1.0
        self.cls_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['cls_pw']], device=self.device))
        self.obj_loss: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp['obj_pw']], device=self.device))

        self.cp, self.cn = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))

    def set_anchor(self, anchors: torch.Tensor) -> None:
        self.anchors = anchors
        self.num_det_layers = len(anchors)
        self.num_anchors = len(anchors[0].view(-1)) // 2
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            self.num_det_layers, [4.0, 1.0, 0.25, 0.06, 0.02])

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor):
        loss_cls = torch.zeros(1, device=self.device)
        loss_box = torch.zeros(1, device=self.device)
        loss_obj = torch.zeros(1, device=self.device)

        tcls, tbox, inds, anchs = self.build_target(preds, targets)

        for idx, pred in enumerate(preds):
            # image, anchor, gridy, gridx
            b, a, gy, gx = inds[idx]
            tobj = torch.zeros(
                pred.shape[:4], dtype=pred.dtype, device=self.device)
            num_targets = b.shape[0]
            if num_targets:
                # target-subset of predictions
                pxy, pwh, _, pcls = pred[b, a, gy, gx].split(
                    (2, 2, 1, self.num_classes), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchs[idx]
                pbox = torch.cat((pxy, pwh), 1)
                # iou = bbox_overlaps(pbox, tbox[idx], mode="ciou", is_aligned=True, box_format="cxcywh")
                iou = bbox_iou(pbox, tbox[idx], CIoU=True).squeeze()
                loss_box += (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gy, gx] = iou

                if self.num_classes > 1:
                    t = torch.full_like(pcls, self.cn, device=self.device)
                    t[range(num_targets), tcls[idx]] = self.cp
                    loss_cls += self.cls_loss(pcls, t)

            objx = self.obj_loss(pred[..., 4], tobj)
            loss_obj += objx * self.balance[idx]

        loss_box *= self.hyp['box']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (loss_box + loss_obj + loss_cls) * bs, torch.cat((loss_box, loss_obj, loss_cls)).detach()

    def build_target(self, preds: torch.Tensor, targets: torch.Tensor):
        num_anchors, num_targets = self.num_anchors, targets.shape[0]
        tcls, tbox, indices, anchs = [], [], [], []

        gain = torch.ones(7, device=self.device)
        anchor_interleave = torch.arange(num_anchors, device=self.device).float().view(num_anchors, 1).repeat(1, num_targets)

        targets = torch.cat((targets.repeat(num_anchors, 1, 1), anchor_interleave[..., None]), 2)

        bias = 0.5
        offset = torch.tensor(
            [
                [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
            ],
            device=self.device
        ).float() * bias

        for i in range(self.num_det_layers):
            anchors, shape = self.anchors[i], preds[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            # Match targets to anchors
            match_target = targets * gain
            if num_targets:
                # Matches
                ratio = match_target[..., 4:6] / anchors[:, None]
                # Compare with predefined anchor target
                compare = torch.max(
                    ratio, 1 / ratio).max(2)[0] < self.hyp['anchor_t']

                match_target = match_target[compare]

                # Offsets
                gxy = match_target[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < bias) & (gxy > 1)).T
                l, m = ((gxi % 1 < bias) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                match_target = match_target.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                match_target = targets[0]
                offsets = 0

            # define
            # (image, class), grid xy, grid wh, anchors
            bc, gxy, gwh, a = match_target.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            # append
            indices.append(
                (b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1))
            )
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anchs.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anchs


class YoloLossOTA:
    def __init__(
        self,
        num_classes: int = 80,
        hyp_cfg: dict = None,
    ) -> None:
        super().__init__()
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes: int = num_classes
        self.hyp: dict = hyp_cfg

        # Define criteria
        self.bce_cls: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp_cfg["cls_pw"]], device=self.device))
        self.bce_obj: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([hyp_cfg["obj_pw"]], device=self.device))

        # Class label smoothing # eqn 3 https://arxiv.org/pdf/1902.04103.pdf
        # positive and negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=hyp_cfg.get("label_smoothing", 0.0))

        self.gr: float = 1.0

    def set_anchor(self, anchors: torch.Tensor, stride: torch.Tensor = None) -> None:
        self.anchors: torch.Tensor = anchors
        self.stride: torch.Tensor = stride
        self.num_det_layers: int = len(anchors)
        self.num_anchors: int = len(anchors[0].view(-1)) // 2
        self.balance: list[float] = {3: [4.0, 1.0, 0.4]}.get(
            self.num_det_layers, [4.0, 1.0, 0.25, 0.06, 0.02])

    def __call__(self, preds, targets, imgs) -> Any:
        loss_cls: torch.Tensor = torch.zeros(1, device=self.device)
        loss_box: torch.Tensor = torch.zeros(1, device=self.device)
        loss_obj: torch.Tensor = torch.zeros(1, device=self.device)

        bs, as_, gjs, gis, targets, anchors = self.build_targets(preds, targets, imgs)
        pre_gen_gains = [torch.tenso(pred.shape, device=self.device)[[3, 2, 3, 2]] for pred in preds]

        # Losses
        for i, pred_i in enumerate(preds):  # layer index, layer predictions
            b, a, gj, gi = bs[i], as_[i], gjs[i], gis[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pred_i[..., 0], device=self.device)   # target obj

            nt = b.shape[0]  # number of targets
            if nt:
                pred_subset = pred_i[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                grid = torch.stack([gi, gj], dim=1)
                pxy = pred_subset[:, :2].sigmoid() * 2 - 0.5
                pwh = (pred_subset[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat([pxy, pwh], dim=1)

                selected_tbox = targets[i][:, 2:6] * pre_gen_gains[i]
                selected_tbox[:, :2] -= grid

                # IoU loss
                iou = bbox_overlaps(pbox.T, selected_tbox, box_format="xywh", mode="ciou", is_aligned=True)
                loss_box += (1.0 - iou).mean()

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * iou.detach().clamp(0).type(tobj.dtype)

                # Classification
                selected_tcls = targets[i][:, 1].long()
                if self.num_classes > 1:
                    t = torch.full_like(pred_subset[:, 5:], self.cn, device=self.device)
                    t[range(nt), selected_tcls] = self.cp
                    loss_cls += self.bce_cls(pred_subset[:, 5:], t)

            obji = self.bce_obj(pred_i[..., 4], tobj)
            loss_obj += obji * self.balance[i]

        loss_box *= self.hyp['box']
        loss_obj *= self.hyp['obj']
        loss_cls *= self.hyp['cls']

        bs = tobj.shape[0]
        loss = loss_box + loss_cls + loss_obj
        return loss * bs, torch.cat([loss_box, loss_obj, loss_cls, loss]).detach()

    def build_targets(self, preds, targets, imgs):
        inds, anch = self.find_3_positive(preds, targets)

        matching_bs = [[] for _ in preds]
        matching_as = [[] for _ in preds]
        matching_gjs = [[] for _ in preds]
        matching_gis = [[] for _ in preds]
        matching_gts = [[] for _ in preds]
        matching_acs = [[] for _ in preds]

        nl = len(preds)
        for batch_idx in range(preds[0].shape[0]):
            b_idx = targets[:, 0] == batch_idx
            this_target = targets[b_idx]
            if this_target.shape[0] == 0:
                continue

            txywh = this_target[:, 2:6] * imgs[batch_idx].shape[1]
            txyxy = xywh_to_xyxy(txywh)

            pxyxys = []
            p_cls = []
            p_obj = []
            from_which_layer = []
            all_b = []
            all_a = []
            all_gj = []
            all_gi = []
            all_ac = []

            for i, pred_i in enumerate(preds):
                b, a, gj, gi = inds[i]
                idx = b == batch_idx
                b, a, gj, gi = b[idx], a[idx], gj[idx], gi[idx]
                ac = anch[i][idx]

                all_b.append(b)
                all_a.append(a)
                all_gj.append(gj)
                all_gi.append(gi)
                all_ac.append(ac)
                from_which_layer.append(torch.ones(size=(len(b), )) * i)

                fg_pred = pred_i[b, a, gj, gi]
                p_obj.append(fg_pred[:, 4:5])
                p_cls.append(fg_pred[:, 5:])

                grid = torch.stack([gi, gj], dim=1)
                pxy = (fg_pred[:, :2].sigmoid() * 2 - 0.5 + grid) * self.stride[i]
                pwh = (fg_pred[:, 2:4].sigmoid() * 2) ** 2 * ac * self.stride[i]
                pxywh: torch.Tensor = torch.cat([pxy, pwh], dim=-1)
                pxyxy: torch.Tensor = xywh_to_xyxy(pxywh)
                pxyxys.append(pxyxy)

            pxyxys = torch.cat(pxyxys, dim=0)
            if pxyxys.shape[0] == 0:
                continue

            p_obj = torch.cat(p_obj, dim=0)
            p_cls = torch.cat(p_cls, dim=0)
            all_b = torch.cat(all_b, dim=0)
            all_a = torch.cat(all_a, dim=0)
            all_gj = torch.cat(all_gj, dim=0)
            all_gi = torch.cat(all_gi, dim=0)
            all_ac = torch.cat(all_ac, dim=0)
            from_which_layer = torch.cat(from_which_layer, dim=0)

            pairwise_iou = box_iou(txyxy, pxyxys)
            pairwise_iou_loss = -torch.log(pairwise_iou + 1e-8)

            top_k, _ = torch.topk(pairwise_iou, min(10, pairwise_iou.shape[1]), dim=1)
            dynamic_k = torch.clamp(top_k.sum(1).int(), min=1)

            gt_cls_per_img = (
                F.one_hot(this_target[:, 1].to(torch.int64), self.num_classes)
                .float()
                .unsqueeze(1)
                .repeat(1, pxyxys.shape[0], 1)
            )

            num_gt = this_target.shape[0]
            cls_preds_ = (
                p_cls.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * p_obj.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            y = cls_preds_.sqrt()
            pairwise_cls_loss = F.binary_cross_entropy_with_logits(
                torch.log(y / (1 - y)), gt_cls_per_img, reduction="none"
            ).sum(-1)
            del cls_preds_

            cost = pairwise_cls_loss + 3.0 * pairwise_iou_loss
            matching_matrix = torch.zeros_like(cost)

            for gt_idx in range(num_gt):
                _, pos_idx = torch.topk(cost[gt_idx], k=dynamic_k[gt_idx].item(), largest=False)
                matching_matrix[gt_idx][pos_idx] = 1.0
            del top_k, dynamic_k

            anchor_matching_gt = matching_matrix.sum(0)
            if (anchor_matching_gt > 1).sum(0) > 0:
                _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
                matching_matrix[:, anchor_matching_gt > 1] *= 0.0
                matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            fg_mask_inboxes = matching_matrix.sum(0) > 0.0
            matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)

            from_which_layer = from_which_layer[fg_mask_inboxes]
            all_b = all_b[fg_mask_inboxes]
            all_a = all_a[fg_mask_inboxes]
            all_gj = all_gj[fg_mask_inboxes]
            all_gi = all_gi[fg_mask_inboxes]
            all_ac = all_ac[fg_mask_inboxes]

            this_target = this_target[matched_gt_inds]

            for i in range(nl):
                layer_idx = from_which_layer == i
                matching_bs[i].append(all_b[layer_idx])
                matching_as[i].append(all_a[layer_idx])
                matching_gjs[i].append(all_gj[layer_idx])
                matching_gis[i].append(all_gi[layer_idx])
                matching_acs[i].append(all_ac[layer_idx])
                matching_gts[i].append(this_target[layer_idx])

        for i in range(nl):
            if matching_gts[i] != []:
                matching_bs[i] = torch.cat(matching_bs[i], dim=0)
                matching_as[i] = torch.cat(matching_as[i], dim=0)
                matching_gjs[i] = torch.cat(matching_gjs[i], dim=0)
                matching_gis[i] = torch.cat(matching_gis[i], dim=0)
                matching_acs[i] = torch.cat(matching_acs[i], dim=0)
                matching_gts[i] = torch.cat(matching_gts[i], dim=0)
            else:
                matching_bs[i] = torch.tensor([], device=self.device, dtype=torch.int64)
                matching_as[i] = torch.tensor([], device=self.device, dtype=torch.int64)
                matching_gjs[i] = torch.tensor([], device=self.device, dtype=torch.int64)
                matching_gis[i] = torch.tensor([], device=self.device, dtype=torch.int64)
                matching_gts[i] = torch.tensor([], device=self.device, dtype=torch.int64)
                matching_acs[i] = torch.tensor([], device=self.device, dtype=torch.int64)

        return matching_bs, matching_as, matching_gjs, matching_gis, matching_gts, matching_acs

    def find_3_positives(self, preds, targets):
        na, nt = self.num_anchors, targets.shape[0]
        inds, anch = [], []
        # Normalized to gridspace gain
        gain: torch.Tensor = torch.ones(7, device=self.device).long()
        # Same as .repeat_interleave(nt)
        anchor_interleave = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)
        # Append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), anchor_interleave[..., None]), 2)

        bias = 0.5
        offset = torch.tensor([
            [0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]
        ], device=self.device).float() * bias

        for i in range(self.num_det_layers):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(preds[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            match_target = targets * gain
            if nt:
                wh_ratio = match_target[..., 4:6] / anchors[:, None]
                j = torch.max(wh_ratio, 1 / wh_ratio).max(2)[0] < self.hyp["anchor_t"]
                t = match_target[j]

                # Offsets
                gxy = match_target[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k, = ((gxy % 1 < bias) & (gxy > 1)).T
                l, m, = ((gxy % 1 < bias) & (gxy > 1)).T
                j = torch.stack(torch.ones_like(j), j, k, l, m)
                match_target = match_target.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + offset[:, None])[j]
            else:
                match_target = targets[0]
                offsets = 0

            # Define
            b, c = match_target[:, :2].long().T  # image, class
            gxy = match_target[:, 2:4]                   # grid xy
            gwh = match_target[:, 4:6]                   # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T                       # grid xy indices

            # Append
            a = match_target[:, 6].long()               # anchor indices
            inds.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
            anch.append(anchors[a])

        return inds, anch
