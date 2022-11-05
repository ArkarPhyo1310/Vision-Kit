import torch
from torch import nn
from vision_kit.utils.bboxes import bbox_iou, bbox_overlaps
from vision_kit.utils.metrics import smooth_BCE


class YoloLoss:
    def __init__(self, hyp: dict = None) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.num_classes = hyp["model"]["num_classes"]
        self.hyp = hyp

        self.gr = 1.0
        self.cls_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp['cls_pw']], device="cuda"))
        self.obj_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([hyp['obj_pw']], device=self.device))

        self.cp, self.cn = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))

    def set_anchor(self, anchors: torch.Tensor):
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
                # iou = bbox_overlaps(
                #     pbox, tbox[idx], mode="ciou", is_aligned=True, box_format="cxcywh")
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
        anchor_interleave = torch.arange(
            num_anchors, device=self.device).float().view(num_anchors, 1).repeat(1, num_targets)

        targets = torch.cat(
            (targets.repeat(num_anchors, 1, 1), anchor_interleave[..., None]), 2)

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
