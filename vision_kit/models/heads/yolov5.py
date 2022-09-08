import math
from typing import List, Tuple

import torch
from torch import nn
from vision_kit.utils.bboxes import bbox_overlaps
from vision_kit.utils.loss_utils import smooth_BCE
from vision_kit.utils.model_utils import check_anchor_order, meshgrid


class YOLOV5Head(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        width: float = 1.00,
        anchors: list = None,
        in_chs: tuple = (256, 512, 1024),
        stride: list = [8., 16., 32.],
        hyp: dict = None,
        device: str = "cpu",
        inplace: bool = True,
        onnx_export: bool = False,
        training: bool = True
    ) -> None:
        super(YOLOV5Head, self).__init__()
        if anchors is None:
            anchors: List[list[int]] = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]

        self.device: str = torch.device("cuda" if device == "gpu" else "cpu")

        self.num_classes: int = num_classes
        self.no: int = num_classes + 5  # number of outputs per anchor
        self.num_det_layers: int = len(anchors)  # number of detection layers
        self.num_anchors: int = len(anchors[0]) // 2  # number of anchors
        self.grid: list[torch.Tensor] = [
            torch.zeros(1)] * self.num_det_layers  # init_grid
        self.anchor_grid: list[torch.Tensor] = [torch.zeros(1)] * \
            self.num_det_layers  # init_anchor_grid

        self.stride: torch.Tensor = torch.tensor(stride, device=self.device)

        self.anchors: torch.Tensor = torch.tensor(anchors, device=self.device).float().view(
            self.num_det_layers, -1, 2)
        self.anchors /= self.stride.view(-1, 1, 1)
        self.anchors = check_anchor_order(self.anchors, self.stride)

        self.m: nn.ModuleList = nn.ModuleList(
            nn.Conv2d(int(x * width), self.no * self.num_anchors, 1)
            for x in in_chs
        )

        self.inplace: bool = inplace
        self.onnx_export: bool = onnx_export

        # loss
        if training:
            self.gr = 1.0
            self.hyp = hyp
            self.cls_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([hyp['cls_pw']], device=self.device))
            self.obj_loss = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([hyp['obj_pw']], device=self.device))

            self.cp, self.cn = smooth_BCE(eps=hyp.get('label_smoothing', 0.0))

            self.balance = {3: [4.0, 1.0, 0.4]}.get(
                self.num_det_layers, [4.0, 1.0, 0.25, 0.06, 0.02])

        self._init_biases()

    def _init_biases(self, cf=None):
        for m, s in zip(self.m, self.stride):
            b = m.bias.view(self.num_anchors, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (self.num_classes - 0.999999)
                                      ) if cf is None else torch.log(cf / cf.sum())
            m.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        z = []
        x = list(x)
        for i in range(self.num_det_layers):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.no, ny,
                             nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.onnx_export or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(
                        nx, ny, i)

                y = x[i].sigmoid()
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2 +
                                   self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                        self.anchor_grid[i]  # wh
                else:
                    # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy, wh, conf = y.split((2, 2, self.num_classes + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), ) if self.onnx_export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0) -> Tuple[torch.Tensor, torch.Tensor]:
        d: torch.device = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = (1, self.num_anchors, ny, nx, 2)
        y: torch.Tensor = torch.arange(ny, device=d, dtype=t)
        x: torch.Tensor = torch.arange(nx, device=d, dtype=t)
        yv, xv = meshgrid(y, x)
        # add grid offset, i.e. y = 2.0 * x - 0.5
        grid: torch.Tensor = torch.stack((xv, yv), 2).expand(shape) - 0.5
        anchor_grid: torch.Tensor = (self.anchors[i] * self.stride[i]
                                     ).view((1, self.num_anchors, 1, 1, 2)).expand(shape)

        return grid, anchor_grid

    def compute_loss(self, preds: torch.Tensor, targets: torch.Tensor):
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
                iou = bbox_overlaps(
                    pbox, tbox[idx], mode="ciou", is_aligned=True, box_format="cxcywh")
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
