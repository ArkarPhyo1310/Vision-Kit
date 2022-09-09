import math
from typing import List, Tuple

import torch
from torch import nn
from vision_kit.utils.bboxes import bbox_iou, bbox_overlaps
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
        device: str = "gpu",
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
            self.BCEcls = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([hyp['cls_pw']], device=self.device))
            self.BCEobj = nn.BCEWithLogitsLoss(
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

                y = x[i].sigmoid().to(self.device)
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

    def compute_loss(self, p: torch.Tensor, targets: torch.Tensor):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split(
                    (2, 2, 1, self.num_classes), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                # if self.sort_obj_iou:
                #     j = iou.argsort()
                #     b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.num_classes > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss

        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.num_anchors, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.num_det_layers):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch
