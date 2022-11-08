from typing import List

import torch
from torch import nn

from vision_kit.models.modules.blocks import Implicit
from vision_kit.utils.model_utils import (check_anchor_order, init_bias,
                                          meshgrid)


class YoloV7Head(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        anchors: list = None,
        in_chs: tuple = (256, 512, 1024),
        stride: tuple = (8., 16., 32.),
        training_mode: bool = False,
        export: bool = False
    ):  # detection layer
        super(YoloV7Head, self).__init__()
        if anchors is None:
            anchors: List[list[int]] = [
                [12, 16, 19, 36, 40, 28],  # P3/8
                [36, 75, 76, 55, 72, 146],  # P4/16
                [142, 110, 192, 243, 459, 401]
            ]
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.training_mode = training_mode
        self.num_classes = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.num_det_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_det_layers  # init grid

        self.stride: torch.Tensor = torch.tensor(stride, device=self.device)

        self.anchors = torch.tensor(anchors, device=self.device).float().view(self.num_det_layers, -1, 2)
        self.anchors /= self.stride.view(-1, 1, 1)
        self.anchors = check_anchor_order(self.anchors, self.stride)

        self.anchor_grid = self.anchors.clone().view(self.num_det_layers, 1, -1, 1, 1, 2)

        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.num_anchors, 1)
                               for x in in_chs)  # output conv

        if self.training_mode:
            self.ia = nn.ModuleList(Implicit(x, ops="add") for x in in_chs)
            self.im = nn.ModuleList(
                Implicit(self.no * self.num_anchors, ops="multiply") for _ in in_chs)

        self.export = export
        init_bias(self.m, self.stride, self.num_anchors, self.num_classes)

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        for i in range(self.num_det_layers):
            if hasattr(self, "ia") and hasattr(self, "im"):
                x[i] = self.m[i](self.ia[i](x[i]))  # conv
                x[i] = self.im[i](x[i])
            else:
                x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training_mode:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                if not torch.onnx.is_in_onnx_export():
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy, wh, conf = y.split((2, 2, self.num_classes + 1), 4)
                    xy = xy * (2. * self.stride[i]) + (self.stride[i] * (self.grid[i] - 0.5))  # new xy
                    wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training_mode else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    # def forward_train(self, x):
    #     # x = x.copy()  # for profiling
    #     z = []  # inference output
    #     self.training |= self.export
    #     for i in range(self.num_det_layers):
    #         x[i] = self.m[i](self.ia[i](x[i]))  # conv
    #         x[i] = self.im[i](x[i])
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, self.num_anchors, self.no, ny, nx).permute(
    #             0, 1, 3, 4, 2).contiguous()

    #         if not self.training:  # inference
    #             if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                 self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

    #             y = x[i].sigmoid()
    #             y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
    #                            self.grid[i]) * self.stride[i]  # xy
    #             y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
    #                 self.anchor_grid[i]  # wh
    #             z.append(y.view(bs, -1, self.no))

    #     return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    # def forward_fuse(self, x):
    #     # x = x.copy()  # for profiling
    #     z = []  # inference output

    #     for i in range(self.num_det_layers):
    #         x[i] = self.m[i](x[i])  # conv
    #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
    #         x[i] = x[i].view(bs, self.num_anchors, self.no, ny, nx).permute(
    #             0, 1, 3, 4, 2).contiguous()

    #         if not self.training:  # inference
    #             if self.grid[i].shape[2:4] != x[i].shape[2:4]:
    #                 self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

    #             y = x[i].sigmoid()
    #             if not torch.onnx.is_in_onnx_export():
    #                 y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
    #                 y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
    #             else:
    #                 # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
    #                 xy, wh, conf = y.split((2, 2, self.num_classes + 1), 4)  # new xy
    #                 xy = xy * (2. * self.stride[i]) + \
    #                     (self.stride[i] * (self.grid[i] - 0.5))
    #                 wh = wh ** 2 * (4 * self.anchor_grid[i].data)  # new wh
    #                 y = torch.cat((xy, wh, conf), 4)
    #             z.append(y.view(bs, -1, self.no))

    #     return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    # def fuse(self):
    #     print("IDetect.fuse")
    #     # fuse ImplicitA and Convolution
    #     for i in range(len(self.m)):
    #         c1, c2, _, _ = self.m[i].weight.shape
    #         c1_, c2_, _, _ = self.ia[i].implicit.shape
    #         self.m[i].bias += torch.matmul(self.m[i].weight.reshape(
    #             c1, c2), self.ia[i].implicit.reshape(c2_, c1_)).squeeze(1)

    #     # fuse ImplicitM and Convolution
    #     for i in range(len(self.m)):
    #         c1, c2, _, _ = self.im[i].implicit.shape
    #         self.m[i].bias *= self.im[i].implicit.reshape(c2)
    #         self.m[i].weight *= self.im[i].implicit.transpose(0, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
