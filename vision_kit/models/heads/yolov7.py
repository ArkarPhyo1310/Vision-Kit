from typing import List, Tuple

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
        deploy: bool = False,
        export: bool = False
    ) -> None:  # detection layer
        super(YoloV7Head, self).__init__()
        if anchors is None:
            anchors: List[list[int]] = [
                [12, 16, 19, 36, 40, 28],  # P3/8
                [36, 75, 76, 55, 72, 146],  # P4/16
                [142, 110, 192, 243, 459, 401]
            ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.export: bool = export
        self.deploy: bool = deploy
        self.num_classes = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.num_det_layers = len(anchors)  # number of detection layers
        self.num_anchors = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.num_det_layers  # init grid

        self.stride: torch.Tensor = torch.tensor(stride, device=self.device)

        self.anchors = torch.tensor(anchors, device=self.device).float().view(self.num_det_layers, -1, 2)
        self.anchor_grid = self.anchors.clone().view(self.num_det_layers, 1, -1, 1, 1, 2)

        self.anchors /= self.stride.view(-1, 1, 1)
        self.anchors: torch.Tensor = check_anchor_order(self.anchors, self.stride)

        self.m: nn.ModuleList = nn.ModuleList(nn.Conv2d(x, self.no * self.num_anchors, 1)
                                              for x in in_chs)  # output conv

        if not self.deploy:
            self.ia: nn.ModuleList = nn.ModuleList(Implicit(x, ops="add") for x in in_chs)
            self.im: nn.ModuleList = nn.ModuleList(
                Implicit(self.no * self.num_anchors, ops="multiply") for _ in in_chs)

        init_bias(self.m, self.stride, self.num_anchors, self.num_classes)

    def forward(self, x: Tuple[torch.Tensor]) -> Tuple[torch.Tensor] | tuple[torch.Tensor, Tuple[torch.Tensor]]:
        # x = x.copy()  # for profiling
        z = []  # inference output
        x = list(x)
        for i in range(self.num_det_layers):
            if self.training or hasattr(self, "ia"):
                x[i] = self.m[i](self.ia[i](x[i]))  # conv
                x[i] = self.im[i](x[i])
            else:
                x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
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

        return x if self.training else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20) -> torch.Tensor:
        yv, xv = meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
