import math
from typing import List, Tuple

import torch
from torch import nn

from vision_kit.utils.model_utils import (check_anchor_order, init_bias,
                                          meshgrid)


class YoloV5Head(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        width: float = 1.00,
        anchors: list = None,
        in_chs: tuple = (256, 512, 1024),
        stride: list = [8., 16., 32.],
        training_mode: bool = False,
        export: bool = False
    ) -> None:
        super(YoloV5Head, self).__init__()
        if anchors is None:
            anchors: List[list[int]] = [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]
            ]

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

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

        self.training_mode = training_mode
        self.export: bool = export

        init_bias(self.m, self.stride, self.num_anchors, self.num_classes)

    def forward(self, x: torch.Tensor):
        z = []
        x = list(x)
        for i in range(self.num_det_layers):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.num_anchors, self.no, ny,
                             nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training_mode:
                if self.export or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(
                        nx, ny, i)

                y = x[i].sigmoid().to(self.device)
                if not self.export:
                    y[..., 0:2] = (y[..., 0:2] * 2 + self.grid[i]) * self.stride[i]  # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                else:
                    # y.tensor_split((2, 4, 5), 4)  # torch 1.8.0
                    xy, wh, conf = y.split((2, 2, self.num_classes + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, -1, self.no))

        return x if self.training_mode else (torch.cat(z, 1), ) if self.export else (torch.cat(z, 1), x)

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
