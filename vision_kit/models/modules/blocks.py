import math
from typing import Any

import torch
from torch import nn
from vision_kit.utils.model_utils import auto_pad, get_act_layer


class ConvBnAct(nn.Module):
    def __init__(
        self,
        ins: int, outs: int,
        kernel: int = 1, stride: int = 1,
        padding: int = None, groups: int = 1,
        act: str = "silu"
    ) -> None:
        super().__init__()
        self.conv: nn.Conv2d = nn.Conv2d(
            ins, outs, kernel, stride,
            auto_pad(kernel, padding),
            groups=groups,
            bias=False
        )
        self.bn: nn.BatchNorm2d = nn.BatchNorm2d(outs)
        self.act = get_act_layer(act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DWConvModule(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(
        self,
        ins: int, outs: int,
        kernel: int, stride: int = 1,
        act: str = "silu"
    ) -> None:
        super().__init__()
        self.dconv: ConvBnAct = ConvBnAct(
            ins, ins,
            kernel, stride=stride,
            groups=ins, act=act
        )
        self.pconv: ConvBnAct = ConvBnAct(
            ins, outs,
            kernel=1, stride=1,
            groups=1, act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.dconv(x)
        return self.pconv(x)


class DWConv(ConvBnAct):
    def __init__(
        self,
        ins: int, outs: int,
        kernel: int = 1, stride: int = 1,
        padding: int = None, groups: int = 1,
        act: str = "silu"
    ) -> None:
        super().__init__(
            ins, outs,
            kernel, stride,
            padding, math.gcd(ins, outs),
            act
        )


class DWConvTranspose2d(nn.ConvTranspose2d):
    def __init__(
        self,
        ins: int, outs: int,
        kernel: int, stride: int = 1,
        padding: int = 0, padding_outs: int = 0
    ) -> None:
        super().__init__(
            ins, outs,
            kernel, stride,
            padding, padding_outs,
            math.gcd(ins, outs)
        )


class SPP(nn.Module):
    """Spatial pyramid pooling layer"""

    def __init__(
        self,
        ins: int, outs: int,
        kernels: tuple = (5, 9, 13),
        act: str = "silu"
    ) -> None:
        super().__init__()
        hidden_chs: int = ins // 2

        self.conv1: ConvBnAct = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1,
            act=act
        )
        self.m: nn.ModuleList = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernels
            ]
        )
        conv2_chs: int = hidden_chs * (len(kernels) + 1)
        self.conv2: ConvBnAct = ConvBnAct(
            conv2_chs, outs,
            kernel=1, stride=1,
            act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.conv1(x)
        x: torch.Tensor = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x: torch.Tensor = self.conv2(x)
        return x


class Concat(nn.Module):
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dim = dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(x, self.dim)


class SPPF(nn.Module):
    def __init__(
        self,
        ins: int, outs: int,
        kernel: int = 5
    ) -> None:
        super().__init__()
        hidden_chs: int = ins // 2
        self.conv1: ConvBnAct = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1
        )
        self.conv2: ConvBnAct = ConvBnAct(
            hidden_chs * 4, outs,
            kernel=1, stride=1
        )
        self.max_pool: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=kernel, stride=1,
            padding=kernel // 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x: torch.Tensor = self.conv1(x)
        y1: Any = self.max_pool(x)
        y2: Any = self.max_pool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.max_pool(y2)), 1))


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(
        self,
        ins: int, outs: int,
        kernel: int = 1, stride: int = 1,
        act: str = "silu"
    ) -> None:
        super().__init__()
        self.conv: ConvBnAct = ConvBnAct(
            ins * 4, outs,
            kernel=kernel, stride=stride,
            act=act
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patch_top_left: torch.Tensor = x[..., ::2, ::2]
        patch_top_right: torch.Tensor = x[..., ::2, 1::2]
        patch_bot_left: torch.Tensor = x[..., 1::2, ::2]
        patch_bot_right: torch.Tensor = x[..., 1::2, 1::2]

        x: torch.Tensor = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right
            ), dim=1
        )

        return self.conv(x)
