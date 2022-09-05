from typing import List, Type, Union

import torch
from torch import nn

from .blocks import ConvBnAct, DWConvModule


class StandardBottleneck(nn.Module):
    def __init__(
        self,
        ins: int, outs: int,
        groups: int = 1,
        expansion: float = 0.5,
        act: str = "silu",
        shortcut: bool = True,
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden_chs: int = int(outs * expansion)
        Conv = DWConvModule if depthwise else ConvBnAct
        self.conv1: ConvBnAct = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1, groups=groups,
            act=act
        )
        self.conv2 = Conv(
            hidden_chs, outs,
            kernel=3, stride=1, groups=groups,
            act=act
        )
        self.use_add: bool = shortcut and ins == outs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y: torch.Tensor = self.conv2(self.conv1(x))
        if self.use_add:
            y: torch.Tensor = y + x
        return y


class C3Bottleneck(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        ins: int, outs: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        act: str = "silu",
        depthwise: bool = False,
    ) -> None:
        super().__init__()
        hidden_chs: int = int(outs * expansion)
        self.conv1: ConvBnAct = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1,
            act=act
        )
        self.conv2: ConvBnAct = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1,
            act=act
        )
        self.conv3: ConvBnAct = ConvBnAct(
            2 * hidden_chs, outs,
            kernel=1, stride=1,
            act=act
        )

        module_list: List[StandardBottleneck] = [
            StandardBottleneck(
                hidden_chs, hidden_chs, expansion=1.0,
                act=act,
                shortcut=shortcut, depthwise=depthwise,
            )
            for _ in range(n)
        ]

        self.m: nn.Sequential = nn.Sequential(*module_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1: torch.Tensor = self.conv1(x)
        x2: torch.Tensor = self.conv2(x)
        x1: torch.Tensor = self.m(x1)
        x: torch.Tensor = torch.cat((x1, x2), dim=1)
        return self.conv3(x)
