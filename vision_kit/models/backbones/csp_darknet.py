from typing import List, Tuple

import torch
from torch import nn
from vision_kit.models.modules.blocks import (SPP, SPPF, ConvBnAct,
                                              DWConvModule, Focus)
from vision_kit.models.modules.bottlenecks import C3Bottleneck


class CSPDarknet(nn.Module):
    def __init__(
        self,
        depth_mul: float,
        width_mul: float,
        act: str = "silu",
        depthwise: bool = False,
        with_focus: bool = False,
    ) -> None:
        super().__init__()

        Conv = DWConvModule if depthwise else ConvBnAct

        base_channels: int = int(width_mul * 64)
        base_depth: int = max(round(depth_mul * 3), 1)

        if with_focus:
            self.stem: Focus = Focus(3, base_channels, kernel=3, act=act)
        else:
            self.stem: ConvBnAct = ConvBnAct(
                3, base_channels,
                kernel=6, stride=2,
                padding=2
            )

        self.stage1: nn.Sequential = nn.Sequential(
            Conv(
                base_channels, base_channels * 2,
                kernel=3, stride=2
            ),
            C3Bottleneck(
                base_channels * 2, base_channels * 2,
                n=base_depth, depthwise=depthwise,
                act=act
            )
        )

        self.stage2: nn.Sequential = nn.Sequential(
            Conv(
                base_channels * 2, base_channels * 4,
                kernel=3, stride=2,
                act=act
            ),
            C3Bottleneck(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3 if with_focus else base_depth * 2,
                depthwise=depthwise,
                act=act
            )
        )

        self.stage3: nn.Sequential = nn.Sequential(
            Conv(
                base_channels * 4, base_channels * 8,
                kernel=3, stride=2,
                act=act
            ),
            C3Bottleneck(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, depthwise=depthwise,
                act=act
            )
        )

        if with_focus:
            self.stage4: nn.Sequential = nn.Sequential(
                Conv(
                    base_channels * 8, base_channels * 16,
                    kernel=3, stride=2,
                    act=act
                ),
                SPP(
                    base_channels * 16, base_channels * 16,
                    act=act
                ),
                C3Bottleneck(
                    base_channels * 16, base_channels * 16,
                    n=base_depth, shortcut=False,
                    depthwise=depthwise,
                    act=act
                )
            )
        else:
            self.stage4: nn.Sequential = nn.Sequential(
                Conv(
                    base_channels * 8, base_channels * 16,
                    kernel=3, stride=2,
                    act=act
                ),
                C3Bottleneck(
                    base_channels * 16, base_channels * 16,
                    n=base_depth, depthwise=depthwise,
                    act=act
                ),
                SPPF(
                    base_channels * 16, base_channels * 16,
                    kernel=5
                )
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c1: torch.Tensor = self.stem(x)
        c2: torch.Tensor = self.stage1(c1)
        c3: torch.Tensor = self.stage2(c2)
        c4: torch.Tensor = self.stage3(c3)
        c5: torch.Tensor = self.stage4(c4)

        return c3, c4, c5
