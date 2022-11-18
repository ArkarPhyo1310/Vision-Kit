from typing import List

import torch
from torch import nn

from vision_kit.models.modules.blocks import Concat, ConvBnAct, DWConvModule
from vision_kit.models.modules.bottlenecks import C3Bottleneck


class PAFPN(nn.Module):
    def __init__(
        self,
        depth_mul: float,
        width_mul: float,
        in_chs: list = [256, 512, 1024],
        act: str = "silu",
        depthwise: bool = False
    ) -> None:
        super().__init__()

        base_depth: int = max(round(depth_mul * 3), 1)
        out_chs: list = in_chs
        for idx, _ in enumerate(out_chs):
            out_chs[idx] = int(out_chs[idx] * width_mul)

        Conv = DWConvModule if depthwise else ConvBnAct

        self.concat: Concat = Concat()
        self.upsample: nn.Upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0: ConvBnAct = ConvBnAct(
            in_chs[2], out_chs[1],
            kernel=1, stride=1,
            act=act
        )
        self.C3_p4: C3Bottleneck = C3Bottleneck(
            int(2 * out_chs[1]), out_chs[1],
            n=base_depth, shortcut=False,
            act=act
        )

        self.reduce_conv1: ConvBnAct = ConvBnAct(
            out_chs[1], out_chs[0],
            kernel=1, stride=1,
            act=act
        )
        self.C3_p3: C3Bottleneck = C3Bottleneck(
            int(2 * out_chs[0]), out_chs[0],
            n=base_depth, shortcut=False,
            act=act
        )

        self.bu_conv2: ConvBnAct = Conv(
            in_chs[0], out_chs[0],
            kernel=3, stride=2,
            act=act
        )
        self.C3_n3: C3Bottleneck = C3Bottleneck(
            int(2 * out_chs[0]), out_chs[1],
            n=base_depth, shortcut=False,
            act=act
        )

        self.bu_conv1: ConvBnAct = Conv(
            out_chs[1], out_chs[1],
            kernel=3, stride=2,
            act=act
        )
        self.C3_n4: C3Bottleneck = C3Bottleneck(
            int(2 * out_chs[1]), out_chs[2],
            n=base_depth, shortcut=False,
            act=act
        )

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        c3, c4, c5 = x

        fpn_out0 = self.lateral_conv0(c5)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = self.concat([f_out0, c4])
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = self.concat([f_out1, c3])
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = self.concat([p_out1, fpn_out1])
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = self.concat([p_out0, fpn_out0])
        pan_out0 = self.C3_n4(p_out0)

        return pan_out2, pan_out1, pan_out0


if __name__ == "__main__":
    from vision_kit.utils.general import dw_multiple_generator
    x = [torch.rand(1, 192, 80, 80),
         torch.rand(1, 384, 40, 40),
         torch.rand(1, 768, 20, 20)]
    width, depth = dw_multiple_generator("m")
    neck = PAFPN(depth_mul=depth, width_mul=width)
    for y in neck(x):
        print(y.shape)
