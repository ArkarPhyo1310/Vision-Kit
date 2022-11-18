from typing import List, Tuple

import torch
from torch import nn

from vision_kit.models.modules.blocks import (ELAN, SPPCSPC, Concat, ConvBnAct,
                                              MPx3Conv, RepConv)


class PAFPNELAN(nn.Module):
    def __init__(
        self,
        variant: str = "base",
        act: str = "silu",
    ) -> None:
        super().__init__()
        assert variant.lower() in ["tiny", "base", "x"], f"Not supported version: {variant}!"
        neck_cfg = {
            "base": {
                "in_chs": (512, 1024),
                "out_chs": (256, 512, 1024),
                "elan_depth": 4
            },
            "x": {
                "in_chs": (640, 1280),
                "out_chs": (320, 640, 1280),
                "elan_depth": 6
            }
        }

        in_chs = neck_cfg[variant.lower()]["in_chs"]
        out_chs = neck_cfg[variant.lower()]["out_chs"]
        depth = neck_cfg[variant.lower()]["elan_depth"]

        self.sppcspc = SPPCSPC(in_chs[1], out_chs[1], act=act)

        self.lateral_conv = ConvBnAct(
            in_chs[0], out_chs[0], kernel=1, stride=1,
            act=act
        )
        self.route_p4 = ConvBnAct(
            in_chs[1], out_chs[0], kernel=1, stride=1,
            act=act
        )
        self.lateral_elan = ELAN(
            in_chs[0], 256, out_chs[0],
            act=act, depth=depth
        )

        self.reduce_conv = ConvBnAct(
            int(in_chs[0] / 2), int(out_chs[0] / 2), 1, 1,
            act=act
        )
        self.route_p3 = ConvBnAct(
            in_chs[0], int(out_chs[0] / 2), 1, 1,
            act=act
        )
        self.reduce_elan = ELAN(
            int(in_chs[0] / 2), 128, int(out_chs[0] / 2),
            act=act, depth=depth
        )

        self.mp_3xconvs_1 = MPx3Conv(
            int(in_chs[0] / 4), int(out_chs[0] / 2), act=act
        )
        self.bu_elan1 = ELAN(
            in_chs[0], 256, out_chs[0], act=act, depth=depth
        )

        self.mp_3xconvs_2 = MPx3Conv(
            int(in_chs[0] / 2), out_chs[0], act=act
        )
        self.bu_elan2 = ELAN(
            in_chs[1], 512, out_chs[1], act=act, depth=depth
        )

        if variant.lower() == "base":
            self.pan_conv2 = RepConv(int(in_chs[0] / 4), out_chs[0], act=act)
            self.pan_conv1 = RepConv(int(in_chs[0] / 2), out_chs[1], act=act)
            self.pan_conv0 = RepConv(in_chs[0], out_chs[2], act=act)
        else:
            self.pan_conv2 = ConvBnAct(int(in_chs[0] / 4), out_chs[0], kernel=3, stride=1, act=act)
            self.pan_conv1 = ConvBnAct(int(in_chs[0] / 2), out_chs[1], kernel=3, stride=1, act=act)
            self.pan_conv0 = ConvBnAct(in_chs[0], out_chs[2], kernel=3, stride=1, act=act)

        self.concat = Concat()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = x

        x_sppcspc = self.sppcspc(p5)

        fpn_out1 = self.lateral_conv(x_sppcspc)
        f_out1 = self.upsample(fpn_out1)
        r_p4 = self.route_p4(p4)
        f_out1 = self.concat([r_p4, f_out1])
        f_out1 = self.lateral_elan(f_out1)

        fpn_out2 = self.reduce_conv(f_out1)
        f_out2 = self.upsample(fpn_out2)
        r_p3 = self.route_p3(p3)
        f_out2 = self.concat([r_p3, f_out2])
        pan_out2 = self.reduce_elan(f_out2)

        x_79, x_77 = self.mp_3xconvs_1(pan_out2)
        p_out1 = self.concat([x_79, x_77, f_out1])
        pan_out1 = self.bu_elan1(p_out1)

        x_92, x_90 = self.mp_3xconvs_2(pan_out1)
        p_out2 = self.concat([x_92, x_90, x_sppcspc])
        pan_out0 = self.bu_elan2(p_out2)

        pan_out2 = self.pan_conv2(pan_out2)
        pan_out1 = self.pan_conv1(pan_out1)
        pan_out0 = self.pan_conv0(pan_out0)

        return pan_out2, pan_out1, pan_out0


if __name__ == "__main__":
    neck = PAFPNELAN()
    x = [torch.rand(1, 512, 80, 80),
         torch.rand(1, 1024, 40, 40),
         torch.rand(1, 1024, 20, 20)]

    for y in neck(x):
        print(y.shape)
