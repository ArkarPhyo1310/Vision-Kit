import math
from typing import Any, List

import numpy as np
import torch
from torch import nn

from vision_kit.utils.model_utils import (auto_pad, fuse_conv_and_bn,
                                          get_act_layer)


class ConvBn(nn.Module):
    def __init__(
        self,
        ins: int, outs: int,
        kernel: int = 1, stride: int = 1,
        padding: int = None, groups: int = 1
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            ins, outs, kernel_size=kernel, stride=stride,
            padding=padding, groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(outs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


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


class Concat(nn.Module):
    def __init__(self, dimension: int = 1) -> None:
        super().__init__()
        self.dim = dimension

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(x, self.dim)


class MP(nn.Module):
    def __init__(self, kernel: int = 2) -> None:
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=kernel, stride=kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mp(x)


class SP(nn.Module):
    def __init__(self, kernel: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.mp = nn.MaxPool2d(
            kernel_size=kernel, stride=stride, padding=kernel//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mp(x)


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
        y1: torch.Tensor = self.max_pool(x)
        y2: torch.Tensor = self.max_pool(y1)
        return self.conv2(torch.cat((x, y1, y2, self.max_pool(y2)), 1))


class SPPCSPC(nn.Module):
    def __init__(
        self,
        ins: int, outs: int,
        groups: int = 1, epsilon: float = 0.5,
        kernel: tuple = (5, 9, 13), act: str = "silu"
    ) -> None:
        super().__init__()
        hidden_chs = int(2 * outs * epsilon)
        self.conv1 = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1,
            groups=groups, act=act
        )
        self.conv2 = ConvBnAct(
            ins, hidden_chs,
            kernel=1, stride=1,
            groups=groups, act=act
        )
        self.conv3 = ConvBnAct(
            hidden_chs, hidden_chs,
            kernel=3, stride=1,
            groups=groups, act=act
        )
        self.conv4 = ConvBnAct(
            hidden_chs, hidden_chs,
            kernel=1, stride=1,
            groups=groups, act=act
        )
        self.conv5 = ConvBnAct(
            4 * hidden_chs, hidden_chs,
            kernel=1, stride=1,
            groups=groups, act=act
        )
        self.conv6 = ConvBnAct(
            hidden_chs, hidden_chs,
            kernel=3, stride=1,
            groups=groups, act=act
        )
        self.conv7 = ConvBnAct(
            2 * hidden_chs, outs,
            kernel=1, stride=1,
            groups=groups, act=act
        )

        self.mp_modules = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2)
                for x in kernel
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv4(self.conv3(self.conv1(x)))
        y1 = self.conv6(self.conv5(torch.cat([x1] + [mp(x1) for mp in self.mp_modules], 1)))
        y2 = self.conv2(x)

        return self.conv7(torch.cat((y1, y2), dim=1))


class RepConv(nn.Module):
    """Represented Convolution"""

    def __init__(
        self,
        ins: int, outs: int,
        kernel: int = 3, stride: int = 1,
        padding: int = None, groups: int = 1,
        act: str = "silu", deploy: bool = False
    ) -> None:
        super().__init__()

        assert kernel == 3
        assert auto_pad(kernel, padding) == 1

        padding_11 = auto_pad(kernel, padding) - kernel // 2

        self.deploy = deploy
        self.groups = groups
        self.ins = ins
        self.outs = outs
        self.act = get_act_layer(act)

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                ins, outs, kernel_size=kernel, stride=stride,
                padding=auto_pad(kernel, padding), groups=groups,
                bias=True
            )
        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=ins) if ins == outs and stride == 1 else None
            self.rbr_dense = ConvBn(ins, outs, kernel, stride, auto_pad(kernel, padding), groups=groups)
            self.rbr_1x1 = ConvBn(ins, outs, kernel=1, stride=stride, padding=padding_11, groups=groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(x))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(x)

        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + id_out)

    def fuse_conv_bn(self, conv, bn) -> nn.Conv2d:
        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         padding=conv.padding,
                         dilation=conv.dilation,
                         groups=conv.groups,
                         bias=True,
                         padding_mode=conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self) -> None:
        if self.deploy:
            return

        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense.conv, self.rbr_dense.bn)
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1.conv, self.rbr_1x1.bn)

        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])

        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            identity_conv_1x1 = nn.Conv2d(
                in_channels=self.ins,
                out_channels=self.outs,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=self.groups,
                bias=False
            )
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.data.unsqueeze(2).unsqueeze(3)

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functaionl.pad(identity_conv_1x1.weight, [1, 1, 1, 1])
        else:
            bias_identity_expanded = torch.nn.Parameter(torch.zeros_like(rbr_1x1_bias))
            weight_identity_expanded = torch.nn.Parameter(torch.zeros_like(weight_1x1_expanded))

        self.rbr_dense.weight = torch.nn.Parameter(
            self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded
        )
        self.rbr_dense.bias = torch.nn.Parameter(
            self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded
        )

        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


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


class ELAN(nn.Module):
    def __init__(
        self,
        ins: int, hidden_chs: int, outs: int,
        act: str = "silu", depth: int = 2
    ) -> None:
        super().__init__()

        assert depth % 2 == 0, "Depth is not multiple of 2."
        chs_mul = 5 if depth == 6 else 4

        self.hidden_chs = hidden_chs
        self.outs = outs

        self.conv1 = ConvBnAct(ins, hidden_chs, act=act)
        self.conv2 = ConvBnAct(ins, hidden_chs, act=act)

        if hidden_chs == outs:
            hidden_ch1 = hidden_chs
            hidden_ch2 = int(hidden_chs / 2)
        else:
            hidden_ch1 = hidden_ch2 = hidden_chs

        if depth >= 2:
            self.conv3 = ConvBnAct(
                hidden_ch1, hidden_ch2, kernel=3, stride=1, act=act
            )
            self.conv4 = ConvBnAct(
                hidden_ch2, hidden_ch2, kernel=3, stride=1, act=act
            )

        if depth >= 4:
            self.conv5 = ConvBnAct(
                hidden_ch2, hidden_ch2, kernel=3, stride=1, act=act
            )
            self.conv6 = ConvBnAct(
                hidden_ch2, hidden_ch2, kernel=3, stride=1, act=act
            )

        if depth >= 6:
            self.conv7 = ConvBnAct(
                hidden_ch2, hidden_ch2, kernel=3, stride=1, act=act
            )
            self.conv8 = ConvBnAct(
                hidden_ch2, hidden_ch2, kernel=3, stride=1, act=act
            )

        self.concat = Concat()
        self.last_conv = ConvBnAct(hidden_chs * chs_mul, outs, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)

        if hasattr(self, "conv3"):
            x3 = self.conv3(x2)
            x4 = self.conv4(x3)
            concat_x = [x4, x3, x2, x1]

        if hasattr(self, "conv5"):
            x5 = self.conv5(x4)
            x6 = self.conv6(x5)
            concat_x = [x6, x5, x3, x1]
            if self.hidden_chs == self.outs:
                concat_x = [x6, x5, x4, x3, x2, x1]

        if hasattr(self, "conv7"):
            x7 = self.conv7(x6)
            x8 = self.conv8(x7)
            concat_x = [x8, x7, x5, x3, x1]

        return self.last_conv(self.concat(concat_x))


class MPx3Conv(nn.Module):
    def __init__(
        self, ins: int, outs: int, act: str = "silu"
    ) -> None:
        super().__init__()
        self.max_pool = MP()

        self.conv1 = ConvBnAct(ins, outs, 1, 1, act=act)
        self.conv2 = ConvBnAct(ins, outs, 1, 1, act=act)
        self.conv3 = ConvBnAct(outs, outs, 3, 2, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mp = self.max_pool(x)
        x1 = self.conv1(x_mp)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)

        return x3, x1


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x


class Implicit(nn.Module):
    def __init__(
        self,
        channel: int, ops: str = "add",
        mean: float = None, std: float = .02
    ) -> None:
        super().__init__()
        assert ops.lower() in ["add", "multiply"], "Not Implemented Operation!"

        self.channel = channel
        self.ops = ops.lower()
        self.mean = mean if mean else 0 if self.ops == "add" else 1
        self.std = std

        weight = torch.zeros(1, channel, 1, 1) if self.ops == "add" else torch.ones(1, channel, 1, 1)

        self.implicit = nn.Parameter(weight)
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ops == "add":
            return self.implicit + x
        else:
            return self.implicit * x


if __name__ == "__main__":
    i_x2 = torch.ones((1, 64, 5, 5))
    s_x2 = ELAN(64, 32, 64)
    print(s_x2)
    print(s_x2(i_x2).shape)

    i_x4 = torch.ones((1, 128, 5, 5))
    s_x4 = ELAN(128, 64, 256, depth=4)
    print(s_x4)
    print(s_x4(i_x4).shape)

    i_x6 = torch.ones((1, 160, 5, 5))
    s_x6 = ELAN(160, 64, 320, depth=6)
    print(s_x6)
    print(s_x6(i_x6).shape)
