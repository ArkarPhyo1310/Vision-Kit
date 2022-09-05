#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Dict

import torch
from torch import nn

_TORCH_VER = [int(x) for x in torch.__version__.split(".")[:2]]

__all__ = ["meshgrid"]

activations_methods: Dict[str, Any] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
    "hard_swish": nn.Hardswish,
    "none": nn.Identity
}


def get_act_layer(name: str) -> Any:
    assert name in activations_methods, f"Activation Name: {name} is not implemented yet!"

    if name == "none":
        return activations_methods[name]

    return activations_methods[name](inplace=True)


def init_weights(model: nn.Module):
    for m in model.modules():
        t = type(m)
        if t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03


def meshgrid(*tensors):
    if _TORCH_VER >= [1, 10]:
        return torch.meshgrid(*tensors, indexing="ij")
    else:
        return torch.meshgrid(*tensors)


def auto_pad(k: Any, p: str = None) -> str:
    if p is None:
        p: str = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def check_anchor_order(anchors: torch.Tensor, stride: torch.Tensor) -> torch.Tensor:
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    # mean anchor area per output layer
    a: torch.Tensor = anchors.prod(-1).mean(-1).view(-1)
    da: torch.Tensor = a[-1] - a[0]  # delta a
    ds: torch.Tensor = stride[-1] - stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        anchors[:] = anchors.flip(0)

    return anchors


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse convolution and batchnorm layers.
    check more info on https://tehnokv.com/posts/fusing-batchnorm-and-conv/

    Args:
        conv (nn.Conv2d): convolution to fuse.
        bn (nn.BatchNorm2d): batchnorm to fuse.

    Returns:
        nn.Conv2d: fused convolution behaves the same as the input conv and bn.
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(
        torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv
