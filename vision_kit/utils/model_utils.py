#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
import os
from copy import deepcopy
import random
from typing import Any, Dict

import numpy as np
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
        if t is nn.Conv2d:
            # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


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


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


def load_model_weight(model, checkpoint, logger):
    state_dict = checkpoint["state_dict"].copy()
    for k in checkpoint["state_dict"]:
        # convert average model weights
        if k.startswith("avg_model."):
            v = state_dict.pop(k)
            state_dict[k[4:]] = v
    # strip prefix of state_dict
    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    if list(state_dict.keys())[0].startswith("model."):
        state_dict = {k[6:]: v for k, v in state_dict.items()}

    model_state_dict = (
        model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    )

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                logger.log(
                    "Skip loading parameter {}, required shape{}, "
                    "loaded shape{}.".format(
                        k, model_state_dict[k].shape, state_dict[k].shape
                    )
                )
                state_dict[k] = model_state_dict[k]
        else:
            logger.log("Drop parameter {}.".format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            logger.log("No param {}.".format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    load_dict = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            print(
                "{} is not in the ckpt. Please double check and see if this is desired.".format(
                    key_model
                )
            )
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            print(
                "Shape of {} in checkpoint is {}, while shape of {} in model is {}.".format(
                    key_model, v_ckpt.shape, key_model, v.shape
                )
            )
            continue
        load_dict[key_model] = v_ckpt
    model.load_state_dict(load_dict, strict=False)
    return model


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model


def init_seeds(seed=0, deterministic=False):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    torch.backends.cudnn.benchmark = True  # for faster training
    if deterministic:  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['PYTHONHASHSEED'] = str(seed)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class ModelEmaV2(nn.Module):
    """ Model Exponential Moving Average V2
    Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e,
                     m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)