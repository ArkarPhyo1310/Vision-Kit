import math
from copy import deepcopy

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from vision_kit.models.backbones import v7Backbone
from vision_kit.models.heads import YoloV7Head
from vision_kit.models.modules.blocks import ConvBnAct, RepConv
from vision_kit.models.necks import PAFPNELAN
from vision_kit.utils.model_utils import (fuse_conv_and_bn, init_weights,
                                          load_ckpt)


class YOLOV7(nn.Module):
    def __init__(
        self,
        num_classes: int = 80,
        variant: str = "base",
        act: str = "silu",
        deploy: bool = True,
        export: bool = False
    ) -> None:
        super(YOLOV7, self).__init__()

        self.backbone: v7Backbone = v7Backbone(variant, act=act)
        self.neck: PAFPNELAN = PAFPNELAN(variant, act=act)
        self.head: YoloV7Head = YoloV7Head(variant=variant, num_classes=num_classes, deploy=deploy, export=export)

        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x

    def fuse(self) -> None:
        for m in self.modules():
            if type(m) is ConvBnAct and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, 'bn')
                m.forward = m.forward_fuse
            if isinstance(m, RepConv):
                m.fuse_repvgg_block()

    def get_optimizer(self, hyp_cfg: dict, max_epochs: int) -> tuple[SGD, LambdaLR]:
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        for k, v in self.named_modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d):
                pg0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)

            if hasattr(v, 'im'):
                if hasattr(v.im, 'implicit'):
                    pg0.append(v.im.implicit)
                else:
                    for iv in v.im:
                        pg0.append(iv.implicit)
            if hasattr(v, 'ia'):
                if hasattr(v.ia, 'implicit'):
                    pg0.append(v.ia.implicit)
                else:
                    for iv in v.ia:
                        pg0.append(iv.implicit)

        optimizer: SGD = SGD(pg0, lr=hyp_cfg.lr0, momentum=hyp_cfg.momentum, nesterov=True)
        optimizer.add_param_group({'params': pg1, 'weight_decay': hyp_cfg.weight_decay})
        optimizer.add_param_group({'params': pg2})

        def lf(x): return ((1 - math.cos(x * math.pi / max_epochs)) / 2) * (hyp_cfg.lrf - 1) + 1
        scheduler: LambdaLR = LambdaLR(optimizer=optimizer, lr_lambda=lf)

        return optimizer, scheduler

    @staticmethod
    def reparameterization(model: "YOLOV7", ckpt_path: str, exclude: list = []) -> "YOLOV7":
        ckpt_state_dict = torch.load(ckpt_path, map_location=next(model.parameters()).device) if isinstance(ckpt_path, str) else ckpt_path

        num_anchors = model.head.num_anchors
        exclude = exclude

        model = load_ckpt(model, ckpt_state_dict)

        if 'head.im.0.implicit' in ckpt_state_dict.keys():
            for i in range((model.head.num_classes + 5) * num_anchors):
                model.state_dict()['head.m.0.weight'].data[i, :, :, :] *= ckpt_state_dict['head.im.0.implicit'].data[:, i, ::].squeeze()
                model.state_dict()['head.m.1.weight'].data[i, :, :, :] *= ckpt_state_dict['head.im.1.implicit'].data[:, i, ::].squeeze()
                model.state_dict()['head.m.2.weight'].data[i, :, :, :] *= ckpt_state_dict['head.im.2.implicit'].data[:, i, ::].squeeze()
            model.state_dict()['head.m.0.bias'].data += ckpt_state_dict['head.m.0.weight'].mul(ckpt_state_dict['head.ia.0.implicit']).sum(1).squeeze()
            model.state_dict()['head.m.1.bias'].data += ckpt_state_dict['head.m.1.weight'].mul(ckpt_state_dict['head.ia.1.implicit']).sum(1).squeeze()
            model.state_dict()['head.m.2.bias'].data += ckpt_state_dict['head.m.2.weight'].mul(ckpt_state_dict['head.ia.2.implicit']).sum(1).squeeze()
            model.state_dict()['head.m.0.bias'].data *= ckpt_state_dict['head.im.0.implicit'].data.squeeze()
            model.state_dict()['head.m.1.bias'].data *= ckpt_state_dict['head.im.1.implicit'].data.squeeze()
            model.state_dict()['head.m.2.bias'].data *= ckpt_state_dict['head.im.2.implicit'].data.squeeze()
        else:
            print("Model is already reparameterized!")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from vision_kit.models.architectures import build_model
    cfg = OmegaConf.load("./configs/yolov7.yaml")
    x = torch.rand((1, 3, 640, 640))
    # model = build_model(cfg)
    model = YOLOV7()
    # model.eval()
    model.fuse()
    for key in model.state_dict().keys():
        print(key)
    # print(model)
