from copy import deepcopy

import torch
from torch import nn

from vision_kit.models.backbones import EELAN
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
        training_mode: bool = True,
        export: bool = False
    ) -> None:
        super(YOLOV7, self).__init__()

        self.training_mode = training_mode

        self.backbone = EELAN(variant, act=act)
        self.neck = PAFPNELAN(variant, act=act)
        self.head = YoloV7Head(num_classes=num_classes, export=export, training_mode=training_mode)

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

    @staticmethod
    def reparameterization(model: "YOLOV7", ckpt_path: str, exclude: list = []) -> "YOLOV7":
        ckpt_state_dict = torch.load(ckpt_path, map_location=next(model.parameters()).device)

        num_anchors = model.head.num_anchors
        exclude = exclude

        # intersect_state_dict = {k: v for k, v in ckpt_state_dict.items() if k in model.state_dict(
        # ) and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
        # model.load_state_dict(intersect_state_dict, strict=False)

        model = load_ckpt(model, ckpt_state_dict)

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

        re_model = deepcopy(model)

        return re_model


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from vision_kit.models.architectures import build_model
    cfg = OmegaConf.load("./configs/yolov7.yaml")
    x = torch.rand((1, 3, 640, 640))
    # model = build_model(cfg)
    model = YOLOV7(training_mode=False)
    # model.eval()
    model.fuse()
    for key in model.state_dict().keys():
        print(key)
    # print(model)
