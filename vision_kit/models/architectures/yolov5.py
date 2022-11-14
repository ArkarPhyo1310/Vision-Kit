import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

from vision_kit.models.backbones import CSPDarknet
from vision_kit.models.heads import YoloV5Head
from vision_kit.models.modules.blocks import ConvBnAct
from vision_kit.models.necks import PAFPN
from vision_kit.utils.general import dw_multiple_generator
from vision_kit.utils.model_utils import fuse_conv_and_bn, init_weights


class YOLOV5(nn.Module):
    def __init__(
        self,
        variant: str = "s",
        act: str = "silu",
        num_classes: int = 80,
        training_mode: bool = True,
        export: bool = False
    ) -> None:
        super(YOLOV5, self).__init__()

        wid_mul, dep_mul = dw_multiple_generator(variant)

        self.backbone: CSPDarknet = CSPDarknet(depth_mul=dep_mul, width_mul=wid_mul, act=act)
        self.neck: PAFPN = PAFPN(depth_mul=dep_mul, width_mul=wid_mul, act=act)
        self.head: YoloV5Head = YoloV5Head(num_classes, width=wid_mul, training_mode=training_mode, export=export)

        init_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x

    def fuse(self) -> None:
        for module in [self.backbone, self.neck, self.head]:
            for m in module.modules():
                if type(m) is ConvBnAct and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse

    def get_optimizer(self, hyp_cfg: dict, max_epochs: int) -> tuple[SGD, LambdaLR]:
        g = [], [], []  # optimizer parameter groups
        # normalization layers, i.e. BatchNorm2d()
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        for v in self.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        optimizer: SGD = SGD(
            g[2], lr=hyp_cfg.lr0, momentum=hyp_cfg.momentum, nesterov=True)

        # add g0 with weight_decay
        optimizer.add_param_group(
            {'params': g[0], 'weight_decay': hyp_cfg.weight_decay})
        # add g1 (BatchNorm2d weights)
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})

        def lf(x): return (1 - x / max_epochs) * \
            (1.0 - hyp_cfg['lrf']) + hyp_cfg['lrf']  # linear
        lr_scheduler: LambdaLR = LambdaLR(optimizer, lr_lambda=lf)

        return optimizer, lr_scheduler


if __name__ == "__main__":
    from omegaconf import OmegaConf

    from vision_kit.models.architectures import build_model
    cfg = OmegaConf.load("./configs/yolov5.yaml")
    x = torch.rand((1, 3, 640, 640))
    model = build_model(cfg)
    anchors = [
        [10, 13, 16, 30, 33, 23],
        [30, 61, 62, 45, 59, 119],
        [116, 90, 156, 198, 373, 326]
    ]
