import torch
from torch import nn

from loguru import logger

from vision_kit.models.backbones import CSPDarknet
from vision_kit.models.necks import PAFPN
from vision_kit.models.heads import YOLOV5Head

from vision_kit.models.modules.blocks import ConvBnAct
from vision_kit.utils.model_utils import init_weights
from vision_kit.utils.model_utils import fuse_conv_and_bn


class YOLOV5(nn.Module):
    def __init__(
        self,
        dep_mul: float,
        wid_mul: float,
        act: str = "silu",
        hyp: dict = None,
        num_classes: int = 80,
        device: str = "gpu",
        training: bool = True
    ) -> None:
        super(YOLOV5, self).__init__()

        self.backbone: CSPDarknet = CSPDarknet(
            depth_mul=dep_mul, width_mul=wid_mul, act=act
        )
        self.neck: PAFPN = PAFPN(
            depth_mul=dep_mul, width_mul=wid_mul, act=act
        )

        self.head: YOLOV5Head = YOLOV5Head(
            num_classes, width=wid_mul, hyp=hyp, training=training, device=device)

        init_weights(self)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x

    def fuse(self):
        logger.info("Fusing Layers...")
        for module in [self.backbone, self.neck, self.head]:
            for m in module.modules():
                if type(m) is ConvBnAct and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse
