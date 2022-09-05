import torch
from torch import nn

from yolo_series.models.backbones import CSPDarknet
from yolo_series.models.necks import PAFPN
from yolo_series.models.heads import YOLOV5Head

from yolo_series.models.modules.blocks import ConvBnAct
from yolo_series.utils.torch_utils import fuse_conv_and_bn


def initialize_weights(model):
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


class YOLOV5(nn.Module):
    def __init__(
        self,
        dep_mul: float,
        wid_mul: float,
        act: str = "silu",
        hyp: dict = None,
        num_classes: int = 80,
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
            num_classes, width=wid_mul, hyp=hyp, training=training)

        initialize_weights(self)

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x

    def fuse(self):
        print("Fusing Layers...")
        for module in [self.backbone, self.neck, self.head]:
            for m in module.modules():
                if type(m) is ConvBnAct and hasattr(m, "bn"):
                    m.conv = fuse_conv_and_bn(m.conv, m.bn)
                    delattr(m, 'bn')
                    m.forward = m.forward_fuse

# if __name__ == "__main__":
#     model = YOLOV5(
#         0.33, 0.5, training=False, num_classes=72
#     )
#     model.fuse()
#     ckpt = torch.load(
#         "./pretrained_weights/yolov5s.pt")
#     model = load_ckpt(model, ckpt)
#     model.eval()
#     print(model)
#     print(model(torch.zeros(1, 3, 640, 640))[0].shape)
