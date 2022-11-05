import torch
from torch import nn
from vision_kit.models.modules.blocks import ELAN, MP, Concat, ConvBnAct, MPx3Conv


class EELAN(nn.Module):
    def __init__(
        self,
        variant: str = "base",
        act: str = "silu",
    ) -> None:
        super().__init__()

        assert variant.lower() in ["tiny", "base",
                                   "extra"], f"Not supported version: {variant}!"

        backbone_cfg = {
            "tiny": {
                "base_chs": 32,
                "elan_depth": 2
            },
            "base": {
                "base_chs": 32,
                "elan_depth": 4
            },
            "extra": {
                "base_chs": 40,
                "elan_depth": 6
            },
        }

        base_chs = backbone_cfg[variant.lower()]["base_chs"]
        elan_depth = backbone_cfg[variant.lower()]["elan_depth"]

        self.concat = Concat()
        # self.max_pool = MP()

        self.stem = ConvBnAct(
            ins=3, outs=base_chs,
            kernel=3, stride=1, act=act
        )

        self.stage1 = nn.Sequential(
            ConvBnAct(
                base_chs, base_chs * 2,
                kernel=3, stride=2, act=act
            ),
            ConvBnAct(
                base_chs * 2, base_chs * 2,
                kernel=3, stride=1, act=act
            )
        )

        self.stage2 = nn.Sequential(
            ConvBnAct(
                base_chs * 2, base_chs * 4,
                kernel=3, stride=2, act=act
            ),
            ELAN(
                base_chs * 4,  64, base_chs * 8,
                depth=elan_depth
            )
        )
        self.stage2_1 = MPx3Conv(base_chs * 8, base_chs * 4)

        self.stage3 = ELAN(
            base_chs * 8, 128, base_chs * 16,
            depth=elan_depth
        )
        self.stage3_1 = MPx3Conv(base_chs * 16, base_chs * 8)

        self.stage4 = ELAN(
            base_chs * 16, 256, base_chs * 32,
            depth=elan_depth
        )
        self.stage4_1 = MPx3Conv(base_chs * 32, base_chs * 16)

        self.stage5 = ELAN(
            base_chs * 32, 256, base_chs * 32,
            depth=elan_depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        p1 = self.stage1(x)

        p2 = self.stage2(p1)
        p2_1, p2_2 = self.stage2_1(p2)
        p2_concat = self.concat([p2_1, p2_2])

        p3 = self.stage3(p2_concat)
        p3_1, p3_2 = self.stage3_1(p3)
        p3_concat = self.concat([p3_1, p3_2])

        p4 = self.stage4(p3_concat)
        p4_1, p4_2 = self.stage4_1(p4)
        p4_concat = self.concat([p4_1, p4_2])

        p5 = self.stage5(p4_concat)

        return p3, p4, p5


if __name__ == "__main__":
    backbone = EELAN()
    img = torch.rand(1, 3, 640, 640)
    for y in backbone(img):
        print(y.shape)
    # print(len(backbone(img)))
