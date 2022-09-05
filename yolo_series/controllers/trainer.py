from collections import defaultdict
from typing import Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import lr_scheduler
from torchinfo import summary
from yolo_series.controllers.evaluator import COCOEvaluator
from yolo_series.models.architectures.yolov5 import YOLOV5
from yolo_series.utils.metrics import MeterBuffer
from yolo_series.utils.models_utils import ModelEmaV2
from yolo_series.utils.postprocess import nms


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model: YOLOV5, evaluator=None) -> None:
        super(TrainingModule, self).__init__()

        self.model = model

        self.hyp = cfg.hypermeters
        self.data = cfg.data
        self.test_cfg = cfg.testing

        self.meter = MeterBuffer()
        self.evaluator: COCOEvaluator = evaluator

        self.data_list = []
        self.output_data = defaultdict()
        self.ema_model = ModelEmaV2(self.model, 0.9998)
        self.mloss = torch.zeros(3, device=self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, label, _, _ = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        # output = self.forward(img)
        # loss, loss_items = self.model.head.compute_loss(output, label)
        # self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)
        # # self.ema_model.update(self.model)

        # return loss

    def on_validation_start(self) -> None:
        self.model.fuse()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, labels, img_infos, ids = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        # output = self.forward(img)[0]
        # output = nms(output, self.test_cfg.conf_thresh,
        #              self.test_cfg.iou_thresh, max_det=100)

        # data_list_elem, image_wise_data = self.evaluator.convert_to_coco(
        #     output, img_infos, ids)
        # self.data_list.extend(data_list_elem)
        # self.output_data.update(image_wise_data)

    # def on_validation_end(self) -> None:
    #     (ap50_95, ap50, summary) = self.evaluator.evaluate_prediction(
    #         self.data_list)

    #     print(ap50_95, ap50, summary)

    def configure_optimizers(self):
        if self.hyp.warmup_epochs > 0:
            lr = self.hyp.warmup_bias_lr
        else:
            lr = self.hyp.basic_lr_per_img * self.data.batch_size

        lr = 0.01

        # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
        # for k, v in self.model.named_modules():
        #     if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        #         pg2.append(v.bias)  # biases
        #     if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        #         pg1.append(v.weight)  # no decay
        #     elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        #         pg0.append(v.weight)  # apply decay

        # optimizer = torch.optim.SGD(
        #     pg2, lr=lr, momentum=self.hyp.momentum, nesterov=True
        # )
        # optimizer.add_param_group(
        #     {"params": pg0, "weight_decay": self.hyp.weight_decay}
        # )  # add pg1 with weight_decay
        # optimizer.add_param_group({"params": pg1, 'weight_decay': 0.0})

        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=self.hyp.momentum
        )

        # pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        # for k, v in self.model.named_modules():
        #     if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
        #         pg2.append(v.bias)  # biases
        #     if isinstance(v, nn.BatchNorm2d) or "bn" in k:
        #         pg0.append(v.weight)  # no decay
        #     elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
        #         pg1.append(v.weight)  # apply decay

        # optimizer = torch.optim.SGD(
        #     pg0, lr=lr, momentum=self.hyp.momentum, nesterov=True
        # )
        # optimizer.add_param_group(
        #     {"params": pg1, "weight_decay": self.hyp.weight_decay}
        # )  # add pg1 with weight_decay
        # optimizer.add_param_group({"params": pg2})

        # g = [], [], []  # optimizer parameter groups
        # # normalization layers, i.e. BatchNorm2d()
        # bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        # for v in self.model.modules():
        #     if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
        #         g[2].append(v.bias)
        #     if isinstance(v, bn):  # weight (no decay)
        #         g[1].append(v.weight)
        #     elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
        #         g[0].append(v.weight)

        # optimizer = torch.optim.SGD(
        #     g[2], lr=lr, momentum=self.hyp.momentum, nesterov=True)

        # # add g0 with weight_decay
        # optimizer.add_param_group(
        #     {'params': g[0], 'weight_decay': self.hyp.weight_decay})
        # optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})

        def lf(x): return (1 - x / self.data.max_epochs) * \
            (1.0 - self.hyp.lrf) + self.hyp.lrf

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return [optimizer], [scheduler]

    def on_train_start(self) -> None:
        # summary(self.model)
        print("Training starts...")

    def on_train_end(self) -> None:
        print("Training ends...")
