from collections import defaultdict
from typing import Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchinfo import summary
from torchmetrics.detection import mean_ap
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.models.architectures.yolov5 import YOLOV5
from vision_kit.utils.lr import CosineWarmupScheduler
from vision_kit.utils.model_utils import ModelEMA
from vision_kit.utils.image_proc import nms


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model, evaluator=None) -> None:
        super(TrainingModule, self).__init__()

        self.model = model
        self.evaluator = evaluator

        self.hyp = cfg.hypermeters
        self.data = cfg.data
        self.test_cfg = cfg.testing

        self.ema_model = ModelEMA(self.model, )
        self.last_opt_step = -1

        self.metrics = mean_ap.MeanAveragePrecision()
        # self.automatic_optimization = False
        self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def on_train_start(self) -> None:
        if self.current_epoch > 0:
            self.lr_schedulers().last_epoch = self.current_epoch - 1

        self.nb = self.trainer.num_training_batches
        self.num_warmups = max(round(self.hyp.warmup_epochs * self.nb), 100)

    def on_train_epoch_start(self) -> None:
        self.mloss = torch.zeros(3, device=self.device)
        self.optimizers().zero_grad()
        # self.scaler = self.trainer.precision_plugin.scaler

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        opt = self.optimizers()

        img, label, _, _ = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        output = self.model(img)
        loss, loss_items = self.model.head.compute_loss(output, label)

        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()
        # self.lr_schedulers().step()

        self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.lr_schedulers().step()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, labels, img_infos, ids = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        output = self.model(img)

        outputs = nms(output[0], self.test_cfg.conf_thresh,
                      self.test_cfg.iou_thresh, multi_label=True)

        predn, targetn = self.evaluator.evaluate(
            img=img, img_infos=img_infos, preds=outputs, targets=labels
        )
        pred_dict = [
            dict(
                boxes=predn[:, 0:4],
                scores=predn[:, 4],
                labels=predn[:, 5]
            )
        ]

        target_dict = [
            dict(
                boxes=targetn[:, 1:5],
                labels=targetn[:, 0]
            )
        ]

        self.metrics.update(pred_dict, target_dict)
        # return loss

    def validation_epoch_end(self, outputs) -> None:
        from pprint import pprint
        pprint(self.metrics.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, weight_decay=0.0005, nesterov=True, momentum=self.hyp.momentum
        )
        # total_steps = self.trainer.estimated_stepping_batches
        # lr_scheduler = CosineWarmupScheduler(optimizer, warmup=0.1 * total_steps, max_iters=total_steps)
        def lf(x): return (1 - x / self.trainer.max_epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

        return [optimizer], [lr_scheduler]

    def on_train_end(self) -> None:
        logger.info("Training ends...")
