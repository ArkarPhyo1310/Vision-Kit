from collections import defaultdict
from typing import Optional

import pytorch_lightning as pl
import torch
from loguru import logger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import lr_scheduler
from torchinfo import summary
from torchmetrics.detection import mean_ap
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.models.architectures.yolov5 import YOLOV5
from vision_kit.utils.model_utils import ModelEmaV2
from vision_kit.utils.image_proc import nms


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model, evaluator=None) -> None:
        super(TrainingModule, self).__init__()

        self.model = model
        self.evaluator = evaluator

        self.hyp = cfg.hypermeters
        self.data = cfg.data
        self.test_cfg = cfg.testing

        self.data_list = []
        self.output_data = defaultdict()
        self.ema_model = ModelEmaV2(self.model, 0.9998)
        self.mloss = torch.zeros(3, device=self.device)

        self.metrics = mean_ap.MeanAveragePrecision()
        # self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        img, label, _, _ = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        output = self.model(img)
        loss, loss_items = self.model.head.compute_loss(output, label)
        self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)

        return loss

    def training_step_end(self, step_output: STEP_OUTPUT) -> STEP_OUTPUT:
        self.ema_model.update(self.model)

    def on_validation_start(self) -> None:
        self.model.fuse()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, labels, img_infos, ids = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        output = self.model(img)[0]
        outputs = nms(output, self.test_cfg.conf_thresh,
                      self.test_cfg.iou_thresh, multi_label=True)
        # outputs = nms(output, 0.25, 0.45, multi_label=True)

        if isinstance(self.evaluator, YOLOEvaluator):
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

        else:
            data_list_elem, image_wise_data = self.evaluator.convert_to_coco(
                outputs, img_infos, ids)
            self.data_list.extend(data_list_elem)
            self.output_data.update(image_wise_data)

    def on_validation_end(self) -> None:
        # (ap50_95, ap50, summary) = self.evaluator.evaluate_prediction(
        #     self.data_list)

        # logger.info(f"mAP@0.5 => {round(ap50, 4) * 100} %")
        # logger.info(f"mAP@0.5:0.95 => {round(ap50_95, 4) * 100} %")
        # logger.info(f"\n{summary}")
        from pprint import pprint
        pprint(self.metrics.compute())

    def configure_optimizers(self):
        if self.hyp.warmup_epochs > 0:
            lr = self.hyp.warmup_bias_lr
        else:
            lr = self.hyp.lr0

        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, bn):
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)

        optimizer = torch.optim.SGD(
            g[2], lr=lr, momentum=self.hyp.momentum, nesterov=True
        )

        optimizer.add_param_group({
            'params': g[0], 'weight_decay': self.hyp.weight_decay
        })
        optimizer.add_param_group({
            'params': g[1], 'weight_decay': 0.0
        })

        def lf(x): return (1 - x / self.data.max_epochs) * \
            (1.0 - self.hyp.lrf) + self.hyp.lrf

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        self.items = super().get_progress_bar_dict()
        self.items.pop("v_num", None)
        self.items.pop("loss", None)
        return self.items

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = 0,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False
    ) -> None:
        if self.global_step <= 500:
            k = (1 - self.global_step / 500) * (1 - 0.001)
            warmup_lr = 0.001 * (1 - k)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def on_train_start(self) -> None:
        logger.info("Training starts...")

    def on_train_end(self) -> None:
        logger.info("Training ends...")
