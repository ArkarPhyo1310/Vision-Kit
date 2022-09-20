from typing import Any, Dict, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from vision_kit.models.architectures import build_model
from vision_kit.utils.drawing import grid_save
from vision_kit.utils.image_proc import nms
from vision_kit.utils.logging_utils import logger
from vision_kit.utils.model_utils import (ModelEMA, extract_ema_weight,
                                          load_ckpt)
from vision_kit.utils.table import RichTable
from torchinfo import summary


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, evaluator=None, pretrained: bool = True) -> None:
        super(TrainingModule, self).__init__()

        cfg = self.update_loss_cfg(cfg)

        if pretrained:
            self.model = self.load_pretrained(cfg)
        else:
            self.model = build_model(cfg)

        self.hyp_cfg = cfg.hypermeters
        self.data_cfg = cfg.data
        self.test_cfg = cfg.testing

        self.evaluator = evaluator
        self.ema_model = ModelEMA(self.model)
        self.metrics_mAP = MeanAveragePrecision()

        self.example_input_array = torch.ones((1, 3, *(cfg.model.input_size)))
        model_info = summary(self.model, input_size=self.example_input_array.shape, verbose=0, depth=2)
        logger.info(f"Model Info\n{model_info}")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, targets, _, _ = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/train")

        outputs = self.model(imgs)
        targets = torch.cat(targets, 0)
        loss, loss_items = self.model.head.compute_loss(outputs, targets)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, targets, img_infos, idxs = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/val")

        if self.ema_model:
            outputs = self.ema_model.module(imgs)
        else:
            outputs = self.model(imgs)

        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, preds=output, targets=targets
        )

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, targets, img_infos, idxs = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/test")

        if self.ema_model:
            outputs = self.ema_model.module(imgs)
        else:
            outputs = self.model(imgs)

        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, preds=output, targets=targets
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

        self.metrics_mAP.update(pred_dict, target_dict)

    def training_step_end(self, step_output) -> STEP_OUTPUT:
        if self.ema_model:
            self.ema_model.update(self.model)

    def training_epoch_end(self, outputs) -> None:
        self.lr_scheduler.step()

    def validation_epoch_end(self, outputs) -> None:
        map50, map95, _ = self.evaluator.evaluate_predictions()
        self.log("mAP@.5", map50, prog_bar=True)
        self.log("mAP@.5:.95", map95, prog_bar=True)

    def test_epoch_end(self, outputs) -> None:
        logger.info("Testing finished...")
        map50, map95, self.per_class_table = self.evaluator.evaluate_predictions(details_per_class=True)
        results = self.metrics_mAP.compute()

        mAP_res = []
        self.mAP_table = RichTable("Average Precision (AP)")
        self.mAP_table.add_headers(["mAP", "mAP(.50)", "mAP(.75)", "mAP(small)", "mAP(medium)", "mAP(large)"])
        mAP_res.append(round(results["map"].detach().item(), 3))
        mAP_res.append(round(results["map_50"].detach().item(), 3))
        mAP_res.append(round(results["map_75"].detach().item(), 3))
        mAP_res.append(round(results["map_small"].detach().item(), 3))
        mAP_res.append(round(results["map_medium"].detach().item(), 3))
        mAP_res.append(round(results["map_large"].detach().item(), 3))
        self.mAP_table.add_content([mAP_res])

        mAR_res = []
        self.mAR_table = RichTable("Average Recall (AR)")
        self.mAR_table.add_headers(["mAR", "mAR(max=10)", "mAR(max=100)", "mAR(small)", "mAR(medium)", "mAR(large)"])
        mAR_res.append(round(results["mar_1"].detach().item(), 3))
        mAR_res.append(round(results["mar_10"].detach().item(), 3))
        mAR_res.append(round(results["mar_100"].detach().item(), 3))
        mAR_res.append(round(results["mar_small"].detach().item(), 3))
        mAR_res.append(round(results["mar_medium"].detach().item(), 3))
        mAR_res.append(round(results["mar_large"].detach().item(), 3))
        self.mAR_table.add_content([mAR_res])

    def on_test_end(self) -> None:
        logger.info(self.per_class_table.table)
        logger.info(self.mAP_table.table)
        logger.info(self.mAR_table.table)

    def configure_optimizers(self):
        g = [], [], []  # optimizer parameter groups
        # normalization layers, i.e. BatchNorm2d()
        bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
                g[2].append(v.bias)
            if isinstance(v, bn):  # weight (no decay)
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
                g[0].append(v.weight)

        optimizer = torch.optim.SGD(
            g[2], lr=self.hyp_cfg.lr0, momentum=self.hyp_cfg.momentum, nesterov=True)

        # add g0 with weight_decay
        optimizer.add_param_group(
            {'params': g[0], 'weight_decay': self.hyp_cfg.weight_decay})
        # add g1 (BatchNorm2d weights)
        optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})

        def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
            (1.0 - self.hyp_cfg['lrf']) + self.hyp_cfg['lrf']  # linear
        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

        return optimizer

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=True,
        using_lbfgs=False
    ) -> None:

        optimizer.zero_grad()

        def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
            (1.0 - self.hyp_cfg.lrf) + self.hyp_cfg.lrf  # linear

        ni = self.trainer.global_step
        if ni <= self.nw:
            xi = [0, self.nw]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.hyp_cfg['warmup_bias_lr'] if j ==
                                             0 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(
                        ni, xi, [self.hyp_cfg.warmup_momentum, self.hyp_cfg.momentum])

        # update params
        optimizer.step(closure=optimizer_closure)
        # optimizer.zero_grad()

    def on_train_start(self) -> None:
        self.lr_scheduler.last_epoch = -1
        self.nw = max(round(self.hyp_cfg['warmup_epochs'] * self.trainer.num_training_batches), 100)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema_model:
            checkpoint['model'] = self.ema_model.module.half().state_dict()
        else:
            checkpoint['model'] = self.model.half().state_dict()

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.ema_model:
            avg_params = extract_ema_weight(checkpoint)
            if len(avg_params) != len(self.model.state_dict()):
                logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.ema_model.module.load_state_dict(avg_params)
                self.ema_model.updates = checkpoint["epoch"]
                logger.info("Loaded average state from checkpoint.")

    @staticmethod
    def load_pretrained(cfg):
        model = build_model(cfg)
        state_dict = torch.load(cfg.model.weight, map_location="cpu")
        model = load_ckpt(model, state_dict)
        return model

    @staticmethod
    def update_loss_cfg(cfg):
        nl = 3
        cfg.hypermeters.box *= 3 / nl
        cfg.hypermeters.cls *= cfg.model.num_classes / 80 * 3 / nl  # scale to classes and layers
        cfg.hypermeters.obj *= (cfg.model.input_size[0] / 640) ** 2 * 3 / nl  # scale to image size and layers

        return cfg
