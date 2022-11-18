from typing import Optional

import numpy as np
import torch
import wandb
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from vision_kit.core.train.base_trainer import TrainingModule
from vision_kit.models.architectures import YOLOV5, YOLOV7, build_model
from vision_kit.models.losses.yolo import YoloLoss
from vision_kit.utils.drawing import grid_save
from vision_kit.utils.image_proc import nms
from vision_kit.utils.logging_utils import logger
from vision_kit.utils.model_utils import ModelEMA
from vision_kit.utils.table import RichTable


class DetTrainer(TrainingModule):
    def __init__(self, cfg, evaluator=None, pretrained: bool = True) -> None:
        super(DetTrainer, self).__init__(cfg, evaluator, pretrained)

        if pretrained:
            self.model = self.load_pretrained(cfg)
        else:
            self.model = build_model(cfg)

        self.hyp_cfg = cfg.hypermeters
        self.data_cfg = cfg.data
        self.test_cfg = cfg.testing

        self.evaluator = evaluator
        self.ema_model = ModelEMA(self.model)
        self.metrics_mAP = MeanAveragePrecision(compute_on_cpu=True)
        self.loss = YoloLoss(cfg.model.num_classes, hyp=self.hyp_cfg)
        self.loss.set_anchor(self.model.head.anchors)

        self.model_info()

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, targets, _, _ = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            self.train_batch_grid = grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/train")

        outputs = self.model(imgs)
        targets = torch.cat(targets, 0)
        loss, loss_items = self.loss(outputs, targets)

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, targets, img_infos, idxs = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            self.val_batch_grid = grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/val")

        outputs = self.get_model()(imgs)
        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        self.evaluator.evaluate(img=imgs, img_infos=img_infos, idxs=idxs, preds=output, targets=targets)

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, targets, img_infos, idxs = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            self.test_batch_grid = grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/test")

        outputs = self.get_model()(imgs)
        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, idxs=idxs, preds=output, targets=targets
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
        self.log("loss", outputs[0])
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({
                    "samples/train": [wandb.Image(f"{self.data_cfg.output_dir}/train.jpg")]
                })
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    'samples/train', self.train_batch_grid, 0)

    def validation_epoch_end(self, outputs) -> None:
        map50, map95, _, _ = self.evaluator.summarize()
        self.log("mAP@.5", map50, prog_bar=True)
        self.log("mAP@.5:.95", map95, prog_bar=True)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({
                    "samples/val": [wandb.Image(f"{self.data_cfg.output_dir}/val.jpg")]
                })
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image(
                    'samples/val', self.val_batch_grid, 0)

    def test_epoch_end(self, outputs) -> None:
        for trainer_logger in self.loggers:
            if isinstance(trainer_logger, WandbLogger):
                trainer_logger.experiment.log({
                    "samples/test": [wandb.Image(f"{self.data_cfg.output_dir}/test.jpg")]
                })
            elif isinstance(trainer_logger, TensorBoardLogger):
                trainer_logger.experiment.add_image(
                    'samples/test', self.test_batch_grid, 0)

        logger.info("Testing finished...")
        _, _, self.per_class_table, _ = self.evaluator.summarize(
            details_per_class=True)
        logger.info(self.per_class_table.table)

        results = self.metrics_mAP.compute()

        mAP_res = []
        self.mAP_table = RichTable("Average Precision (AP)")
        self.mAP_table.add_headers(
            ["mAP", "mAP(.50)", "mAP(.75)", "mAP(small)", "mAP(medium)", "mAP(large)"])
        mAP_res.append(round(results["map"].detach().item(), 3))
        mAP_res.append(round(results["map_50"].detach().item(), 3))
        mAP_res.append(round(results["map_75"].detach().item(), 3))
        mAP_res.append(round(results["map_small"].detach().item(), 3))
        mAP_res.append(round(results["map_medium"].detach().item(), 3))
        mAP_res.append(round(results["map_large"].detach().item(), 3))
        self.mAP_table.add_content([mAP_res])

        mAR_res = []
        self.mAR_table = RichTable("Average Recall (AR)")
        self.mAR_table.add_headers(
            ["mAR", "mAR(max=10)", "mAR(max=100)", "mAR(small)", "mAR(medium)", "mAR(large)"])
        mAR_res.append(round(results["mar_1"].detach().item(), 3))
        mAR_res.append(round(results["mar_10"].detach().item(), 3))
        mAR_res.append(round(results["mar_100"].detach().item(), 3))
        mAR_res.append(round(results["mar_small"].detach().item(), 3))
        mAR_res.append(round(results["mar_medium"].detach().item(), 3))
        mAR_res.append(round(results["mar_large"].detach().item(), 3))
        self.mAR_table.add_content([mAR_res])

        logger.info(self.mAP_table.table)
        logger.info(self.mAR_table.table)

    def configure_optimizers(self) -> tuple[list[SGD], list[LambdaLR]]:
        optimizer, lr_scheduler = self.model.get_optimizer(self.hyp_cfg, self.data_cfg.max_epochs)

        return [optimizer], [lr_scheduler]

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

        if isinstance(self.model, YOLOV5):
            pg_idx = 0
        elif isinstance(self.model, YOLOV7):
            pg_idx = 2

        def lf(x):
            return (1 - x / self.data_cfg.max_epochs) * (1.0 - self.hyp_cfg.lrf) + self.hyp_cfg.lrf  # linear

        ni: int = self.trainer.global_step
        if ni <= self.nw:
            xi: list[int] = [0, self.nw]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.hyp_cfg['warmup_bias_lr'] if j == pg_idx else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(
                        ni, xi, [self.hyp_cfg.warmup_momentum, self.hyp_cfg.momentum])

        # update params
        optimizer.step(closure=optimizer_closure)

    def on_train_start(self) -> None:
        self.nw: int = max(round(self.hyp_cfg['warmup_epochs'] * self.trainer.num_training_batches), 100)
