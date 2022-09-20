from typing import Any, Dict, Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from vision_kit.models.architectures import build_model
from vision_kit.utils.bboxes import xywhn_to_xyxy
from vision_kit.utils.image_proc import nms
from vision_kit.utils.logging_utils import logger
from vision_kit.utils.model_utils import (ModelEMA, extract_ema_weight,
                                          load_ckpt)


def grid_save(imgs, targets, name="train"):
    img_list = []
    row = int(imgs.shape[0] / 2)
    for idx, (img, labels) in enumerate(zip(imgs, targets)):
        img_arr = img.cpu().numpy()
        img_arr = img_arr.transpose((1, 2, 0))
        bboxes = xywhn_to_xyxy(
            labels[:, 2:], img_arr.shape[1], img_arr.shape[0]).cpu().numpy()
        classes = labels[:, 1]
        for bbox, idx in zip(bboxes, classes):
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = int(bbox[2])
            y1 = int(bbox[3])

            text = str(int(idx.cpu().numpy().item()))
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
            cv2.rectangle(img_arr, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(
                img_arr,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                (128, 0, 0),
                -1
            )
            cv2.putText(img_arr, text, (x0, y0 +
                        txt_size[1]), font, 0.8, (255, 255, 255), thickness=2)

        img_transpose = img_arr.transpose((2, 0, 1))
        img_list.append(img_transpose)

    img = np.stack(img_list, 0)
    img_tensor = torch.from_numpy(img)
    batch_grid = torchvision.utils.make_grid(
        img_tensor, normalize=False, nrow=row)
    torchvision.utils.save_image(batch_grid, f"{name}.jpg")


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, evaluator=None, pretrained=True) -> None:
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

        self.automatic_optimization = True
        self.nbs = 64
        self.accumulate = max(round(self.nbs / self.data_cfg.batch_size), 1)
        self.last_opt_step = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        # print("training_step")
        imgs, targets, _, _ = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/train")

        ni = self.trainer.global_step
        # # print(ni)
        # # print(self.nw)
        if ni <= self.nw:
            xi = [0, self.nw]
            self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.data_cfg.batch_size]).round())
            for j, x in enumerate(self.optimizers().param_groups):
                x['lr'] = np.interp(ni, xi, [self.hyp_cfg['warmup_bias_lr'] if j ==
                                             0 else 0.0, x['initial_lr'] * self.lf(self.current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(
                        ni, xi, [self.hyp_cfg['warmup_momentum'], self.hyp_cfg['momentum']])

        outputs = self.model(imgs)
        targets = torch.cat(targets, 0)
        loss, loss_items = self.model.head.compute_loss(outputs, targets)

        # self.log("loss", loss)
        # self.trainer.scaler.scale(loss).backward()

        # self.manual_backward(loss)
        # # print(self.accumulate)
        # # exit()
        # if ni - self.last_opt_step >= self.accumulate:
        #     self.trainer.scaler.unscale_(self.optimizers())  # unscale gradients
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        #     self.trainer.scaler.step(self.optimizers())  # optimizer.step
        #     self.trainer.scaler.update()
        #     self.optimizers().zero_grad()
        #     self.last_opt_step = ni

        return loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, targets, img_infos, idxs = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).half()
        imgs /= 255.0

        # if batch_idx == 0:
        #     grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/val")

        if self.ema_model:
            outputs = self.ema_model.module.half()(imgs)
        else:
            outputs = self.model.half()(imgs)

        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, preds=output, targets=targets
        )

    def validation_epoch_end(self, outpus) -> None:
        map50, map95 = self.evaluator.evaluate_predictions()
        self.log("mAP@.5", map50, prog_bar=True)
        self.log("mAP@.5:.95", map95, prog_bar=True)

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

        # def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
        #     (1.0 - self.hyp_cfg['lrf']) + self.hyp_cfg['lrf']  # linear
        self.lr_scheduler = LambdaLR(optimizer, lr_lambda=self.lf)

        return optimizer

    def lf(self, x): return (1 - x / self.data_cfg.max_epochs) * \
        (1.0 - self.hyp_cfg['lrf']) + self.hyp_cfg['lrf']  # linear

    # def optimizer_step(
    #     self,
    #     epoch,
    #     batch_idx,
    #     optimizer,
    #     optimizer_idx,
    #     optimizer_closure=None,
    #     on_tpu=False,
    #     using_native_amp=True,
    #     using_lbfgs=False
    # ) -> None:
    #     # def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
    #     #     (1.0 - self.hyp_cfg['lrf']) + self.hyp_cfg['lrf']  # linear
    #     # ni = self.trainer.global_step
    #     # if ni <= self.nw:
    #     #     xi = [0, self.nw]
    #     #     self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.data_cfg.batch_size]).round())
    #     #     for j, x in enumerate(optimizer.param_groups):
    #     #         x['lr'] = np.interp(ni, xi, [self.hyp_cfg['warmup_bias_lr'] if j ==
    #     #                                      0 else 0.0, x['initial_lr'] * self.lf(epoch)])
    #     #         if 'momentum' in x:
    #     #             x['momentum'] = np.interp(
    #     #                 ni, xi, [self.hyp_cfg['warmup_momentum'], self.hyp_cfg['momentum']])

    #     # update params
    #     # if (ni - self.last_opt_step) >= self.accumulate:
    #     #     print("HERE")
    #     #     self.last_opt_step = ni
    #     print(optimizer_closure.__dir__)
    #     exit()
    #     optimizer.step(closure=optimizer_closure)
    #     optimizer.zero_grad()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # print("on_train_batch_end")
        if self.ema_model:
            self.ema_model.update(self.model)

    def on_train_epoch_end(self) -> None:
        # return super().on_train_epoch_end()
        self.lr_scheduler.step()

    def on_train_epoch_start(self) -> None:
        # return super().on_train_epoch_start()
        self.optimizers().zero_grad()

    def on_train_start(self) -> None:
        # return super().on_train_start()
        self.lr_scheduler.last_epoch = -1
        self.nw = max(round(self.hyp_cfg['warmup_epochs'] * self.trainer.num_training_batches), 100)

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
        cfg.hypermeters.cls *= cfg.model.num_classes / \
            80 * 3 / nl  # scale to classes and layers
        # scale to image size and layers
        cfg.hypermeters.obj *= (cfg.model.input_size[0] / 640) ** 2 * 3 / nl

        return cfg

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
