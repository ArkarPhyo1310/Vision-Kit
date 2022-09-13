from typing import Any, Optional

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchinfo import summary
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.models.architectures.yolov5 import YOLOV5
from vision_kit.utils.bboxes import xywhn_to_xyxy
from vision_kit.utils.image_proc import nms
from vision_kit.utils.model_utils import ModelEMA


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


def smart_optimizer(model, lr=0.01, momentum=0.9, decay=1e-5):
    # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
    g = [], [], []  # optimizer parameter groups
    # normalization layers, i.e. BatchNorm2d()
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)

    # add g0 with weight_decay
    optimizer.add_param_group({'params': g[0], 'weight_decay': decay})
    # add g1 (BatchNorm2d weights)
    optimizer.add_param_group({'params': g[1], 'weight_decay': 0.0})
    print(f"'optimizer:' {type(optimizer).__name__}(lr={lr}) with parameter groups "
          f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias")
    return optimizer


class TrainingModule(pl.LightningModule):
    def __init__(self, cfg, model, evaluator=None) -> None:
        super(TrainingModule, self).__init__()

        self.model = model
        self.evaluator = evaluator

        self.hyp_cfg = cfg.hypermeters
        self.data_cfg = cfg.data
        self.test_cfg = cfg.testing

        self.ema_model = ModelEMA(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        imgs, targets, _, _ = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            grid_save(imgs, targets)

        outputs = self.model(imgs)
        targets = torch.cat(targets, 0)
        loss, loss_items = self.model.head.compute_loss(outputs, targets)

        self.log("train_loss", loss)

        return loss

    # def training_epoch_end(self, outputs) -> None:
    #     self.lr_scheduler.step()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        imgs, targets, img_infos, idxs = batch
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        if batch_idx == 0:
            grid_save(imgs, targets, name="val")

        outputs = self.eval_model(imgs)
        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, preds=output, targets=targets
        )

    def validation_epoch_end(self, outputs) -> None:
        map50, map95 = self.evaluator.evaluate_predictions()
        self.log("mAP@.5", map50, prog_bar=True)
        self.log("mAP@.5:.95", map95, prog_bar=True)

        return {
            "map50": map50,
            "map95": map95
        }

    def configure_optimizers(self):
        optimizer = smart_optimizer(
            self.model, self.hyp_cfg.lr0, self.hyp_cfg.momentum, self.hyp_cfg.momentum
        )

        def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
            (1.0 - self.hyp_cfg['lrf']) + self.hyp_cfg['lrf']  # linear
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

        return [optimizer], [lr_scheduler]

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx=0,
        optimizer_closure=None,
        on_tpu=False,
        using_native_amp=True,
        using_lbfgs=False
    ) -> None:
        def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
            (1.0 - self.hyp_cfg['lrf']) + self.hyp_cfg['lrf']  # linear
        nw = self.trainer.num_training_batches
        ni = self.trainer.global_step
        if ni <= nw:
            xi = [0, nw]
            for j, x in enumerate(optimizer.param_groups):
                x['lr'] = np.interp(ni, xi, [self.hyp_cfg['warmup_bias_lr'] if j ==
                                    0 else 0.0, x['initial_lr'] * lf(epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(
                        ni, xi, [self.hyp_cfg['warmup_momentum'], self.hyp_cfg['momentum']])

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        if self.ema_model:
            self.ema_model.update(self.model)
            self.ema_model.update_attr(self.model)

    def on_validation_epoch_start(self) -> None:
        if self.ema_model:
            self.eval_model = self.ema_model.ema
        else:
            self.eval_model = self.model

        self.eval_model.eval()
