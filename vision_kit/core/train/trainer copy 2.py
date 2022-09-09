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

        self.data_list = []
        self.output_data = defaultdict()
        self.ema_model = ModelEMA(self.model)
        self.last_opt_step = -1

        self.metrics = mean_ap.MeanAveragePrecision()
        for k, v in self.model.named_parameters():
            v.requires_grad = True  # train all layers

        self.automatic_optimization = False

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

        self.ni = batch_idx + self.nb * self.current_epoch

        # import numpy as np

        # def lf(x): return (1 - x / self.data.max_epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear

        # if self.ni <= self.num_warmups:
        #     xi = [0, self.num_warmups]
        #     self.accumulate = max(1, np.interp(self.ni, xi, [1, self.nbs / self.data.batch_size]).round())
        #     for j, x in enumerate(self.optimizers().param_groups):
        #         # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
        #         x['lr'] = np.interp(self.ni, xi, [self.hyp.warmup_bias_lr if j ==
        #                             0 else 0.0, x['initial_lr'] * lf(self.current_epoch)])
        #         if 'momentum' in x:
        #             x['momentum'] = np.interp(self.ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

        with torch.cuda.amp.autocast(True):
            output = self.model(img)
            loss, loss_items = self.model.head.compute_loss(output, label)

        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        # self.scaler.scale(loss).backward()
        # self.scaler.unscale_(self.optimizers())
        # self.scaler.step(self.optimizers())
        # self.scaler.update()

        # if self.ni - self.last_opt_step >= self.accumulate:
        #     # print("HERE")
        #     # self.scaler.unscale_(self.optimizers())  # unscale gradients
        #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
        #     # self.scaler.step(self.optimizers())  # optimizer.step
        #     self.optimizers().step()
        #     # self.scaler.update()
        #     self.optimizers().zero_grad()
        #     if ema:
        #         ema.update(model)
        #     self.last_opt_step = self.ni

        self.mloss = (self.mloss * batch_idx + loss_items) / (batch_idx + 1)
        return loss

    def training_epoch_end(self, outputs) -> None:
        self.lr_schedulers().step()

    # def on_validation_start(self) -> None:
    #     self.model.fuse()

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        img, labels, img_infos, ids = batch
        img = torch.permute(img, (0, 3, 1, 2))
        img = img.float() / 255

        output = self.model(img)
        # loss, loss_items = self.model.head.compute_loss(output, labels)

        outputs = nms(output[0], self.test_cfg.conf_thresh,
                      self.test_cfg.iou_thresh, multi_label=True)

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

        # return loss

    def on_validation_end(self) -> None:
        # (ap50_95, ap50, summary) = self.evaluator.evaluate_prediction(
        #     self.data_list)

        # logger.info(f"mAP@0.5 => {round(ap50, 4) * 100} %")
        # logger.info(f"mAP@0.5:0.95 => {round(ap50_95, 4) * 100} %")
        # logger.info(f"\n{summary}")
        from pprint import pprint
        pprint(self.metrics.compute())

    def configure_optimizers(self):
        # if self.hyp.warmup_epochs > 0:
        #     lr = self.hyp.warmup_bias_lr
        # else:
        #     lr = self.hyp.lr0

        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.data.batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.data.batch_size * self.accumulate / self.nbs  # scale weight_decay

        g = [], [], []
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
        for v in self.model.modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                g[2].append(v.bias)
            if isinstance(v, bn):
                g[1].append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
                g[0].append(v.weight)

        def lf(x): return (1 - x / self.data.max_epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear

        optimizer = torch.optim.SGD(
            g[2], lr=self.hyp.lr0, momentum=self.hyp.momentum, nesterov=True
        )

        optimizer.add_param_group({
            'params': g[0], 'weight_decay': self.hyp.weight_decay
        })
        optimizer.add_param_group({
            'params': g[1], 'weight_decay': 0.0
        })

        # optimizer = torch.optim.AdamW(
        #     self.parameters(), lr=0.001, weight_decay=0.05
        # )
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=300, eta_min=0.00005)

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        print(f"{'optimizer:'} {type(optimizer).__name__}(lr={self.hyp.lr0}) with parameter groups "
              f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={self.hyp['weight_decay']}), {len(g[2])} bias")

        return [optimizer], [scheduler]

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = 0,
        optimizer_closure=None,
        on_tpu: bool = False,
        using_native_amp: bool = True,
        using_lbfgs: bool = False
    ) -> None:
        # if self.global_step <= 500:
        #     k = (1 - self.global_step / 500) * (1 - 0.001)
        #     warmup_lr = 0.001 * (1 - k)
        #     for pg in optimizer.param_groups:
        #         pg["lr"] = warmup_lr
        import numpy as np

        def lf(x): return (1 - x / self.data.max_epochss) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear

        if self.ni <= self.num_warmups:
            xi = [0, self.num_warmups]
            self.accumulate = max(1, np.interp(self.ni, xi, [1, 64 / self.data.batch_size]).round())
            for j, x in enumerate(self.optimizers().param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(self.ni, xi, [self.hyp.warmup_bias_lr if j ==
                                                  0 else 0.0, x['initial_lr'] * lf(self.current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(self.ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

        if self.ni - last_opt_step >= self.accumulate:
            last_opt_step = self.ni

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def on_train_end(self) -> None:
        logger.info("Training ends...")
