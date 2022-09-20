import math
import os
from copy import deepcopy

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch import nn
from tqdm import tqdm
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule, grid_save
from vision_kit.data.datamodule import LitDataModule
from vision_kit.models.architectures import build_model
from vision_kit.utils.image_proc import nms
from vision_kit.utils.logging_utils import setup_logger
from vision_kit.utils.model_utils import copy_attr, de_parallel, load_ckpt
from vision_kit.utils.training_helpers import (get_callbacks, get_loggers,
                                               get_profilers)

pl.seed_everything(21, workers=True)


class ModelEMA:
    """ Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        self.updates += 1
        d = self.decay(self.updates)

        msd = de_parallel(model).state_dict()  # model state_dict
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:  # true for FP16 and FP32
                v *= d
                v += (1 - d) * msd[k].detach()
        # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype} and model {msd[k].dtype} must be FP32'

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        copy_attr(self.ema, model, include, exclude)


class TrainingTask:
    def __init__(self, cfg, datamodule, evaluator) -> None:
        cfg = self.update_loss_cfg(cfg)
        self.cfg = cfg
        self.model = self.load_pretrained(cfg).to(cfg.model.device)
        self.evaluator = evaluator
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()

        self.hyp_cfg = cfg.hypermeters
        self.data_cfg = cfg.data
        self.test_cfg = cfg.testing

        self.ema = ModelEMA(self.model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.optimizer = self.configure_optimizers()
        self.scheduler = self.configure_schudulers(self.optimizer)

    def train(self):
        self.on_train_start()
        for epoch in range(self.data_cfg.max_epochs):
            self.current_epoch = epoch
            self.on_train_epoch_start()
            for i, (imgs, targets, _, _) in self.tbar:
                self.on_train_iter_start(i, imgs, targets)
            self.on_train_epoch_end()

            self.on_val_epoch_start()
            for i, (imgs, targets, img_infos, idxs) in self.vbar:
                self.on_val_iter_start(i, imgs, targets, img_infos, idxs)
            self.on_val_epoch_end()

    def lf(self, x): return (1 - x / self.data_cfg.max_epochs) * \
        (1.0 - self.hyp_cfg.lrf) + self.hyp_cfg.lrf  # linear

    def on_train_start(self):
        self.nb = len(self.train_loader)  # number of batches
        # number of warmup iterations, max(3 epochs, 100 iterations)
        self.nw = max(round(self.hyp_cfg.warmup_epochs * self.nb), 100)
        self.last_opt_step = -1
        self.scheduler.last_epoch = -1  # do not move

    def on_train_epoch_start(self):
        self.model.train()
        self.optimizer.zero_grad()
        self.tbar = enumerate(self.train_loader)
        self.tbar = tqdm(self.tbar, total=self.nb)

    def on_train_iter_start(self, idx, imgs, targets):
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0
        if idx == 0:
            grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/train")
        imgs = imgs.to(self.cfg.model.device, non_blocking=True)
        targets = torch.cat(targets, 0)

        self.ni = idx + self.nb * self.current_epoch
        # Warmup
        if self.ni <= self.nw:
            xi = [0, self.nw]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            self.accumulate = max(1, np.interp(
                self.ni, xi, [1, self.nbs / self.data_cfg.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(
                    self.ni, xi, [self.hyp_cfg.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(self.current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(
                        self.ni, xi, [self.hyp_cfg.warmup_momentum, self.hyp_cfg.momentum])

        # Forward
        with torch.cuda.amp.autocast(True):
            pred = self.model(imgs)  # forward
            loss, loss_items = self.model.head.compute_loss(
                pred, targets.to(self.cfg.model.device))  # loss scaled by batch_size

        # Backward
        self.scaler.scale(loss).backward()

        # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
        if self.ni - self.last_opt_step >= self.accumulate:
            self.scaler.unscale_(self.optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=10.0)  # clip gradients
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = self.ni

    def on_train_epoch_end(self):
        self.scheduler.step()

    def on_val_start(self):
        pass

    def on_val_epoch_start(self):
        if self.ema:
            self.eval_model = self.ema.ema
        else:
            self.eval_model = self.model

        self.eval_model.eval()
        self.vbar = enumerate(self.val_loader)
        self.vbar = tqdm(self.vbar, total=len(self.val_loader))

    def on_val_iter_start(self, i, imgs, targets, img_infos, idxs):
        imgs = imgs.to(self.cfg.model.device, non_blocking=True)
        targets = targets.to(self.cfg.model.device)
        imgs = torch.permute(imgs, (0, 3, 1, 2)).float()
        imgs /= 255.0

        outputs = self.eval_model(imgs)
        output = nms(outputs[0], self.test_cfg.conf_thresh,
                     self.test_cfg.iou_thresh, multi_label=True)

        targets = torch.cat(targets, 0)
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, preds=output, targets=targets
        )

    def on_val_epoch_end(self):
        map50, map95 = self.evaluator.evaluate_predictions()
        print("mAP@.5", round(map50, 3))
        print("mAP@.5:.95", round(map95, 3))

    def configure_optimizers(self):
        self.nbs = 64  # nominal batch size
        # accumulate loss before optimizing
        self.accumulate = max(round(self.nbs / self.data_cfg.batch_size), 1)
        self.hyp_cfg.weight_decay *= self.data_cfg.batch_size * \
            self.accumulate / self.nbs  # scale weight_decay

        # YOLOv5 3-param group optimizer: 0) weights with decay, 1) weights no decay, 2) biases no decay
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
        print(f"'optimizer:' {type(optimizer).__name__}(lr={self.hyp_cfg.lr0}) with parameter groups "
              f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={self.hyp_cfg.weight_decay}), {len(g[2])} bias")

        return optimizer

    def configure_schudulers(self, optimizer):
        def lf(x): return (1 - x / self.data_cfg.max_epochs) * \
            (1.0 - self.hyp_cfg.lrf) + self.hyp_cfg.lrf  # linear
        # plot_lr_scheduler(optimizer, scheduler, epochs)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        return scheduler

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


def train(cfg, loggers, callbacks, profiler):
    datamodule = LitDataModule(
        data_cfg=cfg.data,
        aug_cfg=cfg.augmentations,
        num_workers=cfg.data.num_workers,
        img_sz=cfg.model.input_size,
    )
    datamodule.setup()
    evaluator = YOLOEvaluator(
        class_labels=cfg.data.class_labels,
        img_size=cfg.model.input_size
    )

    cfg.model.weight = "./pretrained_weights/yolov5s.pt"

    model_module = TrainingModule(cfg, evaluator=evaluator, pretrained=True)
    trainer = pl.Trainer(
        accelerator="auto",
        gradient_clip_val=0.5,
        precision=16,
        max_epochs=cfg.data.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.testing.val_interval,
        devices="auto",
        callbacks=list(callbacks),
        logger=list(loggers),
        profiler=profiler,
    )

    trainer.fit(model_module, datamodule=datamodule)


if __name__ == "__main__":
    cfg = OmegaConf.load("./configs/yolov5.yaml")

    os.makedirs(cfg.data.output_dir, exist_ok=True)

    setup_logger(cfg.data.output_dir)

    callbacks = get_callbacks(cfg.data.output_dir)
    profiler = get_profilers(cfg.data.output_dir, filename="perf-train-logs")
    loggers = get_loggers(cfg.data.output_dir)

    train(cfg, loggers, callbacks, profiler)
