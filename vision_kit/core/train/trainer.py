from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torchinfo import summary
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from vision_kit.models.architectures import build_model
from vision_kit.utils.drawing import grid_save
from vision_kit.utils.image_proc import nms
from vision_kit.utils.logging_utils import logger
from vision_kit.utils.model_utils import (ModelEMA, extract_ema_weight,
                                          load_ckpt, remove_ema_weight)
from vision_kit.utils.table import RichTable
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger


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
            self.train_batch_grid = grid_save(imgs, targets, name=f"{self.data_cfg.output_dir}/train")

        outputs = self.model(imgs)
        targets = torch.cat(targets, 0)
        loss, loss_items = self.model.head.compute_loss(outputs, targets)

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
        predn, targetn = self.evaluator.evaluate(
            img=imgs, img_infos=img_infos, preds=output, targets=targets
        )

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
        self.log("loss", outputs[0])
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({
                    "samples/train": [wandb.Image(f"{self.data_cfg.output_dir}/train.jpg")]
                })
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image('samples/train', self.train_batch_grid, 0)

    def validation_epoch_end(self, outputs) -> None:
        map50, map95, _ = self.evaluator.evaluate_predictions()
        self.log("mAP@.5", map50, prog_bar=True)
        self.log("mAP@.5:.95", map95, prog_bar=True)

        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.log({
                    "samples/val": [wandb.Image(f"{self.data_cfg.output_dir}/val.jpg")]
                })
            elif isinstance(logger, TensorBoardLogger):
                logger.experiment.add_image('samples/val', self.val_batch_grid, 0)

    def test_epoch_end(self, outputs) -> None:
        for trainer_logger in self.loggers:
            if isinstance(trainer_logger, WandbLogger):
                trainer_logger.experiment.log({
                    "samples/test": [wandb.Image(f"{self.data_cfg.output_dir}/test.jpg")]
                })
            elif isinstance(trainer_logger, TensorBoardLogger):
                trainer_logger.experiment.add_image('samples/test', self.test_batch_grid, 0)

        logger.info("Testing finished...")
        map50, map95, self.per_class_table = self.evaluator.evaluate_predictions(details_per_class=True)
        logger.info(self.per_class_table.table)

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
        lr_scheduler = LambdaLR(optimizer, lr_lambda=lf)

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

    def on_train_start(self) -> None:
        # self.lr_scheduler.last_epoch = -1
        self.nw = max(round(self.hyp_cfg['warmup_epochs'] * self.trainer.num_training_batches), 100)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["model"] = self.get_model(half=True).state_dict()

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
        else:
            checkpoint["state_dict"] = remove_ema_weight(checkpoint)

    @torch.no_grad()
    def to_torchscript(self, file_path: Optional[Union[str, Path]] = None, method: Optional[str] = "script",
                       example_inputs: Optional[Any] = None, **kwargs):
        script_model = self.get_model()

        script_model.head.export = True
        script_model.fuse()
        script_model.eval()

        logger.info(f'Starting export with torch {torch.__version__}...')
        if method == "script":
            ts = torch.jit.script(script_model, **kwargs)
        elif method == "trace":
            # if no example inputs are provided, try to see if model has example_input_array set
            if example_inputs is None:
                if self.example_input_array is None:
                    raise ValueError(
                        "Choosing method=`trace` requires either `example_inputs`"
                        " or `model.example_input_array` to be defined."
                    )
                example_inputs: torch.Tensor = self.example_input_array

            example_inputs = example_inputs.to(self.device)
            script_model = script_model.to(self.device)
            ts = torch.jit.trace(script_model, example_inputs=example_inputs, **kwargs)
        else:
            raise ValueError(f"The 'method' parameter only supports 'script' or 'trace', but value given was: {method}")

        ts.save(file_path)
        logger.info(f'Saved torchscript model @ {file_path}')

    @torch.no_grad()
    def to_onnx(
            self, file_path: Union[str, Path],
            input_sample: Optional[Any] = None, simplify: bool = True, **kwargs):
        import onnx

        model = self.get_model()
        model.head.export = True
        model.fuse()
        model.eval()

        if input_sample is None:
            if self.example_input_array is None:
                raise ValueError(
                    "Onnx conversion requires either `input_sample`"
                    " or `model.example_input_array` to be defined."
                )
            input_sample: torch.Tensor = self.example_input_array
        input_sample = input_sample.cpu()
        model = model.cpu()

        torch.onnx.export(
            model,
            input_sample,
            file_path,
            **kwargs
        )
        # Checks
        model_onnx = onnx.load(file_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify
        if simplify:
            try:
                import onnxsim

                logger.info(f'Simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, 'assert check failed'
                onnx.save(model_onnx, file_path)
            except Exception as e:
                logger.info(f'Simplifier failure: {e}')
        logger.info(f'Saved onnx model @ {file_path}')

    def get_model(self, half: bool = False):
        if self.ema_model:
            tmp_model = self.ema_model.module
        else:
            tmp_model = self.model
        model = deepcopy(tmp_model)
        if half:
            model.half()

        del tmp_model

        return model

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
