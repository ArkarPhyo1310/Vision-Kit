import os

import pytorch_lightning as pl
from omegaconf import OmegaConf
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.utils.general import mk_output_dir
from vision_kit.utils.logging_utils import setup_logger
from vision_kit.utils.training_helpers import (get_callbacks, get_loggers,
                                               get_profilers)

pl.seed_everything(21, workers=True)


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
        profiler=profiler,
    )

    trainer.fit(model_module, datamodule=datamodule)


if __name__ == "__main__":
    cfg = OmegaConf.load("./configs/yolov5.yaml")
    output_dir = mk_output_dir(cfg.data.output_dir, cfg.model.name)

    setup_logger(output_dir)

    callbacks = get_callbacks(output_dir)
    profiler = get_profilers(output_dir, filename="perf-train-logs")
    loggers = get_loggers(output_dir)

    train(cfg, loggers, callbacks, profiler)
