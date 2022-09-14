import os
from warnings import filterwarnings

import pytorch_lightning as pl
from omegaconf import OmegaConf
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.utils.logging_utils import setup_logger
from vision_kit.utils.training_helpers import get_callbacks, get_loggers, get_profilers

filterwarnings(action="ignore")

cfg = OmegaConf.load("./configs/yolov5.yaml")
os.makedirs(cfg.data.output_dir, exist_ok=True)

setup_logger(
    path=cfg.data.output_dir,
    filename="val.log",
)

callbacks = get_callbacks(cfg.data.output_dir)
profiler = get_profilers(cfg.data.output_dir, filename="perf-test-logs")
loggers = get_loggers(cfg.data.output_dir)

datamodule = LitDataModule(
    data_cfg=cfg.data,
    aug_cfg=cfg.augmentations,
    num_workers=cfg.data.num_workers,
    img_sz=cfg.model.input_size,
)
datamodule.setup()
evaluator = YOLOEvaluator(
    class_labels=cfg.data.class_labels,
    img_size=cfg.model.input_size,
    details_per_class=True
)

cfg.model.weight = "./pretrained_weights/yolov5s.pt"
model_module = TrainingModule.load_from_checkpoint(
    "outputs/ckpts/epoch=18-mAP@.5=0.67.ckpt", cfg=cfg, evaluator=evaluator, pretrained=True
)

trainer = pl.Trainer(
    accelerator="auto",
    devices="auto",
    gradient_clip_val=0.5,
    precision=16,
    max_epochs=cfg.data.max_epochs,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=cfg.testing.val_interval,
    callbacks=list(callbacks),
    logger=list(loggers),
    profiler=profiler
)

trainer.validate(model_module, datamodule.val_dataloader())
