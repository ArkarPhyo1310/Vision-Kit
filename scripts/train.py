import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.models.architectures import build_model
from vision_kit.utils.model_utils import load_ckpt
from vision_kit.utils.training_helpers import (get_callbacks, get_loggers,
                                               get_profilers)

pl.seed_everything(21, workers=True)


def train(cfg, loggers, callbacks, profiler):
    datamodule = LitDataModule(
        data_cfg=cfg.data,
        aug_cfg=cfg.augmentations,
        num_workers=8,
        img_sz=cfg.model.input_size,
    )
    datamodule.setup()
    evaluator = YOLOEvaluator(img_size=(640, 640))

    weight = "./pretrained_weights/yolov5s.pt"
    model = build_model(cfg)
    state_dict = torch.load(weight, map_location="cpu")
    model = load_ckpt(model, state_dict)
    model = model.to("cuda")

    model_module = TrainingModule(cfg, model=model, evaluator=evaluator)
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
    cfg = OmegaConf.load("./configs/yolov5_yolo.yaml")

    callbacks = get_callbacks(cfg.data.output_dir)
    profiler = get_profilers(cfg.data.output_dir, filename="perf-logs")
    loggers = get_loggers(cfg.data.output_dir)

    train(cfg, loggers, callbacks, profiler)
