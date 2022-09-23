import argparse
import logging
import os
import warnings

import pytorch_lightning as pl
from omegaconf import OmegaConf
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.utils.general import mk_output_dir
from vision_kit.utils.logging_utils import logger, setup_logger
from vision_kit.utils.training_helpers import (get_callbacks, get_loggers,
                                               get_profilers)
warnings.filterwarnings(action="ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def main(cfg, opt):

    callbacks = get_callbacks(cfg.data.output_dir, bar_leave=True)
    profiler = get_profilers(cfg.data.output_dir, filename="perf-logs")
    loggers = get_loggers(cfg.data.output_dir) if opt.task == "train" else ()

    datamodule = LitDataModule(
        data_cfg=cfg.data,
        aug_cfg=cfg.augmentations,
        num_workers=cfg.data.num_workers,
        img_sz=cfg.model.input_size,
    )
    evaluator = YOLOEvaluator(
        class_labels=cfg.data.class_labels,
        img_size=cfg.model.input_size
    )
    cfg.model.weight = "./pretrained_weights/yolov5s.pt"

    trainer = pl.Trainer(
        accelerator="gpu",
        gradient_clip_val=10,
        precision=16,
        max_epochs=cfg.data.max_epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=cfg.testing.val_interval,
        devices="auto",
        callbacks=list(callbacks),
        logger=list(loggers),
        profiler=profiler,
        enable_model_summary=False
    )

    model_module = TrainingModule(cfg, evaluator=evaluator, pretrained=True)

    if opt.ckpt_dir:
        filename = "last.ckpt" if opt.task == "train" else "best.ckpt"
        ckpt_path = os.path.join(opt.ckpt_dir, filename)
    else:
        ckpt_path = None

    if opt.task == "train":
        trainer.fit(model_module, datamodule=datamodule, ckpt_path=ckpt_path)
        trainer.test(model_module, datamodule=datamodule, verbose=False)
    elif opt.task == "eval":
        trainer.test(model_module, datamodule=datamodule, verbose=False, ckpt_path=ckpt_path)
    else:
        if ckpt_path:
            model_module = TrainingModule.load_from_checkpoint(
                checkpoint_path=ckpt_path, cfg=cfg, evaluator=evaluator, pretrained=True)
        file_name = f"{cfg.model.name.lower()}_{cfg.model.version}"
        save_path = os.path.join(cfg.data.output_dir, "weights")
        os.makedirs(save_path, exist_ok=True)

        model_module.to_onnx(
            os.path.join(save_path, file_name + ".onnx"),
            opset_version=13,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {
                    0: 'batch',
                    2: 'height',
                    3: 'width'},  # shape(1,3,640,640)
                'output': {
                    0: 'batch',
                    1: 'anchors'}  # shape(1,25200,85)
            }
        )
        model_module.to_torchscript(os.path.join(save_path, file_name + ".tspt"), method="trace", strict=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLOv5 Training with Pytorch Lightning")
    parser.add_argument("task", type=str, choices=["train", "eval", "export"],
                        default="train", help="Please specify task to perform")
    parser.add_argument("--config", "-c", type=str,
                        default="./configs/yolov5.yaml", help="Model/Dataset config file")
    parser.add_argument("--ckpt-dir", "-d", type=str, default=None, help="Checkpoint folder path")
    parser.add_argument("--seed", "-s", type=int, default=21, help="Seeding everything for reproducibility")

    opt = parser.parse_args()

    pl.seed_everything(opt.seed, workers=True)

    cfg = OmegaConf.load(opt.config)
    output_dir = mk_output_dir(cfg.data.output_dir, cfg.model.name)
    setup_logger(output_dir, filename="log.log")
    cfg.data.output_dir = output_dir

    logger.info(f"Global seed set to {opt.seed}")
    main(cfg, opt)
