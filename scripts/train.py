import logging

import pytorch_lightning as pl
from omegaconf import OmegaConf
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.utils.general import mk_output_dir
from vision_kit.utils.logging_utils import setup_logger
from vision_kit.utils.training_helpers import (get_callbacks, get_loggers,
                                               get_profilers)

pl.seed_everything(98, workers=True)
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def main(cfg, loggers, callbacks, profiler, task):
    datamodule = LitDataModule(
        data_cfg=cfg.data,
        aug_cfg=cfg.augmentations,
        num_workers=cfg.data.num_workers,
        img_sz=cfg.model.input_size,
    )
    # datamodule.setup()
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

    if task == "train":
        model_module = TrainingModule(cfg, evaluator=evaluator, pretrained=True)

        trainer.fit(model_module, datamodule=datamodule)
        trainer.test(model_module, datamodule=datamodule, verbose=False)
    else:
        model_module = TrainingModule.load_from_checkpoint(
            "outputs/YOLOv5/20220921123538/ckpts/epoch=0-mAP@.5=0.02.ckpt", cfg=cfg, evaluator=evaluator,
            pretrained=True)
        trainer.test(model_module, datamodule=datamodule, verbose=False)
    model_module.to_torchscript("test1.pt", method="trace", strict=False)


if __name__ == "__main__":
    cfg = OmegaConf.load("./configs/yolov5.yaml")
    output_dir = mk_output_dir(cfg.data.output_dir, cfg.model.name)

    setup_logger(output_dir)

    callbacks = get_callbacks(output_dir, bar_leave=True)
    profiler = get_profilers(output_dir, filename="perf-logs")
    loggers = get_loggers(output_dir)

    main(cfg, loggers, callbacks, profiler, task="train")
