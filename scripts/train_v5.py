import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.models.architectures import build_model
from vision_kit.utils.model_utils import intersect_dicts, load_ckpt
from pytorch_lightning.plugins.precision import NativeMixedPrecisionPlugin

cfg = OmegaConf.load("./configs/yolov5_yolo.yaml")


weight = "./pretrained_weights/yolov5s.pt"
model = build_model(cfg)
state_dict = torch.load(weight, map_location="cpu")
state_dict = intersect_dicts(state_dict, model.state_dict())
model.load_state_dict(state_dict, strict=False)

datamodule = LitDataModule(
    data_cfg=cfg.data,
    aug_cfg=cfg.augmentations,
    num_workers=0,
    img_sz=cfg.model.input_size,
)
datamodule.setup()


class_ids = datamodule.val_dataloader().dataset.class_ids

# evaluator = COCOEvaluator(img_size=(640, 640), class_ids=class_ids,
#                           gt_json="/home/arkar/Downloads/Compressed/Aquarium_coco/val.json")
evaluator = YOLOEvaluator(img_size=(640, 640))


class DetectionTrainer:
    def __init__(self, cfg, model, evaluator=None) -> None:
        self.data_cfg = cfg.data
        self.hparams = cfg.hypermeters
        self.test_cfg = cfg.testing

        self.model = model
        self.evaluator = evaluator

    def configure_optimizers(self):
        pass
