import os
from warnings import filterwarnings

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from yolo_series.core.evaluator import COCOEvaluator
from yolo_series.core.trainer import TrainingModule
from yolo_series.data.datamodule import COCODataModule
from yolo_series.models.architectures import build_model
from yolo_series.utils.logging_utils import setup_logger

filterwarnings(action="ignore")

cfg = OmegaConf.load("./configs/yolov5.yaml")

output_dir = os.path.join(
    cfg.data.output_dir, cfg.data.experiment_name)

os.makedirs(output_dir, exist_ok=True)

setup_logger(
    file_name="val.log",
    save_dir=output_dir,
    mode='o'
)

datamodule = COCODataModule(
    data_dir=cfg.data.data_dir,
    train_json=cfg.data.train_json,
    val_json=cfg.data.val_json,
    test_json=cfg.data.test_json,
    batch_sz=cfg.data.batch_size,
    img_sz=cfg.model.input_size,
    aug_config=cfg.augmentations
)
datamodule.setup()

class_ids = datamodule.val_dataloader().dataset.class_ids

evaluator = COCOEvaluator(img_size=(640, 640), class_ids=class_ids)

weight = "./pretrained_weights/yolov5s.pt"
model = build_model(cfg)
state_dict = torch.load(weight, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.fuse()
model.eval()

model_module = TrainingModule(cfg, model=model, evaluator=evaluator)

trainer = pl.Trainer(num_sanity_val_steps=0)

trainer.validate(model_module, datamodule=datamodule)
