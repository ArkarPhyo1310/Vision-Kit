import os
from warnings import filterwarnings

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.eval.yolo_eval import YOLOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.models.architectures import build_model
from vision_kit.utils.logging_utils import setup_logger

# filterwarnings(action="ignore")

cfg = OmegaConf.load("./configs/yolov5_yolo.yaml")

output_dir = os.path.join(
    cfg.data.output_dir, cfg.data.experiment_name)

os.makedirs(output_dir, exist_ok=True)

# setup_logger(
#     file_name="val.log",
#     save_dir=output_dir,
#     mode='o'
# )

datamodule = LitDataModule(
    data_cfg=cfg.data,
    aug_cfg=cfg.augmentations,
    num_workers=0,
    img_sz=cfg.model.input_size,
)
datamodule.setup()

class_ids = datamodule.val_dataloader().dataset.class_ids

# evaluator = COCOEvaluator(img_size=(640, 640), class_ids=class_ids)

evaluator = YOLOEvaluator(img_size=(640, 640))

weight = "./pretrained_weights/yolov5s.pt"
model = build_model(cfg)
state_dict = torch.load(weight, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.fuse()

model_module = TrainingModule(cfg, model=model, evaluator=evaluator)

trainer = pl.Trainer(num_sanity_val_steps=0)

trainer.validate(model_module, datamodule=datamodule)
