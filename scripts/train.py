import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from vision_kit.core.eval.coco_eval import COCOEvaluator
from vision_kit.core.train.trainer import TrainingModule
from vision_kit.data.datamodule import LitDataModule
from vision_kit.models.architectures import build_model
from vision_kit.utils.logging_utils import setup_logger
from vision_kit.utils.models_utils import load_ckpt

cfg = OmegaConf.load("./configs/yolov5_yolo.yaml")

output_dir = os.path.join(
    cfg.data.output_dir, cfg.data.experiment_name)

os.makedirs(output_dir, exist_ok=True)

# setup_logger(
#     file_name="train.log",
#     save_dir=output_dir
# )

datamodule = LitDataModule(
    data_cfg=cfg.data,
    aug_cfg=cfg.augmentations,
    num_workers=0,
    img_sz=cfg.model.input_size,
)
datamodule.setup()

class_ids = datamodule.val_dataloader().dataset.class_ids
evaluator = COCOEvaluator(img_size=(640, 640), class_ids=class_ids)

weight = "./pretrained_weights/yolov5s.pt"
model = build_model(cfg)
state_dict = torch.load(weight, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model_module = TrainingModule(cfg, model=model, evaluator=evaluator)
trainer = pl.Trainer(
    accelerator="auto",
    max_epochs=cfg.data.max_epochs,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1
)

trainer.fit(model_module, datamodule=datamodule)
