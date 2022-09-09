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

output_dir = os.path.join(
    cfg.data.output_dir, cfg.data.experiment_name)

os.makedirs(output_dir, exist_ok=True)

progress_bar = RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    ),
    leave=True
)

amp = NativeMixedPrecisionPlugin(
    precision="torch.float16",
    device="cuda",
    scaler=torch.cuda.amp.GradScaler(enabled=True)
)


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

# evaluator = COCOEvaluator(img_size=(640, 640), class_ids=class_ids,
#                           gt_json="/home/arkar/Downloads/Compressed/Aquarium_coco/val.json")
evaluator = YOLOEvaluator(img_size=(640, 640))

weight = "./pretrained_weights/yolov5s.pt"
model = build_model(cfg)
state_dict = torch.load(weight, map_location="cpu")
# model.load_state_dict(state_dict, strict=False)
# model = load_ckpt(model, state_dict)

state_dict = intersect_dicts(state_dict, model.state_dict())
model.load_state_dict(state_dict, strict=False)
# exit()

model_module = TrainingModule(cfg, model=model, evaluator=evaluator)
trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=20,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1,
    devices=1,
    callbacks=[progress_bar],
    plugins=[amp]
)

trainer.fit(model_module, datamodule=datamodule)
