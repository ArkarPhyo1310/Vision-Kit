from omegaconf.dictconfig import DictConfig

from .yolov5 import YOLOV5
from .yolov7 import YOLOV7


def build_model(cfg: DictConfig):
    model_name = cfg.model.name
    if model_name == "YOLOv5":
        model: YOLOV5 = YOLOV5(
            variant=cfg.model.version,
            act=cfg.model.act,
            num_classes=cfg.model.num_classes,
            training_mode=cfg.model.training_mode
        )
    elif model_name == "YOLOv7":
        variant = cfg.model.version
        model = YOLOV7(
            variant=variant,
            act=cfg.model.act,
            num_classes=cfg.model.num_classes,
            training_mode=cfg.model.training_mode
        )
    else:
        raise NotImplemented

    return model
