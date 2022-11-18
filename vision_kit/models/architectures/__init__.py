from omegaconf.dictconfig import DictConfig

from .yolov5 import YOLOV5
from .yolov7 import YOLOV7


def build_model(cfg: DictConfig):
    model_name: str = cfg.model.name
    if model_name == "YOLOv5":
        model: YOLOV5 = YOLOV5(
            variant=cfg.model.version,
            act=cfg.model.act,
            num_classes=cfg.model.num_classes,
            deploy=cfg.model.deploy
        )
    elif model_name == "YOLOv7":
        model: YOLOV7 = YOLOV7(
            variant=cfg.model.version,
            act=cfg.model.act,
            num_classes=cfg.model.num_classes,
            deploy=cfg.model.deploy
        )
    else:
        raise NotImplemented

    return model
