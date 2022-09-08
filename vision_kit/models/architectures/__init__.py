from omegaconf.dictconfig import DictConfig
from vision_kit.utils.general import dw_multiple_generator

from .yolov5 import YOLOV5


def build_model(cfg: DictConfig):
    model_name = cfg.model.name
    width, depth = dw_multiple_generator(cfg.model.version)
    if model_name == "YOLOv5":
        model: YOLOV5 = YOLOV5(
            wid_mul=width,
            dep_mul=depth,
            act=cfg.model.act,
            num_classes=cfg.model.num_classes,
            hyp=cfg.hypermeters
        )
    else:
        raise NotImplemented

    return model
