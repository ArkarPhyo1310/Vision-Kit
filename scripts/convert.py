from typing import Any, Dict

import fire
import torch

from vision_kit.models.architectures.yolov5 import YOLOV5
from vision_kit.models.architectures.yolov7 import YOLOV7
from vision_kit.utils.general import dw_multiple_generator


@torch.no_grad()
def convert_yolov5(version: str = "s") -> None:
    version = version.lower()
    model_name = "yolov5" + version
    assert version in ["nano", "s", "m", "l",
                               "x"], f"\"{version}\" is either wrong or unsupported!"

    width, depth = dw_multiple_generator(version)
    model: YOLOV5 = YOLOV5(depth, width)
    model.eval()

    new_model: Dict[str, Any] = model.state_dict()

    modelv5 = torch.hub.load('/home/arkar/ME/yolov5/',
                             model_name, autoshape=False, force_reload=False, source="local")
    state_dict = modelv5.state_dict()
    state_dict.pop('model.model.24.anchors', None)

    for new, old in zip(new_model.keys(), state_dict.keys()):
        new_model[new] = state_dict[old]

    # with torch.no_grad():
    model.load_state_dict(new_model)
    torch.save(model.half().state_dict(),
               f"./pretrained_weights/{model_name}.pt")


@torch.no_grad()
def convert_yolov7(version: str = "base") -> None:
    version = version.lower()
    model_name = "yolov7" + version
    assert version in ["tiny", "base", "extra"], f"\"{version}\" is either wrong or unsupported!"

    model: YOLOV7 = YOLOV7(variant=version, training_mode=True)
    model.eval()

    new_model: Dict[str, Any] = model.state_dict()

    modelv7 = torch.hub.load('/home/arkar/ME/yolov7/', 'custom', path_or_model="/home/arkar/ME/yolov7/yolov7_training.pt",
                             autoshape=False, force_reload=False, source="local", verbose=False)
    state_dict = modelv7.state_dict()
    state_dict.pop('model.105.anchors', None)
    state_dict.pop('model.105.anchor_grid', None)

    for new, old in zip(new_model.keys(), state_dict.keys()):
        new_model[new] = state_dict[old]

    # with torch.no_grad():
    model.load_state_dict(new_model)
    torch.save(model.half().state_dict(),
               f"./pretrained_weights/{model_name}.pt")


if __name__ == "__main__":
    fire.Fire(convert_yolov7)
