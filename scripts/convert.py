from copy import deepcopy
from typing import Any, Dict

import torch

from vision_kit.models.architectures import YOLOV5, YOLOV7
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

    modelv5 = torch.hub.load('/home/arkar/ME/yolov5/', model_name, autoshape=False, force_reload=False, source="local")
    state_dict = modelv5.state_dict()
    state_dict.pop('model.model.24.anchors', None)

    for new, old in zip(new_model.keys(), state_dict.keys()):
        new_model[new] = state_dict[old]

    # with torch.no_grad():
    model.load_state_dict(new_model)
    torch.save(model.half().state_dict(), f"./pretrained_weights/{model_name}.pt")


@torch.no_grad()
def convert_yolov7(version: str = "base") -> None:
    version = version.lower()
    model_name = "yolov7" + version
    assert version in [
        "base", "extra"], f"\"{version}\" is either wrong or unsupported!"

    v7model = YOLOV7(deploy=False).to("cpu")
    v7state_dict: Dict[str, Any] = v7model.state_dict()

    modelv7 = torch.hub.load('/home/arkar/ME/yolov7/', 'custom', path_or_model="/home/arkar/ME/yolov7/yolov7_training.pt",
                             autoshape=False, force_reload=False, source="local", verbose=False)
    state_dict = modelv7.state_dict()
    state_dict.pop('model.105.anchors', None)
    state_dict.pop('model.105.anchor_grid', None)

    for new, old in zip(v7state_dict.keys(), state_dict.keys()):
        v7state_dict[new] = state_dict[old]

    v7model.load_state_dict(v7state_dict)
    torch.save(v7model.half().state_dict(),
               f"./pretrained_weights/{model_name}.pt")


if __name__ == "__main__":
    convert_yolov7()
