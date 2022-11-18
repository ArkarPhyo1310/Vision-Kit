from typing import Any, Dict

import torch

from vision_kit.models.architectures import YOLOV5, YOLOV7


@torch.no_grad()
def convert_yolov5(version: str = "s") -> None:
    version = version.lower()
    model_name = "yolov5" + version
    assert version in ["nano", "s", "m", "l", "x"], f"\"{version}\" is either wrong or unsupported!"

    model: YOLOV5 = YOLOV5(variant=version)
    new_model: Dict[str, Any] = model.state_dict()

    modelv5 = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True, autoshape=False)

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
    assert version in ["base", "x"], f"\"{version}\" is either wrong or unsupported!"

    v7model = YOLOV7(variant=version, deploy=False)
    v7state_dict: Dict[str, Any] = v7model.state_dict()

    modelv7 = torch.hub.load("WongKinYiu/yolov7", "custom", autoshape=False,
                             path_or_model="D:\Personal_Projects\yolov7\yolov7x_training.pt")
    state_dict = modelv7.state_dict()

    if version == "base":
        state_dict.pop('model.105.anchors', None)
        state_dict.pop('model.105.anchor_grid', None)
    else:
        state_dict.pop('model.121.anchors', None)
        state_dict.pop('model.121.anchor_grid', None)

    for new, old in zip(v7state_dict.keys(), state_dict.keys()):
        v7state_dict[new] = state_dict[old]

    v7model.load_state_dict(v7state_dict)
    torch.save(v7model.half().state_dict(), f"./pretrained_weights/{model_name}.pt")


if __name__ == "__main__":
    # convert_yolov5("x")
    convert_yolov7("x")
