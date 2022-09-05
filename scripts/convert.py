from typing import Any, Dict

import fire
import torch
from yolo_series.models.architectures.yolov5 import YOLOV5
from yolo_series.utils.general import dw_multiple_generator


def convert_yolov5(version: str = "s") -> None:
    version = version.lower()
    model_name = "yolov5" + version
    assert version in ["nano", "s", "m", "l",
                               "x"], f"\"{version}\" is either wrong or unsupported!"

    width, depth = dw_multiple_generator(version)
    model: YOLOV5 = YOLOV5(depth, width, training=False)
    # model.fuse()
    model.eval()
    # model: YOLOV5 = model.fuse()

    new_model: Dict[str, Any] = model.state_dict()

    modelv5 = torch.hub.load('D:\Personal_Projects\yolov5',
                             model_name, autoshape=False, force_reload=False, source="local")
    state_dict = modelv5.state_dict()
    state_dict.pop('model.model.24.anchors', None)

    print(len(new_model.keys()))
    print(len(state_dict.keys()))

    for new, old in zip(new_model.keys(), state_dict.keys()):
        new_model[new] = state_dict[old]

    # with torch.no_grad():
    model.load_state_dict(new_model)
    torch.save(model.state_dict(),
               f"./pretrained_weights/{model_name}.pt")


if __name__ == "__main__":
    fire.Fire(convert_yolov5)
