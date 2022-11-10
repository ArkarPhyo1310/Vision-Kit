from copy import deepcopy
from typing import Any, Dict

import fire
import torch

from vision_kit.models.architectures.yolov5 import YOLOV5
from vision_kit.models.architectures.yolov7 import YOLOV7
from vision_kit.utils.general import dw_multiple_generator
from vision_kit.utils.model_utils import load_ckpt


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


def reparameterization(model: "YOLOV7", ckpt_path: str, exclude: list = []) -> "YOLOV7":
    ckpt_state_dict = torch.load(ckpt_path, map_location=next(model.parameters()).device)

    num_anchors = model.head.num_anchors
    exclude = exclude

    # intersect_state_dict = {k: v for k, v in ckpt_state_dict.items() if k in model.state_dict(
    # ) and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    # model.load_state_dict(intersect_state_dict, strict=False)
    # print(model.head.m[1].bias)
    model = load_ckpt(model, ckpt_state_dict)
    # print(model.head.m[1].bias)

    for i in range((model.head.num_classes + 5) * num_anchors):
        model.state_dict()['head.m.0.weight'].data[i, :, :, :] *= ckpt_state_dict['head.im.0.implicit'].data[:, i, ::].squeeze()
        model.state_dict()['head.m.1.weight'].data[i, :, :, :] *= ckpt_state_dict['head.im.1.implicit'].data[:, i, ::].squeeze()
        model.state_dict()['head.m.2.weight'].data[i, :, :, :] *= ckpt_state_dict['head.im.2.implicit'].data[:, i, ::].squeeze()
    model.state_dict()['head.m.0.bias'].data += ckpt_state_dict['head.m.0.weight'].mul(ckpt_state_dict['head.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['head.m.1.bias'].data += ckpt_state_dict['head.m.1.weight'].mul(ckpt_state_dict['head.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['head.m.2.bias'].data += ckpt_state_dict['head.m.2.weight'].mul(ckpt_state_dict['head.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['head.m.0.bias'].data *= ckpt_state_dict['head.im.0.implicit'].data.squeeze()
    model.state_dict()['head.m.1.bias'].data *= ckpt_state_dict['head.im.1.implicit'].data.squeeze()
    model.state_dict()['head.m.2.bias'].data *= ckpt_state_dict['head.im.2.implicit'].data.squeeze()
    print(model.head.m[1].bias)
    re_model = deepcopy(model)

    return re_model


@torch.no_grad()
def convert_yolov7(version: str = "base") -> None:
    version = version.lower()
    model_name = "yolov7" + version
    assert version in ["tiny", "base", "extra"], f"\"{version}\" is either wrong or unsupported!"

    # model: YOLOV7 = YOLOV7(variant=version, training_mode=True)
    # model.load_state_dict(torch.load("./pretrained_weights/yolov7base.pt"), strict=False)
    # model.to("cuda")
    v7model = YOLOV7(training_mode=True).to("cuda")
    model = reparameterization(v7model, "./pretrained_weights/v7training.pt")
    # model.eval()
    # print(model)
    # exit(0)
    new_model: Dict[str, Any] = model.state_dict()

    modelv7 = torch.hub.load('/home/arkar/ME/yolov7/', 'custom', path_or_model="/home/arkar/ME/yolov7/yolov7.pt",
                             autoshape=False, force_reload=False, source="local", verbose=False)
    # modelv7.fuse()
    # print(modelv7)
    # exit(0)
    state_dict = modelv7.state_dict()
    state_dict.pop('model.105.anchors', None)
    state_dict.pop('model.105.anchor_grid', None)
    # print(modelv7.model[105].m[1].bias)
    # print(state_dict)
    # for key in state_dict.keys():
    #     print(key)

    for (new_k, new_v), (old_k, old_v) in zip(new_model.items(), state_dict.items()):
        s = new_v == old_v
        if not s.any():
            print(new_k, "\t", old_k)

    # print("Check Finish")
    # for new, old in zip(new_model.keys(), state_dict.keys()):
    #     new_model[new] = state_dict[old]

    # # with torch.no_grad():
    # model.load_state_dict(new_model)
    # torch.save(model.half().state_dict(),
    #            f"./pretrained_weights/{model_name}.pt")

    # model.load_state_dict(new_model)

    # print("Re-Check")
    # for (new_k, new_v), (old_k, old_v) in zip(new_model.items(), state_dict.items()):
    #     s = new_v == old_v
    #     if not s.any():
    #         print(new_k, "\t", old_k)


if __name__ == "__main__":
    fire.Fire(convert_yolov7)
