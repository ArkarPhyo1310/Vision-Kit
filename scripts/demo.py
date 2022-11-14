from copy import deepcopy
from time import time

import cv2
import torch

from vision_kit.classes.coco import COCO
from vision_kit.demo.processing import ImageProcessor
from vision_kit.models.architectures import YOLOV5, YOLOV7
from vision_kit.utils.drawing import Drawing
from vision_kit.utils.model_utils import load_ckpt


def reparameterization(model: "YOLOV7", ckpt_path: str, exclude: list = []) -> "YOLOV7":
    ckpt_state_dict = torch.load(ckpt_path, map_location=next(model.parameters()).device) if isinstance(ckpt_path, str) else ckpt_path

    num_anchors = model.head.num_anchors
    exclude = exclude

    # intersect_state_dict = {k: v for k, v in ckpt_state_dict.items() if k in model.state_dict(
    # ) and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    # model.load_state_dict(intersect_state_dict, strict=False)
    model = load_ckpt(model, ckpt_state_dict)

    for i in range((model.head.num_classes + 5) * num_anchors):
        model.state_dict()['head.m.0.weight'].data[i, :, :, :] *= ckpt_state_dict['model.105.im.0.implicit'].data[:, i, ::].squeeze()
        model.state_dict()['head.m.1.weight'].data[i, :, :, :] *= ckpt_state_dict['model.105.im.1.implicit'].data[:, i, ::].squeeze()
        model.state_dict()['head.m.2.weight'].data[i, :, :, :] *= ckpt_state_dict['model.105.im.2.implicit'].data[:, i, ::].squeeze()
    model.state_dict()['head.m.0.bias'].data += ckpt_state_dict['model.105.m.0.weight'].mul(ckpt_state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['head.m.1.bias'].data += ckpt_state_dict['model.105.m.1.weight'].mul(ckpt_state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['head.m.2.bias'].data += ckpt_state_dict['model.105.m.2.weight'].mul(ckpt_state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['head.m.0.bias'].data *= ckpt_state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['head.m.1.bias'].data *= ckpt_state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['head.m.2.bias'].data *= ckpt_state_dict['model.105.im.2.implicit'].data.squeeze()

    re_model = deepcopy(model)

    return re_model

# model: YOLOV5 = YOLOV5(variant="m", num_classes=80, training_mode=False)
# model.load_state_dict(torch.load("./pretrained_weights/yolov5m.pt"), strict=False)


model = YOLOV7(training_mode=False).to("cuda")
model.load_state_dict(torch.load("./pretrained_weights/yolov7base.pt"), strict=False)
model = YOLOV7.reparameterization(model, "./pretrained_weights/yolov7_train.pt")


# v7model = torch.hub.load('/home/myat/ME/yolov7/', 'custom', path_or_model="/home/myat/ME/yolov7/yolov7.pt",
                        #  autoshape=False, force_reload=False, source="local", verbose=False)
# model = reparameterization(model, v7model.state_dict())
# model = load_ckpt(model, v7model.state_dict())
# model = v7model

model.to("cuda")
model.fuse()
model.eval()

image_processor: ImageProcessor = ImageProcessor(conf_thres=0.2, iou_thres=0.45)
drawer: Drawing = Drawing(COCO)
webcam = cv2.VideoCapture(0)

while True:
    has_frame, frame = webcam.read()
    if not has_frame:
        break
    dummy_input = frame
    pre_start_time = time()
    x, _ = image_processor.preprocess(dummy_input)
    pre_proc_time = (time() - pre_start_time) * 1e3

    inf_start_time = time()
    with torch.no_grad():
        y = model(x.cuda())
    inf_time = (time() - inf_start_time) * 1e3

    post_start_time = time()
    i = image_processor.postprocess(y[0])
    dummy_input = drawer.draw(dummy_input, i)
    post_proc_time = (time() - post_start_time) * 1e3

    print(f"Pre: {pre_proc_time:.1f} ms, Inf: {inf_time:.1f} ms, Post: {post_proc_time:.1f} ms => Total: {(pre_proc_time+inf_time+post_proc_time):.1f} ms")

    cv2.imshow("Demo", dummy_input)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
