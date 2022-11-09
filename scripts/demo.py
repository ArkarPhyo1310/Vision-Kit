from time import time

import cv2
import torch

from vision_kit.classes.coco import COCO
from vision_kit.demo.processing import ImageProcessor
from vision_kit.models.architectures import YOLOV5, YOLOV7
from vision_kit.utils.drawing import Drawing

# model: YOLOV5 = YOLOV5(variant="m", num_classes=80, training_mode=False)
# model.load_state_dict(torch.load("./pretrained_weights/yolov5m.pt", map_location="cpu"), strict=False)

v7model = YOLOV7(training_mode=False)
# model.load_state_dict(torch.load("./remodel.pt", map_location="cpu"), strict=False)
# v7model.load_state_dict(torch.load("./pretrained_weights/v7.pt", map_location="cpu"), strict=False)
model = YOLOV7.reparameterization(v7model, "./pretrained_weights/v7.pt")
model.fuse()
model.eval()
model = model.to("cuda")

image_processor: ImageProcessor = ImageProcessor(conf_thres=0, iou_thres=0)
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
