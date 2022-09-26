from time import time

import cv2
import torch
from vision_kit.classes.coco import COCO
from vision_kit.demo.processing import ImageProcessor
from vision_kit.models.architectures import YOLOV5
from vision_kit.utils.drawing import Drawing
from vision_kit.utils.general import dw_multiple_generator

width, depth = dw_multiple_generator("m")

model: YOLOV5 = YOLOV5(dep_mul=depth, wid_mul=width, num_classes=80)
model.load_state_dict(torch.load("./pretrained_weights/yolov5m.pt", map_location="cpu"), strict=False)
model.fuse()
model.eval()
model.cuda()

image_processor: ImageProcessor = ImageProcessor(auto=False)
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
    with torch.inference_mode():
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
