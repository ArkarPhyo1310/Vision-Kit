import argparse
import os
from time import time

import cv2
import torch

from vision_kit.classes.coco import COCO
from vision_kit.demo.processing import ImageProcessor
from vision_kit.models.architectures import YOLOV5, YOLOV7
from vision_kit.utils.demo_helpers import IMAGE_EXT, ImageReader, VideoReader
from vision_kit.utils.drawing import Drawing

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes: list[str] = COCO


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Demo Script for Vision Kit")
    parser.add_argument("-p", "--path", type=str, default="0",
                        help="Path for image and video or 0 for webcam")
    parser.add_argument("-m", "--model", type=str, default="yolov5", choices=["yolov5", "yolov7"],
                        help="Name of the model architecture")
    parser.add_argument("-v", "--variant", type=str, default="s", choices=["s", "m", "l", "x", "base"],
                        help="Version of the model")
    parser.add_argument("-w", "--weight", type=str, required=True,
                        help="Weight of the model")

    return parser.parse_args()


def load_model(name: str, version: str, ckpt: str) -> YOLOV5 | YOLOV7:
    if name == "yolov5":
        model = YOLOV5(variant=version).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    elif name == "yolov7":
        model = YOLOV7(variant=version).to(device)
        YOLOV7.reparameterization(model, ckpt)
    else:
        raise ValueError(f"{name} is wrong! Must be 'YOLOv5' or 'YOLOv7'.")

    model.fuse()
    model.eval()

    return model


def main(opt: argparse.ArgumentParser) -> None:

    model = load_model(opt.model, opt.variant, opt.weight)
    image_processor: ImageProcessor = ImageProcessor(auto=False)
    drawer: Drawing = Drawing(class_names=classes)

    path_type = os.path.splitext(opt.path)[1]
    source = "image" if path_type in IMAGE_EXT or os.path.isdir(opt.path) else "video"
    data_flow = ImageReader(opt.path) if source == "image" else VideoReader(opt.path, use_thread=False)

    for has_frame, frame in data_flow:
        key = cv2.waitKey(0 if source == "image" else 1)
        if key == ord("q") or not has_frame:
            data_flow.stop()
            break

        pre_start_time = time()
        x, _ = image_processor.preprocess(frame)
        pre_proc_time = (time() - pre_start_time) * 1e3

        inf_start_time = time()
        with torch.inference_mode():
            y = model(x.to(device))
        inf_time = (time() - inf_start_time) * 1e3

        post_start_time = time()
        i = image_processor.postprocess(y[0])
        frame = drawer.draw(frame, i)
        post_proc_time = (time() - post_start_time) * 1e3

        print(f"Pre: {pre_proc_time:.1f} ms, Inf: {inf_time:.1f} ms, Post: {post_proc_time:.1f} ms \t => Total: {(pre_proc_time+inf_time+post_proc_time):.1f} ms")

        cv2.imshow("Demo", frame)


if __name__ == "__main__":
    main(get_args())
