from time import time

import cv2
import torch

from vision_kit.classes.coco import COCO
from vision_kit.demo.processing import ImageProcessor
from vision_kit.models.architectures import YOLOV5, YOLOV7
from vision_kit.utils.drawing import Drawing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name: str, ckpt: str) -> YOLOV5 | YOLOV7:
    if name == "YOLOv5":
        model = YOLOV5().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    elif name == "YOLOv7":
        model = YOLOV7(deploy=True).to(device)
        YOLOV7.reparameterization(model, ckpt)
    else:
        raise ValueError(f"{name} is wrong! Must be 'YOLOv5' or 'YOLOv7'.")

    model.fuse()
    model.eval()

    return model


def main():
    model = load_model("YOLOv7", "./pretrained_weights/yolov7base.pt")

    image_processor: ImageProcessor = ImageProcessor(auto=False)
    drawer: Drawing = Drawing(COCO)
    dummy_input = cv2.imread("./assets/zidane.jpg")

    pre_start_time = time()
    x, _ = image_processor.preprocess(dummy_input)
    pre_proc_time = (time() - pre_start_time) * 1e3

    inf_start_time = time()
    with torch.inference_mode():
        y = model(x.to(device))
    inf_time = (time() - inf_start_time) * 1e3

    post_start_time = time()
    i = image_processor.postprocess(y[0])
    dummy_input = drawer.draw(dummy_input, i)
    post_proc_time = (time() - post_start_time) * 1e3

    print(f"Pre: {pre_proc_time:.1f} ms, Inf: {inf_time:.1f} ms, Post: {post_proc_time:.1f} ms => Total: {(pre_proc_time+inf_time+post_proc_time):.1f} ms")

    cv2.imshow("Demo", dummy_input)
    key = cv2.waitKey(0)
    if key == ord("q"):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
