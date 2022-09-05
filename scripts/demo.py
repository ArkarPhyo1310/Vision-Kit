import cv2
import numpy as np
import torch
from yolo_series.data.processing import ImageProcessor
from yolo_series.models.architectures import YOLOV5
from yolo_series.utils.general import dw_multiple_generator
from yolo_series.utils.postprocess import nms, scale_coords

width, depth = dw_multiple_generator("s")

model: YOLOV5 = YOLOV5(dep_mul=depth, wid_mul=width, training=False)
model.load_state_dict(torch.load(
    "./pretrained_weights/yolov5s.pt", map_location="cpu"))

model.fuse()
model.eval()

dummy_input = cv2.imread("./assets/cat.jpg")
image_processor: ImageProcessor = ImageProcessor(auto=False)

x = image_processor.preprocess(dummy_input)
# exit(0)
with torch.no_grad():
    y = model(x)
i = image_processor.postprocess(y[0])
print(i)

for pred in i:
    bbox = pred[:4]
    cls = pred[-1]
    bbox = bbox.numpy()
    cls = cls.numpy()
    (x0, y0), (x1, y1) = (int(pred[0]), int(
        pred[1])), (int(pred[2]), int(pred[3]))
    cls = int(cls)
    from omegaconf import OmegaConf
    COCO_CLASSES = OmegaConf.load(
        "./configs/coco_classes.yaml")['COCO']
    cv2.rectangle(
        dummy_input, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), color=(255, 0, 0), thickness=1
    )
    text = '{}'.format(COCO_CLASSES[cls])
    txt_color = (0, 0, 0) if np.mean(
        (255, 0, 0)) > 0.5 else (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
    txt_bk_color = ((255, 0, 0))
    cv2.rectangle(
        dummy_input,
        (x0, y0 + 1),
        (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
        txt_bk_color,
        -1
    )
    cv2.putText(
        dummy_input, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

cv2.imshow("Demo", dummy_input)
cv2.waitKey(0)
cv2.destroyAllWindows()
