import cv2
import numpy as np
import torch
import torchvision
from vision_kit.utils.bboxes import xywhn_to_xyxy


def grid_save(imgs, targets, name="train"):
    img_list = []
    row = int(imgs.shape[0] / 2)
    for idx, (img, labels) in enumerate(zip(imgs, targets)):
        img_arr = img.cpu().numpy()
        img_arr = img_arr.transpose((1, 2, 0))
        bboxes = xywhn_to_xyxy(
            labels[:, 2:], img_arr.shape[1], img_arr.shape[0]).cpu().numpy()
        classes = labels[:, 1]
        for bbox, idx in zip(bboxes, classes):
            x0 = int(bbox[0])
            y0 = int(bbox[1])
            x1 = int(bbox[2])
            y1 = int(bbox[3])

            text = str(int(idx.cpu().numpy().item()))
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.8, 1)[0]
            cv2.rectangle(img_arr, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.rectangle(
                img_arr,
                (x0, y0 + 1),
                (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
                (128, 0, 0),
                -1
            )
            cv2.putText(img_arr, text, (x0, y0 +
                        txt_size[1]), font, 0.8, (255, 255, 255), thickness=2)

        img_transpose = img_arr.transpose((2, 0, 1))
        img_list.append(img_transpose)

    img = np.stack(img_list, 0)
    img_tensor = torch.from_numpy(img)
    batch_grid = torchvision.utils.make_grid(
        img_tensor, normalize=False, nrow=row)
    torchvision.utils.save_image(batch_grid, f"{name}.jpg")

    return batch_grid
