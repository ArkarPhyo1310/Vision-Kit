import cv2
import numpy as np
import torch
import torchvision
from PIL import ImageColor
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


class COLOR:
    """Color Generator class
    """

    def __init__(self, color_fmt: str = "rgb") -> None:
        self._predefined_colors = [
            "#00FFFF", "#7FFFD4", "#838B8B", "#E3CF57", "#FFEBCD", "#0000FF", "#8A2BE2",
            "#9C661F", "#A52A2A", "#DEB887", "#8A360F", "#5F9EA0", "#FF6103", "#FF9912",
            "#7FFF00", "#D2691E", "#3D59AB", "#3D9140", "#808A87", "#FF7F50", "#6495ED",
            "#DC143C", "#B8860B", "#A9A9A9", "#006400", "#BDB76B", "#556B2F", "#FF8C00",
            "#9932CC", "#E9967A", "#8FBC8F", "#483D8B", "#2F4F4F", "#00CED1", "#9400D3",
            "#FF1493", "#00BFFF", "#1E90FF", "#FCE6C9", "#00C957", "#B22222", "#FF7D40",
            "#FFFAF0", "#228B22", "#DCDCDC", "#F8F8FF", "#FFD700", "#DAA520", "#008000",
            "#ADFF2F", "#FF69B4", "#B0171F", "#4B0082", "#F0E68C", "#E6E6FA", "#7CFC00",
            "#ADD8E6", "#F08080", "#FAFAD2", "#D3D3D3", "#FFB6C1", "#8B5F65", "#FFA07A",
            "#20B2AA", "#87CEFA", "#8470FF", "#778899", "#B0C4DE", "#32CD32", "#FAF0E6",
            "#FF00FF", "#03A89E", "#800000", "#BA55D3", "#9370DB", "#3CB371", "#7B68EE",
            "#00FA9A", "#48D1CC", "#C71585", "#E3A869", "#191970", "#BDFCC9", "#F5FFFA",
            "#FFE4B5", "#000080", "#FDF5E6", "#808000", "#6B8E23", "#FF8000", "#FF4500",
            "#DA70D6", "#EEE8AA", "#98FB98", "#BBFFFF", "#DB7093", "#FFEFD5", "#FFDAB9",
            "#33A1C9", "#FFC0CB", "#DDA0DD", "#B0E0E6", "#800080", "#872657", "#C76114",
            "#FF0000", "#BC8F8F", "#4169E1", "#FA8072", "#F4A460", "#308014", "#5E2612",
            "#8E388E", "#C5C1AA", "#71C671", "#7D9EC0", "#AAAAAA", "#8E8E38", "#C67171",
            "#7171C6", "#388E8E", "#A0522D", "#C0C0C0", "#87CEEB", "#6A5ACD", "#708090",
            "#00FF7F", "#4682B4", "#D2B48C", "#008080", "#D8BFD8", "#FF6347", "#40E0D0",
            "#00C78C", "#EE82EE", "#D02090", "#808069", "#F5DEB3", "#FFFFFF", "#FFFF00",
        ]

        assert color_fmt in ["rgb", "bgr"], f"Color format: {color_fmt} is not supported!"

        if color_fmt == "rgb":
            self._color_func = COLOR._rgb_format
        elif color_fmt == "bgr":
            self._color_func = COLOR._bgr_format

    @classmethod
    def _rgb_format(cls, code: str) -> tuple:
        """Returns color in rgb format"""
        rgb = ImageColor.getrgb(code)
        return rgb

    @classmethod
    def _bgr_format(cls, code: str) -> tuple:
        """Returns color in bgr format"""
        rgb = list(ImageColor.getrgb(code))
        rgb[0], rgb[2] = rgb[2], rgb[0]
        return tuple(rgb)

    def __call__(self, index: int):
        color_code = self._predefined_colors[int(index) % len(self._predefined_colors)]
        return self._color_func(color_code)


class Drawing:
    def __init__(self, class_names: list) -> None:
        self._class_names = class_names
        self._color = COLOR(color_fmt="bgr")

    def draw(self, img: np.ndarray, dets: list, filled: bool = False) -> np.ndarray:
        for det in dets:
            pred = det.cpu().numpy()

            bbox = list(map(int, pred[:4].tolist()))
            x0, y0, x1, y1 = bbox
            label = int(pred[-1])
            score = pred[-2].item()

            color = self._color(label)
            text = "{}:{:.1f}%".format(self._class_names[label], score * 100)
            txt_color = (0, 0, 0) if (np.mean(color) / 255) > 0.5 else (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            if filled:
                overlay = img.copy()
                cv2.rectangle(
                    overlay, (x0, y0), (x1, y1),
                    color, -1
                )
                alpha = 0.5
                img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            else:
                cv2.rectangle(
                    img, (x0, y0), (x1, y1),
                    color, 2
                )

            cv2.rectangle(
                img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                color, -1
            )
            cv2.putText(
                img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, 1
            )

        return img
