import math
import random

import albumentations as A
import cv2
import numpy as np


def get_aug_params(value, center=0):
    if isinstance(value, float):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    target_size,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    twidth, theight = target_size

    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(
        translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(
        translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(targets, target_size, M, scale):
    num_gts = len(targets)

    # warp corner points
    twidth, theight = target_size
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1),
             corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight)

    targets[:, :4] = new_bboxes

    return targets


def random_affine(
    img,
    targets=(),
    target_size=(640, 640),
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(
        target_size, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=target_size,
                         borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(targets) > 0:
        targets = apply_affine_to_bboxes(targets, target_size, M, scale)

    return img, targets


class TrainAugPipeline:
    def __init__(
        self,
        flip_lr_prob: float = 0.5,
        flip_ud_prob: float = 0.0,
        hsv_prob: float = 1.0,
        img_sz: tuple = (640, 640),
        bbox_format: str = "coco",
    ) -> None:

        img_sz = (img_sz, img_sz) if isinstance(img_sz, int) else img_sz

        T: list = [
            A.Blur(p=0.01),
            A.MedianBlur(p=0.01),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.RandomBrightnessContrast(p=0.0),
            A.RandomGamma(p=0.0),
            A.ImageCompression(quality_lower=75, p=0.0),
            A.HueSaturationValue(p=hsv_prob),
            A.HorizontalFlip(p=flip_lr_prob),
            A.VerticalFlip(p=flip_ud_prob),
            A.PadIfNeeded(min_height=img_sz[0], min_width=img_sz[1],
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ]
        self.transform: A.Compose = A.Compose(
            T,
            bbox_params=A.BboxParams(
                format=bbox_format, label_fields=["class_labels"])
        )

    def __call__(self, img, labels):
        transformed = self.transform(
            image=img, bboxes=labels[:, 0:4], class_labels=labels[:, 4]
        )

        transformed_img = transformed["image"]
        transformed_ann = np.array(
            [
                [*b, c] for c, b in zip(transformed["class_labels"], transformed["bboxes"])
            ]
        )

        if transformed_ann.ndim < 2:
            transformed_ann = np.zeros((1, 5))

        return transformed_img, transformed_ann


class ValAugPipeline:
    def __init__(
        self,
        img_sz: tuple = (640, 640),
        bbox_format: str = "coco",
    ) -> None:

        img_sz = (img_sz, img_sz) if isinstance(img_sz, int) else img_sz

        T: list = [
            A.PadIfNeeded(min_height=img_sz[0], min_width=img_sz[1],
                          border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        ]
        self.transform: A.Compose = A.Compose(
            T,
            bbox_params=A.BboxParams(
                format=bbox_format, label_fields=["class_labels"])
        )

    def __call__(self, img, labels):
        transformed = self.transform(
            image=img, bboxes=labels[:, 0:4], class_labels=labels[:, 4]
        )

        transformed_img = transformed["image"]
        transformed_ann = np.array(
            [
                [*b, c] for c, b in zip(transformed["class_labels"], transformed["bboxes"])
            ]
        )

        if transformed_ann.ndim < 2:
            transformed_ann = np.zeros((1, 5))

        return transformed_img, transformed_ann
