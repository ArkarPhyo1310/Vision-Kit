import os
from multiprocessing.pool import ThreadPool
from typing import Tuple

import cv2
import numpy as np
from loguru import logger
from pycocotools.coco import COCO
from rich.progress import Progress
from yolo_series.utils import remove_useless_info
from yolo_series.utils.bboxes import xywh_to_cxcywh, xywh_to_xyxy, xyxy_to_xywh

from .datasets_wrapper import Dataset

NUM_THREADS = min(8, os.cpu_count())


class COCODataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        json_file: str = "instances_train2017.json",
        name: str = "train2017",
        img_sz: Tuple[int, int] = (640, 640),
        cache: bool = True,
        aug_pipeline: bool = None,
    ) -> None:
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_sz)
        self.data_dir: str = data_dir
        self.json_file: str = json_file
        self.name: str = name
        self.img_sz: Tuple[int, int] = (img_sz, img_sz) if isinstance(
            img_sz, int) else img_sz
        self.aug_pipeline = aug_pipeline
        self.imgs = None

        self.coco: COCO = COCO(os.path.join(self.data_dir, self.json_file))
        # remove_useless_info(self.coco)
        self.ids: list = self.coco.getImgIds()
        self.class_ids: list = sorted(self.coco.getCatIds())

        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self.classes: tuple = tuple([c["name"] for c in self.cats])
        self.annotations = self._load_coco_annotations()
        if cache:
            self._cache_images()

    def __len__(self) -> int:
        return len(self.ids)

    def __del__(self) -> None:
        del self.imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(idx) for idx in self.ids]

    def _cache_images(self):
        logger.info(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM. For COCO need 200G+ RAM space.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_sz[0]
        max_w = self.img_sz[1]
        cache_file = os.path.join(
            self.data_dir, f"img_cache_{self.name}.array")

        if not os.path.exists(cache_file):
            self.imgs = np.memmap(
                cache_file,
                dtype=np.uint8,
                mode="w+",
                shape=(len(self.ids), max_h, max_w, 3)
            )

            loaded_images = ThreadPool(NUM_THREADS).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Processing...", total=len(self.annotations))

                for i, out in enumerate(loaded_images):
                    self.imgs[i][: out.shape[0],
                                 : out.shape[1], :] = out.copy()
                    progress.update(task, advance=i)

            self.imgs.flush()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!\n"
                "Everytime the self.input_size is changed in your exp file, you need to delete\n"
                "the cached data and re-generate them.\n"
            )

        logger.info("Loading cached images...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, index: int):
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(index)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                x = int(np.max((0, obj["bbox"][0])))
                y = int(np.max((0, obj["bbox"][1])))
                w = int(np.min((width, np.max((0, obj["bbox"][2])))))
                h = int(np.min((height, np.max((0, obj["bbox"][3])))))
                obj["clean_bbox"] = [x, y, w, h]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, :4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_sz[0] / height, self.img_sz[1] / width)
        res[:, :4] *= r
        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(index) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def load_anno(self, index: int):
        return self.annotations[index][0]

    def load_resized_img(self, index: int):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.data_dir, self.name, file_name)
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        assert img is not None, f"file named {img_file} not found"

        r = min(self.img_sz[0] / img.shape[0],
                self.img_sz[1] / img.shape[1])

        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def pull_item(self, index: int):
        idx = self.ids[index]

        res, img_info, resized_info, _ = self.annotations[index]
        if self.imgs is not None:
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)

        res = xywh_to_xyxy(res)

        return img, res.copy(), img_info, np.array([idx])

    @Dataset.mosaic_getitem
    def __getitem__(self, index: int):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)
        target = xyxy_to_xywh(target)
        if self.aug_pipeline is not None:
            # If width and height of bbox is 0, discard it.
            img, target = self.aug_pipeline(img, target)

        # if target.shape[0] == 0:
        #     target = np.zeros((1, 5))
        target = xywh_to_cxcywh(target)
        return img, target, img_info, img_id
