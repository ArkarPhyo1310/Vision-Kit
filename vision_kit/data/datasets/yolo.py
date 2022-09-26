import glob
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from PIL import Image, ImageOps
from typing import Any, List, Tuple

import cv2
import numpy as np
from rich.progress import Progress
from vision_kit.data.datasets.datasets_wrapper import Dataset
from vision_kit.utils.bboxes import xywhn_to_xyxy, xyxy_to_xywhn
from vision_kit.utils.general import exif_size
from vision_kit.utils.logging_utils import logger

# include image suffixes
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'
NUM_THREADS = min(8, os.cpu_count())


class YOLODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        filter_class: list = [],
        img_sz: Tuple[int, int] = (640, 640),
        cache_type: str = None,
        aug_pipeline: Any = None,
    ) -> None:
        super().__init__(img_sz)

        self.img_sz = img_sz
        self.img_files = self.get_img_files(data_path)  # images
        self.label_files = self.get_label_files(self.img_files)  # labels
        self.aug_pipeline = aug_pipeline
        self.cache_type = cache_type
        # self.class_ids = coco80_to_coco91_class()
        self.class_ids = [0, 1, 2, 3, 4, 5, 6]

        assert cache_type in ["ram", "storage", "none",
                              None], f"Cache Type: {cache_type} is not supported."

        # Caching
        cache_path = (
            Path(data_path) if os.path.isfile(data_path) else Path(
                self.label_files[0]).parent
        ).with_suffix('.cache')
        try:
            cache, exist = np.load(cache_path, allow_pickle=True).item(), True
        except Exception:
            cache, exist = self.cache_labels(cache_path), False

        # found, missing, empty, corrupt, total
        num_found, num_missing, num_empty, num_corrupt, total = cache.pop(
            'results')
        if exist:
            logger.info(
                f"Loaded cache labels: {num_found} found, {num_missing} missing, {num_empty} empty, {num_corrupt} corrupt."
            )

        labels, shapes = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes)
        self.img_files = list(cache.keys())
        self.label_files = self.get_label_files(cache.keys())
        self.total = total

        self.class_filtering(filter_class)
        self.cache_images()

    def __len__(self):
        return len(self.img_files)

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        img, targets, orig_sz, idx = self.pull_item(index)
        targets = xyxy_to_xywhn(
            targets, w=self.resized_sz[1], h=self.resized_sz[0])
        if self.aug_pipeline is not None:
            # If width and height of bbox is 0, discard it.
            img, targets = self.aug_pipeline(img, targets)

        return img, targets, orig_sz, idx

    def pull_item(self, index: int):
        if self.imgs.count(None) == len(self.imgs):
            img, orig_sz, self.resized_sz = self.load_resized_image(index)
        else:
            img = self.imgs[index]
            orig_sz = self.orig_hw[index]
            self.resized_sz = self.resized_hw[index]

        labels = self.labels[index]

        bboxes = xywhn_to_xyxy(
            labels[:, 1:5], w=self.resized_sz[1], h=self.resized_sz[0])
        classes = labels[:, 0].reshape(-1, 1)
        targets = np.hstack((bboxes, classes))

        return img, targets, orig_sz, np.array([index])

    def load_anno(self, index: int):
        return self.labels[index]

    def class_filtering(self, filter_class: list):
        include_class_array = np.array(filter_class).reshape(1, -1)
        for i, label in enumerate(self.labels):
            if filter_class:
                j = (label[:, 0:1] == include_class_array).any(1)
                self.labels[i] = label[j]

    def cache_images(self):
        self.imgs = [None] * self.total
        self.npy_files = [Path(f).with_suffix('.npy') for f in self.img_files]
        if self.cache_type:
            gb = 0  # Gigabytes of cached images
            self.orig_hw, self.resized_hw = [
                None] * self.total, [None] * self.total
            cache_fn = self.cache_images_to_disk if self.cache_type == 'storage' else self.load_resized_image
            results = ThreadPool(NUM_THREADS).imap(cache_fn, range(self.total))

            with Progress() as progress:
                task = progress.add_task(
                    "[green]Caching images ...", total=len(self.total)
                )
                for i, x in enumerate(results):
                    if self.cache_type == "storage":
                        gb += self.npy_files[i].stat().st_size
                    else:
                        self.imgs[i], self.orig_hw[i], self.resized_hw[i] = x
                        gb += self.imgs[i].nbytes

                progress.update(
                    task, advance=i, description=f"[green]Caching images ({gb / 1e9:.1f}GB) {self.cache_type}")

    def cache_images_to_disk(self, index: int):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[index]
        if not f.exists():
            img = cv2.imread(self.img_files[index])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            np.save(f.as_posix(), img)

    def load_resized_image(self, index: int):
        # Loads 1 image from dataset index 'i', returns (im, original hw, resized hw)
        img, file, npy_file = self.imgs[index], self.img_files[index], self.npy_files[index],
        if img is None:  # not cached in RAM
            if npy_file.exists():  # load npy
                img = np.load(npy_file)
            else:  # read image
                img = cv2.imread(file)  # BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                assert img is not None, f'Image Not Found {file}'
            orig_h, orig_w = img.shape[:2]  # orig hw
            r = max(self.img_sz) / max(orig_h, orig_w)  # ratio
            if r != 1:  # if sizes are not equal
                img = cv2.resize(img, (int(orig_w * r), int(orig_h * r)),
                                 interpolation=cv2.INTER_LINEAR)
            # im, hw_original, hw_resized
            return img, (orig_h, orig_w), img.shape[:2]
        # im, hw_original, hw_resized
        return self.imgs[index], self.orig_hw[index], self.resized_hw[index]

    def cache_labels(self, path=Path("./labels.cache")):
        x = {}
        num_miss, num_found, num_empty, num_corrupt = 0, 0, 0, 0
        results = ThreadPool(NUM_THREADS).starmap(
            lambda img_f, lbl_f: self.check_data(img_f, lbl_f),
            zip(self.img_files, self.label_files),
        )

        with Progress() as progress:
            task = progress.add_task(
                "[green]Processing...", total=len(self.img_files))

            for i, (img_f, lbl, shape, nm, nf, ne, nc) in enumerate(results):
                num_miss += nm
                num_found += nf
                num_empty += ne
                num_corrupt += nc
                if img_f:
                    x[img_f] = [lbl, shape]
                progress.update(task, advance=i)

            logger.info(
                f"{num_found} found, {num_miss} missing, {num_empty} empty, {num_corrupt} corrupt.")

        if num_found:
            logger.warning(f"No labels found in {path}")

        x['results'] = num_found, num_miss, num_empty, num_corrupt, len(
            self.img_files)

        try:
            np.save(path, x)
            path.with_suffix(".cache.npy").rename(path)
            logger.info(f'New cache created: {path}')
        except Exception as e:
            raise Exception(
                f'Cache directory {path.parent} is not writeable: {e}')

        return x

    def check_data(self, img_file, label_file):
        num_miss, num_found, num_empty, num_corrupt = 0, 0, 0, 0
        try:
            im_file, shape = self.check_img(img_file)
            lbl, num_miss, num_found, num_empty, num_corrupt = self.check_label(
                label_file)
        except Exception as e:
            num_corrupt = 1
            im_file = None
            lbl = None
            shape = None
            logger.warning(f'{im_file}: ignoring corrupt image/label: {e}')

        return im_file, lbl, shape, num_miss, num_found, num_empty, num_corrupt

    @staticmethod
    def get_img_files(data_path):
        try:
            files = []  # image files
            for p in data_path if isinstance(data_path, list) else [data_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    files += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        # local to global path
                        files += [x.replace('./', parent)
                                  if x.startswith('./') else x for x in t]
                else:
                    raise FileNotFoundError(f'{p} does not exist')
            img_files = sorted(x.replace('/', os.sep)
                               for x in files if x.split('.')[-1].lower() in IMG_FORMATS)
            assert img_files, f'No images found'
        except Exception as e:
            raise Exception(f'Error loading data from {data_path}')

        return img_files

    @staticmethod
    def get_label_files(img_paths):
        # Define label paths as a function of image paths
        # /images/, /labels/ substrings
        sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
        return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]

    @staticmethod
    def check_img(img_file):
        img = Image.open(img_file)
        img.verify()
        shape = exif_size(img)
        assert (shape[0] > 9) & (
            shape[1] > 9), f'image size {shape} <10 pixels'
        assert img.format.lower(
        ) in IMG_FORMATS, f'invalid image format {img.format}'
        if img.format.lower() in ('jpg', 'jpeg'):
            with open(img_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':  # corrupt JPEG
                    ImageOps.exif_transpose(Image.open(img_file)).save(
                        img_file, 'JPEG', subsampling=0, quality=100)
                    logger.warning(
                        f'{img_file}: corrupt JPEG restored and saved')
        return img_file, shape

    @staticmethod
    def check_label(label_file):
        num_missing, num_found, num_empty, num_corrupt = 0, 0, 0, 0
        if os.path.isfile(label_file):
            num_found = 1
            with open(label_file) as f:
                lbl = [x.split()
                       for x in f.read().strip().splitlines() if len(x)]
                lbl = np.array(lbl, dtype=np.float32)

            num_label = len(lbl)
            if num_label:
                assert lbl.shape[1] == 5, f'labels require 5 columns, {lbl.shape[1]} columns detected'
                assert (lbl >= 0).all(
                ), f'negative label values {lbl[lbl < 0]}'
                assert (lbl[:, 1:] <= 1).all(
                ), f'non-normalized or out of bounds coordinates {lbl[:, 1:][lbl[:, 1:] > 1]}'
                _, i = np.unique(lbl, axis=0, return_index=True)
                if len(i) < num_label:
                    lbl = lbl[i]
                    logger.warning(
                        f"{label_file}: {num_label - len(i)} duplicate labels removed")
            else:
                num_empty = 1
                lbl = np.zeros((0, 5), dtype=np.float32)
        else:
            num_missing = 1
            lbl = np.zeros((0, 5), dtype=np.float32)

        return lbl, num_missing, num_found, num_empty, num_corrupt


if __name__ == "__main__":
    yolo = YOLODataset(
        data_path="D:\Personal_Projects\datasets\coco128",
        filter_class=[45, 22]
    )
    print(yolo.load_ann(0))
    # exit()

    img, label, size, idx = yolo.__getitem__(0)
    print(img.shape)
    print(size)
    print(label)
    label = xywhn_to_xyxy(label, w=img.shape[1], h=img.shape[0])
    # label = xywhn_to_xyxy(label, w=yolo.resized_sz[1], h=yolo.resized_sz[0])
    print(yolo.resized_sz)
    print(label)

    for i in label:
        # label = COCO[int(i[-1])]
        i = list(map(int, i.tolist()))
        cv2.rectangle(img, i[0:2], i[2:4], (255, 0, 0), 2)

    # print(size)
    cv2.imshow("S", img)
    cv2.waitKey(0)
    # exit(0)
