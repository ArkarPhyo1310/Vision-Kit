import random
import uuid

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data.dataloader import default_collate


def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    return [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def remove_useless_info(dataset):
    """
    Remove useless info in coco dataset. COCO object is modified inplace.
    This function is mainly used for saving memory (save about 30% mem).
    """
    if isinstance(dataset, COCO):
        dataset = dataset.dataset
        dataset.pop("info", None)
        dataset.pop("licenses", None)
        for img in dataset["images"]:
            img.pop("license", None)
            img.pop("coco_url", None)
            img.pop("date_captured", None)
            img.pop("flickr_url", None)
        if "annotations" in dataset.dataset:
            for anno in dataset.dataset["annotations"]:
                anno.pop("segmentation", None)


def collate_fn(batch):
    im, label, path, shapes = zip(*batch)  # transposed
    padded_labels = []

    if isinstance(im[0], np.ndarray):
        im = np.stack(im, 0)
        im = torch.from_numpy(im).float()
    else:
        im = torch.stack(im, 0)

    for i, lb in enumerate(label):  # add target image index for build_targets()
        padded_label = np.zeros((lb.shape[0], 6)) if isinstance(
            lb, np.ndarray) else torch.zeros((lb.shape[0], 6))
        padded_label[:, 0] = i
        padded_label[:, 1] = lb[:, -1]
        padded_label[:, 2:] = lb[:, :4]
        padded_label = torch.from_numpy(padded_label) if isinstance(
            lb, np.ndarray) else padded_label
        padded_labels.append(padded_label)

    return im, padded_labels, path, shapes


def list_collate(batch):
    """
    Function that collates lists or tuples together into one list (of lists/tuples).
    Use this as the collate function in a Dataloader, if you want to have a list of
    items as an output, as opposed to tensors (eg. Brambox.boxes).
    """
    items = list(zip(*batch))

    for i in range(len(items)):
        if isinstance(items[i][0], (list, tuple)):
            items[i] = list(items[i])
        else:
            items[i] = default_collate(items[i])

    return items


def worker_init_reset_seed(worker_id):
    seed = uuid.uuid4().int % 2**32
    random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    np.random.seed(seed)
