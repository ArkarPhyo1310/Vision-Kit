import random
import uuid

import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data.dataloader import default_collate


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

    return im, torch.cat(padded_labels, 0), path, shapes


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
