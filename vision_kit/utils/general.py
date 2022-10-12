import contextlib
import os
from datetime import datetime
from typing import List, Tuple

from PIL import ExifTags


def update_loss_cfg(cfg):
    nl = 3
    cfg.hypermeters.box *= 3 / nl
    cfg.hypermeters.cls *= cfg.model.num_classes / 80 * 3 / nl  # scale to classes and layers
    cfg.hypermeters.obj *= (cfg.model.input_size[0] / 640) ** 2 * 3 / nl  # scale to image size and layers

    return cfg


# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break


def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:  # rotation 270 or 90
            s = (s[1], s[0])
    return s


def search_dir(path: str, set_type: str = "train") -> List[str]:
    dir_list: List[str] = [name for name in os.listdir(path) if os.path.isdir(
        os.path.join(path, name))]
    if len(dir_list) == 0:
        return dir_list
    return list(filter(lambda x: set_type in x, dir_list))


def mk_output_dir(path: str, model_name: str, task: str):
    ct_format = datetime.now().strftime('%Y%m%d%H%M')
    output_path = os.path.join(path, model_name, task, ct_format)
    os.makedirs(output_path, exist_ok=True)

    return output_path


def dw_multiple_generator(version: str = "s") -> Tuple[float, float]:
    width, depth = 0.25, 0.33
    if version.lower() == "s":
        depth *= 1.01
        width *= 2
    elif version.lower() == "m":
        depth *= 2.02
        width *= 3
    elif version.lower() == "l":
        depth *= 3.03
        width *= 4
    elif version.lower() == "x":
        depth *= 4.04
        width *= 5
    elif version.lower() == "n":
        depth *= 1
        width *= 1
    else:
        raise Exception(f"{version.lower} is not supported!")

    return width, round(depth, 2)
