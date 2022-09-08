import contextlib
import os
from typing import List, Tuple

from PIL import ExifTags

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
