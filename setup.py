import re
import sys
from typing import Dict, List

import setuptools


def get_package_dir() -> Dict[str, str]:
    pkg_dir: Dict[str, str] = {
        "vision_kit.scripts": "scripts",
        "vision_kit.configs": "configs",
    }
    return pkg_dir


def get_requirements() -> List[str]:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        reqs: List[str] = [x.strip() for x in f.read().splitlines()]

    reqs: List[str] = [x for x in reqs if not x.startswith("#")]
    return reqs


def get_version() -> str:
    with open("./vision_kit/__init__.py", "r") as f:
        version: str = re.search(
            r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return version


def get_author() -> str:
    with open("./vision_kit/__init__.py", "r") as f:
        author: str = re.search(
            r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]',
            f.read(), re.MULTILINE
        ).group(1)
    return author


def get_long_descriptions() -> str:
    with open("README.md", "r", encoding="utf-8") as f:
        long_desc = f.read()
    return long_desc


setuptools.setup(
    name="vision_kit",
    version=get_version(),
    author=get_author(),
    url="https://github.com/ArkarPhyo1310/Vision-Kit",
    package_dir=get_package_dir(),
    packages=setuptools.find_packages(
        exclude=("tests", "tools")) + list(get_package_dir().keys()),
    python_requires=">=3.8",
    install_requires=get_requirements(),
    setup_requires=["wheel"],
    long_description=get_long_descriptions(),
    classifiers=[
        "Programming Language :: Python :: 3", "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    project_urls={
        "Source": "https://github.com/ArkarPhyo1310/Vision-Kit",
        "Tracker": "https://github.com/ArkarPhyo1310Vision-Kit/issues",
    },
)
