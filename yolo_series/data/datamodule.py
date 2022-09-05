from typing import Optional, Tuple

from loguru import logger
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, SequentialSampler
from yolo_series.data.augmentations import TrainAugPipeline, ValAugPipeline
from yolo_series.data.sampling import InfiniteSampler, YoloBatchSampler
from yolo_series.utils.dataset_utils import collate_fn, worker_init_reset_seed
from yolo_series.utils.general import search_dir

from .datasets.coco import COCODataset
from .mosiac_dataset import MosaicDataset


class COCODataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        train_json: str = "instances_train2017.json",
        val_json: str = "instances_val2017.json",
        test_json: str = None,
        num_workers: int = 0,
        batch_sz: int = 16,
        img_sz: Tuple[int, int] = (640, 640),
        aug_config: dict = None,
        seed: int = None,
        cache: bool = True,

    ) -> None:
        super().__init__()

        self.data_dir: str = data_dir
        self.train_json: str = train_json
        self.val_json: str = val_json
        self.test_json: str = test_json
        self.img_sz: Tuple[int, int] = img_sz
        self.cache: bool = cache
        self.aug_config: dict = aug_config
        self.batch_sz: int = batch_sz
        self.seed = seed
        self.num_workers = num_workers

        train_folder = search_dir(data_dir, "train")
        val_folder = search_dir(data_dir, "val")
        test_folder = search_dir(data_dir, "test")

        if train_folder:
            self.train_name = train_folder[0]
            self.val_name = train_folder[0] if not val_folder else val_folder[0]
        else:
            raise Exception("Training or Validation data are not found!")

        self.enable_test: bool = True

        if self.test_json is None:
            self.enable_test: bool = False

        if self.enable_test:
            if test_folder:
                self.test_name = test_folder[0]
            else:
                logger.warning(
                    "Testing data not found! Validation data will be used.")

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.train_json,
                name=self.train_name,
                img_sz=self.img_sz,
                aug_pipeline=TrainAugPipeline(
                    flip_lr_prob=self.aug_config["flip_lr_prob"],
                    flip_ud_prob=self.aug_config["flip_ud_prob"],
                    hsv_prob=self.aug_config["hsv_prob"],
                    img_sz=self.img_sz
                ),
                cache=self.cache,
            )
            self.train_dataset = MosaicDataset(
                self.train_dataset,
                mosaic=self.aug_config["enable_mosaic"],
                img_size=self.img_sz,
                degrees=self.aug_config["degrees"],
                translate=self.aug_config["translate"],
                mosaic_scale=self.aug_config["mosaic_scale"],
                mixup_scale=self.aug_config["mixup_scale"],
                shear=self.aug_config["shear"],
                enable_mixup=self.aug_config["enable_mixup"],
                mosaic_prob=self.aug_config["mosaic_prob"],
                mixup_prob=self.aug_config["mixup_prob"],
                aug_pipeline=TrainAugPipeline(
                    flip_lr_prob=self.aug_config["flip_lr_prob"],
                    flip_ud_prob=self.aug_config["flip_ud_prob"],
                    hsv_prob=self.aug_config["hsv_prob"],
                    img_sz=self.img_sz
                )
            )

            self.val_dataset = COCODataset(
                data_dir=self.data_dir,
                json_file=self.val_json,
                name=self.val_name,
                img_sz=self.img_sz,
                aug_pipeline=ValAugPipeline(img_sz=self.img_sz)
            )

        if stage == "test" or stage is None:
            if self.enable_test:
                self.test_dataset = COCODataset(
                    data_dir=self.data_dir,
                    json_file=self.test_json,
                    name=self.test_name,
                    img_sz=self.img_sz,
                    aug_pipeline=ValAugPipeline(img_sz=self.img_sz)
                )
            else:
                self.test_dataset = COCODataset(
                    data_dir=self.data_dir,
                    json_file=self.val_json,
                    name=self.val_name,
                    img_sz=self.img_sz,
                    aug_pipeline=ValAugPipeline(img_sz=self.img_sz)
                )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler: InfiniteSampler = InfiniteSampler(
            len(self.train_dataset), seed=self.seed if self.seed else 13)
        batch_sampler: YoloBatchSampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=self.batch_sz,
            drop_last=False,
            mosaic=self.aug_config["enable_mosaic"]
        )

        train_dataloder: DataLoader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
            worker_init_fn=worker_init_reset_seed,
            collate_fn=collate_fn
        )

        return train_dataloder

    def test_dataloader(self) -> EVAL_DATALOADERS:
        sampler: SequentialSampler = SequentialSampler(self.test_dataset)

        test_dataloader: DataLoader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_sz,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

        return test_dataloader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        sampler: SequentialSampler = SequentialSampler(self.test_dataset)

        val_dataloader: DataLoader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_sz,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )

        return val_dataloader
