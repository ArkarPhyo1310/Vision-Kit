import os
from typing import Optional, Tuple

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import (EVAL_DATALOADERS,
                                               TRAIN_DATALOADERS)
from torch.utils.data import DataLoader, SequentialSampler
from vision_kit.data.augmentations import TrainAugPipeline, ValAugPipeline
from vision_kit.data.datasets.yolo import YOLODataset
from vision_kit.data.sampling import InfiniteSampler, YoloBatchSampler
from vision_kit.utils.dataset_utils import collate_fn, worker_init_reset_seed

from .datasets.coco import COCODataset
from .mosiac_dataset import MosaicDataset


class LitDataModule(LightningDataModule):
    def __init__(
        self,
        data_cfg: dict,
        aug_cfg: dict = None,
        num_workers: int = 8,
        img_sz: Tuple[int, int] = (640, 640),
        seed: int = None
    ) -> None:
        super().__init__()

        self.img_sz: Tuple[int, int] = img_sz
        self.aug_cfg: dict = aug_cfg
        self.seed = seed
        self.num_workers = num_workers

        self.label_format = data_cfg.data_format
        self.batch_sz = data_cfg.batch_size

        self.train_path = os.path.join(data_cfg.data_dir, data_cfg.train_path)
        self.val_path = os.path.join(data_cfg.data_dir, data_cfg.val_path)
        if data_cfg.test_path:
            self.test_path = os.path.join(
                data_cfg.data_dir, data_cfg.test_path)
        self.filter_classes = data_cfg.filter_classes

    def get_dataset(self, data_path: str, aug_pipeline):
        if self.label_format == "coco":
            dataset_cls = COCODataset
        elif self.label_format == "yolo":
            dataset_cls = YOLODataset

        dataset = dataset_cls(
            data_path=data_path,
            filter_class=self.filter_classes,
            img_sz=self.img_sz,
            aug_pipeline=aug_pipeline
        )
        return dataset

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = self.get_dataset(
                data_path=self.train_path,
                aug_pipeline=TrainAugPipeline(
                    flip_lr_prob=self.aug_cfg["flip_lr_prob"],
                    flip_ud_prob=self.aug_cfg["flip_ud_prob"],
                    hsv_prob=self.aug_cfg["hsv_prob"],
                    img_sz=self.img_sz,
                    bbox_format=self.label_format
                )
            )

            self.train_dataset = MosaicDataset(
                self.train_dataset,
                mosaic=self.aug_cfg["enable_mosaic"],
                img_size=self.img_sz,
                degrees=self.aug_cfg["degrees"],
                translate=self.aug_cfg["translate"],
                mosaic_scale=self.aug_cfg["mosaic_scale"],
                mixup_scale=self.aug_cfg["mixup_scale"],
                shear=self.aug_cfg["shear"],
                enable_mixup=self.aug_cfg["enable_mixup"],
                mosaic_prob=self.aug_cfg["mosaic_prob"],
                mixup_prob=self.aug_cfg["mixup_prob"],
                label_format=self.label_format,
                aug_pipeline=TrainAugPipeline(
                    flip_lr_prob=self.aug_cfg["flip_lr_prob"],
                    flip_ud_prob=self.aug_cfg["flip_ud_prob"],
                    hsv_prob=self.aug_cfg["hsv_prob"],
                    img_sz=self.img_sz,
                    bbox_format=self.label_format
                )
            )

            self.val_dataset = self.get_dataset(
                data_path=self.val_path,
                aug_pipeline=ValAugPipeline(
                    img_sz=self.img_sz, bbox_format=self.label_format)
            )

        if stage == "test" or stage is None:
            if hasattr(self, "test_path"):
                self.test_dataset = self.get_dataset(
                    data_path=self.test_path,
                    aug_pipeline=ValAugPipeline(
                        img_sz=self.img_sz, bbox_format=self.label_format)
                )
            else:
                self.test_dataset = self.get_dataset(
                    data_path=self.val_path,
                    aug_pipeline=ValAugPipeline(
                        img_sz=self.img_sz, bbox_format=self.label_format)
                )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        sampler: InfiniteSampler = InfiniteSampler(
            len(self.train_dataset), seed=self.seed if self.seed else 13)
        batch_sampler: YoloBatchSampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=self.batch_sz,
            drop_last=False,
            mosaic=self.aug_cfg["enable_mosaic"]
        )

        train_dataloder: DataLoader = DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            pin_memory=True,
            batch_sampler=batch_sampler,
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
            drop_last=False,
            shuffle=False,
            collate_fn=collate_fn
        )

        return val_dataloader
