import os
from datetime import timedelta
from typing import Any, Dict, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import (AdvancedProfiler, PyTorchProfiler,
                                        SimpleProfiler)
from pytorch_lightning.utilities.types import _PATH


class RichPbar(RichProgressBar):
    def __init__(self, refresh_rate: int = 1, leave: bool = False, theme: RichProgressBarTheme = ...,
                 console_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(refresh_rate, leave, theme, console_kwargs)

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> Dict[str, Union[int, str]]:
        # don't show the version number
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items


class ModelCkpt(ModelCheckpoint):
    def __init__(
            self, dirpath: Optional[_PATH] = None, filename: Optional[str] = None, monitor: Optional[str] = None,
            verbose: bool = False, save_last: Optional[bool] = None, save_top_k: int = 1, save_weights_only: bool = False,
            mode: str = "min", auto_insert_metric_name: bool = True, every_n_train_steps: Optional[int] = None,
            train_time_interval: Optional[timedelta] = None, every_n_epochs: Optional[int] = None,
            save_on_train_epoch_end: Optional[bool] = None):
        super().__init__(dirpath, filename, monitor, verbose, save_last, save_top_k, save_weights_only, mode,
                         auto_insert_metric_name, every_n_train_steps, train_time_interval, every_n_epochs, save_on_train_epoch_end)

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        score = round(self.best_model_score.detach().cpu().item(), 2)
        best_ckpt = torch.load(self.best_model_path, map_location="cpu")
        save_path = os.path.join(self.dirpath, f'best-map50_{score}.pt')
        torch.save(best_ckpt["model"], save_path)


def get_profilers(dirpath: str, filename: str, name: str = "simple"):
    if name == "advanced":
        profiler = AdvancedProfiler(
            dirpath=dirpath,
            filename=filename
        )
    elif name == "pytorch":
        # To find bottleneck/breakdowns
        profiler = PyTorchProfiler(
            dirpath=dirpath,
            filename=filename,
            emit_nvtx=True
        )
    elif name == "simple":
        # To find end-to-end overview clock time
        profiler = SimpleProfiler(
            dirpath=dirpath,
            filename=filename
        )

    return profiler


def get_callbacks(dirpath: str, monitor: str = "mAP@.5", mode: str = "max", bar_leave: bool = True):
    ckpt_filename = "{epoch}-{" + f"{monitor}" + ":.2f}"
    model_checkpointing = ModelCkpt(
        dirpath=f"{dirpath}/ckpts",
        filename=ckpt_filename,
        monitor=monitor,
        save_last=True,
        mode=mode
    )

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=5,
        mode=mode
    )

    progress_bar = RichPbar(
        theme=RichProgressBarTheme(
            description="green_yellow",
            progress_bar="green1",
            progress_bar_finished="green1",
            progress_bar_pulse="#6206E0",
            batch_progress="green_yellow",
            time="grey82",
            processing_speed="grey82",
            metrics="grey82",
        ),
        leave=bar_leave
    )

    return (model_checkpointing, early_stopping, progress_bar)


def get_loggers(dirpath):
    tb_logger = TensorBoardLogger(
        save_dir=dirpath,
        name="tb_logs"
    )

    wandb_logger = WandbLogger(
        project="VisionKit",
        save_dir=dirpath
    )

    return (tb_logger, wandb_logger)
