from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         RichProgressBar)
from pytorch_lightning.callbacks.progress.rich_progress import \
    RichProgressBarTheme
from pytorch_lightning.profiler import PyTorchProfiler, SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


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


def get_callbacks(dirpath: str, monitor: str = "map50", mode: str = "max"):
    ckpt_filename = "{epoch}-{" + f"{monitor}" + ":.2f}"
    model_checkpointing = ModelCheckpoint(
        dirpath=dirpath,
        filename=ckpt_filename,
        monitor=monitor,
        save_last=True,
        mode=mode
    )

    early_stopping = EarlyStopping(
        monitor="map50",
        patience=5,
        mode=mode
    )

    progress_bar = RichProgressBar(
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
        leave=True
    )

    return (model_checkpointing, early_stopping, progress_bar)


def get_loggers(dirpath):
    wandb_logger = WandbLogger(
        project="VisionKit",
        save_dir=dirpath
    )

    tb_logger = TensorBoardLogger(
        save_dir=dirpath,
        name="tb_logs",
        log_graph=True
    )

    return (wandb_logger, tb_logger)
