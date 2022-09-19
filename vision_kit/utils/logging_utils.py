import inspect
import logging
import os
import sys

from rich.logging import RichHandler
from rich.console import Console
from loguru import logger as Log


logger = Log
# logger = logging.getLogger("VisionKit")
console = Console()


def get_caller_name(depth=0):
    """
    Args:
        depth (int): Depth of caller conext, use 0 for caller depth.
        Default value: 0.
    Returns:
        str: module name of the caller
    """
    # the following logic is a little bit faster than inspect.stack() logic
    frame = inspect.currentframe().f_back
    for _ in range(depth):
        frame = frame.f_back

    return frame.f_globals["__name__"]


class StreamToLoguru:
    """
    stream object that redirects writes to a logger instance.
    """

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools", "pytorch_lightning")):
        """
        Args:
            level(str): log level string of loguru. Default value: "INFO".
            caller_names(tuple): caller names of redirected module.
                Default value: (apex, pycocotools).
        """
        self.level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buf):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buf.rstrip().splitlines():
                # use caller level log
                logger.opt(depth=1).log(self.level, line.rstrip())
        else:
            sys.__stdout__.write(buf)

        # for line in buf.rstrip().splitlines():
        #     # use caller level log
        #     logger.opt(depth=1).log(self.level, line.rstrip())

        # sys.__stdout__.write(buf)

    def flush(self):
        pass

    def isatty(self):
        return True


class StreamToLogger:

    def __init__(self, level="INFO"):
        self._level = level
        self.linebuf = ""

    def write(self, buffer):
        for line in buffer.rstrip().splitlines():
            logger.opt(depth=2).log(self._level, line.rstrip())

    def flush(self):
        pass


def redirect_sys_output(log_level="INFO"):
    redirect_logger = StreamToLoguru(log_level)
    sys.stdout = redirect_logger


# def setup_logger(path, filename="train.log"):
#     # the handler determines where the logs go: stdout/file
#     rich_handler = RichHandler(rich_tracebacks=True, console=console, )
#     file_handler = logging.FileHandler(
#         os.path.join(path, filename), encoding="utf-8")

#     logger.setLevel(logging.DEBUG)
#     rich_handler.setLevel(logging.DEBUG)
#     file_handler.setLevel(logging.DEBUG)

#     # the formatter determines what our logs will look like
#     fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
#     file_formatter = logging.Formatter(fmt_file)
#     file_handler.setFormatter(file_formatter)

#     logger.addHandler(rich_handler)
#     logger.addHandler(file_handler)

#     # redirect_sys_output()
#     sys.stdout = StreamToLoguru()


def setup_logger(path, filename="log.log", mode="a"):
    """setup logger for training and testing.
    Args:
        save_dir(str): location to save log file
        distributed_rank(int): device rank when multi-gpu environment
        filename (string): log save name.
        mode(str): log file write mode, `append` or `override`. default is `a`.
    Return:
        logger instance.
    """
    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    logger.remove()
    # logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])
    save_file = os.path.join(path, filename)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)
    # only keep logger in rank0 process
    logger.add(
        sys.stdout,
        format=loguru_format,
        level="DEBUG",
        enqueue=True,
    )
    # logger.add(
    #     RichHandler(),
    #     format="{message}",
    #     enqueue=True
    # )
    logger.add(save_file)

    stream = StreamToLogger()
    sys.stderr = stream
    sys.stdout = stream


if __name__ == "__main__":
    setup_logger('./')
    print("Print Message!")
    logger.info("Log Message")
    raise Exception("Raise Message!")
