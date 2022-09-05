import inspect
import os
import sys

from loguru import logger


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


class StreamToLogger:

    def __init__(self, level="INFO", caller_names=("apex", "pycocotools")):
        self._level = level
        self.linebuf = ""
        self.caller_names = caller_names

    def write(self, buffer):
        full_name = get_caller_name(depth=1)
        module_name = full_name.rsplit(".", maxsplit=-1)[0]
        if module_name in self.caller_names:
            for line in buffer.rstrip().splitlines():
                logger.opt(depth=2).log(self._level, line.rstrip(), )
        else:
            sys.__stdout__.write(buffer)

    def flush(self):
        pass


def setup_logger(save_dir, file_name: str = "log.log", mode: str = "a"):
    """Setup logging for debugging

    Args:
        save_dir ([type]): directory to save the log file
        file_name (str, optional): log file name. Defaults to "exe.log".
        mode (str, optional): log file write mode, "append" or "overwrite". Defaults to "a".
    """
    logger.remove()

    loguru_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    save_file = os.path.join(save_dir, file_name)
    if mode == "o" and os.path.exists(save_file):
        os.remove(save_file)

    logger.add(sys.stdout, format=loguru_format, level="INFO", enqueue=True)
    logger.add(save_file)

    sys.stderr = StreamToLogger("ERROR")
    sys.stdout = StreamToLogger("INFO")


if __name__ == "__main__":
    setup_logger('./')
    print("Print Message!")
    logger.info("Log Message")
    raise Exception("Raise Message!")
