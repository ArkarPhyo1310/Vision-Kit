import logging
import os

from rich.logging import RichHandler
from rich.console import Console


logger = logging.getLogger("VisionKit")
console = Console()


def setup_logger(path, filename="train.log"):
    # the handler determines where the logs go: stdout/file
    rich_handler = RichHandler(rich_tracebacks=True, console=console)
    file_handler = logging.FileHandler(
        os.path.join(path, filename), encoding="utf-8")

    logger.setLevel(logging.DEBUG)
    rich_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # the formatter determines what our logs will look like
    fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'
    file_formatter = logging.Formatter(fmt_file)
    file_handler.setFormatter(file_formatter)

    logger.addHandler(rich_handler)
    logger.addHandler(file_handler)


if __name__ == "__main__":
    setup_logger('./')
    print("Print Message!")
    logger.info("Log Message")
    raise Exception("Raise Message!")
