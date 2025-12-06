import logging
import os
from pathlib import Path
from typing import Any

from ignite.handlers import Timer
from torch.utils.tensorboard.writer import SummaryWriter


class Timers:
    def __init__(self):
        self._dict: dict[str, Timer] = {}

    def start(self, name: str):
        if name in self._dict:
            self._dict[name].resume()
            return
        self._dict[name] = Timer(average=True)
        self._dict[name].resume()

    def stop(self, name: str):
        if name in self._dict:
            self._dict[name].pause()
            self._dict[name].step()
        else:
            raise ValueError(f"Timer '{name}' has not been started.")

    def reset(self):
        self._dict = {}

    def items(self):
        return ((name, timer.value()) for name, timer in self._dict.items())


def make_conf(base: dict[str, Any], upd: str) -> dict[str, Any]:
    conf = {}
    for k, v in base.items():
        if not isinstance(v, dict):
            conf[k] = v
    if upd in base:
        for k, v in base[upd].items():
            conf[k] = v
    return conf


def make_logger(name: str, logdir: Path) -> tuple[logging.Logger, SummaryWriter]:
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(logdir / "images", exist_ok=True)
    os.makedirs(logdir / "models", exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logdir / "train.log")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(message)s",
        datefmt="%y%m%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    writer = SummaryWriter(log_dir=logdir)
    return logger, writer
