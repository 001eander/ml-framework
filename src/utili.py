from contextlib import contextmanager
from typing import Any

from ignite.handlers import Timer


class MyTimer:
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

    def value(self, name: str) -> float:
        if name in self._dict:
            return self._dict[name].value()
        else:
            raise ValueError(f"Timer '{name}' has not been started.")

    def __getitem__(self, name: str) -> float:
        return self.value(name)

    def reset(self):
        self._dict = {}

    def items(self):
        return ((name, timer.value()) for name, timer in self._dict.items())

    def get_timer(self):
        @contextmanager
        def timeit(name: str):
            self.start(name)
            yield
            self.stop(name)

        return timeit


def make_conf(base: dict[str, Any], upd: str) -> dict[str, Any]:
    conf = {}
    for k, v in base.items():
        if not isinstance(v, dict):
            conf[k] = v
    if upd in base:
        for k, v in base[upd].items():
            conf[k] = v
    return conf


def make_table(columns: list[str], rows: list[list[Any]]) -> str:
    header = "| " + " | ".join(columns) + " |\n"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |\n"
    body = ""
    for row in rows:
        body += "| " + " | ".join([str(item) for item in row]) + " |\n"
    return header + separator + body
