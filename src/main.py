import argparse
import json
import os
import pathlib
from typing import Any

import torch


def main(name: str, config: dict[str, Any]):
    print(f"CUDA is available? {torch.cuda.is_available()}")
    print(f"Number of cpu? {os.cpu_count()}")

    print(f'Config "{name}": {config}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System Info")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()

    conf_path = pathlib.Path(args.config)

    with open(conf_path, "r") as f:
        config = json.load(f)

    main(conf_path.stem, config)
