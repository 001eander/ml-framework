import argparse
import json
import os
import pathlib

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = (os.cpu_count() or 1) // 2


def main(config_path: pathlib.Path):
    name = config_path.stem
    config = json.load(open(config_path, "r"))
    hparams = config["hparams"]

    print(f"Using device: {device}")
    print(f"Number of workers: {num_workers}")
    print(f'Config "{name}": {config_path}')
    print(f"Hparams: {hparams}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main application with the specified config file.")
    parser.add_argument("config", type=str, help="Path to the config file")
    args = parser.parse_args()

    main(pathlib.Path(args.config))
