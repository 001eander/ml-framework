from pathlib import Path
from typing import Literal

from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor


def make_dataset(
    name: Literal["MNIST", "CIFAR10"],
    size: int,
    image_size: int,
    dir: str | Path = "~/data",
    transform=None,
) -> tuple[Subset, int]:
    transform = transform or Compose([Resize(image_size), ToTensor(), Normalize((0.5,), (0.5 + 1e-5,))])
    ds_dict = {
        "MNIST": (MNIST, 1),
        "CIFAR10": (CIFAR10, 3),
    }
    ds, channels = ds_dict[name]
    dataset = Subset(ds(root=dir, download=True, transform=transform), range(size))
    return dataset, channels
