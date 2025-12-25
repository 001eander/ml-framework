import math
from typing import Literal

import torch
from torch import nn


def _init_linear(
    linear: nn.Linear,
    weight_init: Literal["xavier", "kaiming"] = "kaiming",
    nonlinear: Literal["prelu", "relu", "leaky_relu"] = "prelu",
    neg_k: float | None = 0.25,
) -> None:
    if nonlinear in ("leaky_relu", "prelu") and neg_k is None:
        raise ValueError("neg_k should be specified when nonlinear is leaky_relu or prelu")

    if nonlinear == "relu":
        neg_k = None

    if weight_init == "xavier":
        gain = 1.0
        if nonlinear == "relu":
            gain = nn.init.calculate_gain("relu")
        else:
            gain = nn.init.calculate_gain("leaky_relu", neg_k)
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    else:
        if nonlinear == "relu":
            nn.init.kaiming_uniform_(linear.weight, nonlinearity="relu")
        else:
            assert neg_k is not None
            nn.init.kaiming_uniform_(linear.weight, a=neg_k, nonlinearity="leaky_relu")
    nn.init.zeros_(linear.bias)


class MLP(nn.Module):
    def __init__(
        self,
        in_shape: list[int],
        out_shape: list[int],
        layers: int,
        size: int,
        weight_init: Literal["xavier", "kaiming"] | None = "kaiming",
        nonlinear: Literal["prelu", "relu", "leaky_relu"] | None = "prelu",
        neg_k: float | None = 0.25,
    ) -> None:
        super().__init__()

        if (nonlinear is None or nonlinear == "relu") and neg_k is not None:
            raise ValueError("neg_k and n_param should be None when nonlinear is None or relu")
        elif nonlinear in ("leaky_relu", "prelu") and neg_k is None:
            raise ValueError("neg_k should be specified when nonlinear is leaky_relu or prelu")

        def block(in_dim: int, out_dim: int) -> list[nn.Module]:
            linear = nn.Linear(in_dim, out_dim)
            if weight_init is not None and nonlinear is not None:
                _init_linear(linear, weight_init, nonlinear, neg_k)
            nonlinear_layer = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(negative_slope=neg_k),  # type: ignore
                "prelu": nn.PReLU(num_parameters=out_dim, init=neg_k),  # type: ignore
            }[nonlinear]
            return [linear, nonlinear_layer]

        self.out_shape = out_shape

        self.layers = block(math.prod(in_shape), size)
        for _ in range(layers - 1):
            self.layers.extend(block(size, size))
        self.layers.append(nn.Linear(size, math.prod(out_shape)))
        if weight_init is not None and nonlinear is not None:
            _init_linear(self.layers[-1], weight_init, nonlinear, neg_k)  # type: ignore
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return x.unflatten(1, self.out_shape)
