import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


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


class DenseMLP(MLP):
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
        super().__init__(
            in_shape,
            out_shape,
            layers,
            size,
            weight_init=weight_init,
            nonlinear=nonlinear,
            neg_k=neg_k,
        )

        self.layers[-1] = nn.Linear(layers * size, math.prod(out_shape))
        if nonlinear is not None and weight_init is not None:
            _init_linear(self.layers[-1], weight_init, nonlinear, neg_k)  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        xs = []
        for layer in self.layers[:-1]:
            x = layer(x)
            if isinstance(layer, nn.Linear):
                xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.layers[-1](x)
        return x.unflatten(1, self.out_shape)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int,
        size: int,
        pre_activated: bool = True,
        weight_init: Literal["xavier", "kaiming"] = "kaiming",
        nonlinear: Literal["prelu", "relu", "leaky_relu"] = "prelu",
        neg_k: float | None = 0.25,
        dim_match: Literal["pad", "linear"] = "linear",
        BN: bool = False,
    ) -> None:
        if (nonlinear is None or nonlinear == "relu") and neg_k is not None:
            raise ValueError("neg_k and n_param should be None when nonlinear is None or relu")
        elif nonlinear in ("leaky_relu", "prelu") and neg_k is None:
            raise ValueError("neg_k should be specified when nonlinear is leaky_relu or prelu")
        super().__init__()

        self.layers = layers
        self.pre_activated = pre_activated
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_match = dim_match

        self.linears = (
            [nn.Linear(in_dim, size)] + [nn.Linear(size, size) for _ in range(layers - 1)] + [nn.Linear(size, out_dim)]
        )
        for linear in self.linears:
            _init_linear(linear, weight_init, nonlinear, neg_k)
        self.linears = nn.ModuleList(self.linears)

        if dim_match == "linear" and in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
            _init_linear(self.shortcut, weight_init, nonlinear, neg_k)

        if nonlinear in ("relu", "leaky_relu"):
            assert neg_k is not None
            nonlinear_cls = {
                "relu": nn.ReLU,
                "leaky_relu": lambda: nn.LeakyReLU(negative_slope=neg_k),  # type: ignore
            }[nonlinear]
            self.nonliners = nn.ModuleList([nonlinear_cls() for _ in range(layers + 1)])
        else:
            assert neg_k is not None
            nonlinears = [nn.PReLU(num_parameters=size, init=neg_k) for _ in range(layers)]
            if pre_activated:
                nonlinears = [nn.PReLU(num_parameters=in_dim, init=neg_k)] + nonlinears
            else:
                nonlinears += [nn.PReLU(num_parameters=out_dim, init=neg_k)]
            self.nonliners = nn.ModuleList(nonlinears)

        self.BN = BN
        if BN:
            self.bns = nn.ModuleList([nn.BatchNorm1d(size) for _ in range(layers)])
            if pre_activated:
                self.bns = nn.ModuleList([nn.BatchNorm1d(in_dim)] + list(self.bns))
            else:
                self.bns = nn.ModuleList(list(self.bns) + [nn.BatchNorm1d(out_dim)])

    def _connect(self, identity: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            x += identity
            return x
        if self.dim_match == "linear":
            x += self.shortcut(identity)
            return x
        if self.in_dim < self.out_dim:
            shortcut_x = F.pad(identity, (0, x.shape[1] - identity.shape[1]))
            x += shortcut_x
        else:
            shortcut_x = identity[:, : x.shape[1]]
            x += shortcut_x
        return x

    def _pre_activated_forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.BN:
            x = self.bns[0](x)
        x = self.nonliners[0](x)
        for i in range(self.layers):
            x = self.linears[i](x)
            if self.BN:
                x = self.bns[i + 1](x)
            x = self.nonliners[i + 1](x)
        x = self.linears[-1](x)
        x = self._connect(identity, x)
        return x

    def _post_activated_forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for i in range(self.layers - 1):
            x = self.linears[i](x)
            if self.BN:
                x = self.bns[i](x)
            x = self.nonliners[i](x)
        x = self.linears[-1](x)
        if self.BN:
            x = self.bns[-1](x)
        x = self._connect(identity, x)
        x = self.nonliners[-1](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_activated:
            return self._pre_activated_forward(x)
        return self._post_activated_forward(x)


class ResMLP(nn.Module):
    def __init__(
        self,
        in_shape: list[int],
        out_shape: list[int],
        blocks: int,
        block_layers: int,
        block_size: int,
        pre_activated: bool = True,
        weight_init: Literal["xavier", "kaiming"] = "kaiming",
        nonlinear: Literal["prelu", "relu", "leaky_relu"] = "prelu",
        neg_k: float | None = 0.25,
        dim_match: Literal["linear", "pad"] = "linear",
        BN: bool = False,
    ) -> None:
        super().__init__()

        if (nonlinear is None or nonlinear == "relu") and neg_k is not None:
            raise ValueError("neg_k and n_param should be None when nonlinear is None or relu")
        elif nonlinear in ("leaky_relu", "prelu") and neg_k is None:
            raise ValueError("neg_k should be specified when nonlinear is leaky_relu or prelu")

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.net = nn.Sequential(
            *[
                ResBlock(
                    in_dim=math.prod(in_shape) if i == 0 else block_size,
                    out_dim=math.prod(out_shape) if i == blocks - 1 else block_size,
                    layers=block_layers,
                    pre_activated=pre_activated,
                    size=block_size,
                    weight_init=weight_init,
                    nonlinear=nonlinear,
                    neg_k=neg_k,
                    dim_match=dim_match,
                    BN=BN,
                )
                for i in range(blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        y = self.net(x)
        return y.unflatten(1, self.out_shape)
