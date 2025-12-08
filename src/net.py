import math
from typing import Any, Literal

from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_shape: list[int],
        out_shape: list[int],
        hidden_layers: list[int],
        weight_init: Literal["xavier", "kaiming"] | None = "kaiming",
        nonlinear: Literal["prelu", "relu", "leaky_relu"] | None = "prelu",
        neg_k: float | None = 0.25,
        final_nonlinear: Literal["tanh", "sigmoid", "softmax"] | None = None,
    ) -> None:
        super().__init__()
        if (nonlinear is None or nonlinear == "relu") and neg_k is not None:
            raise ValueError("neg_k and n_param should be None when nonlinear is None or relu")
        elif nonlinear in ("leaky_relu", "prelu") and neg_k is None:
            raise ValueError("neg_k should be specified when nonlinear is leaky_relu or prelu")

        def block(in_dim: int, out_dim: int, is_final: bool = False) -> list[Any]:
            layers: list[Any] = [nn.Linear(in_dim, out_dim)]
            # w init
            if weight_init == "xavier":
                gain = 1.0
                if nonlinear == "relu":
                    gain = nn.init.calculate_gain("relu")
                elif nonlinear in ("prelu", "leaky_relu"):
                    gain = nn.init.calculate_gain("leaky_relu", neg_k)
                nn.init.xavier_uniform_(layers[-1].weight, gain=gain)
            elif weight_init == "kaiming" and nonlinear is not None:
                if nonlinear == "relu":
                    nn.init.kaiming_uniform_(layers[-1].weight, nonlinearity="relu")
                elif nonlinear in ("leaky_relu", "prelu"):
                    assert neg_k is not None
                    nn.init.kaiming_uniform_(
                        layers[-1].weight,
                        a=neg_k,
                        nonlinearity="leaky_relu",
                    )
            # b init
            if weight_init is not None:
                nn.init.zeros_(layers[-1].bias)
            # nonlinear
            if is_final:
                if final_nonlinear == "tanh":
                    layers.append(nn.Tanh())
                elif final_nonlinear == "sigmoid":
                    layers.append(nn.Sigmoid())
                elif final_nonlinear == "softmax":
                    layers.append(nn.Softmax(dim=1))
                return layers
            if nonlinear == "relu":
                layers.append(nn.ReLU())
            if nonlinear == "leaky_relu":
                assert neg_k is not None
                layers.append(nn.LeakyReLU(negative_slope=neg_k))
            if nonlinear == "prelu":
                assert neg_k is not None
                layers.append(nn.PReLU(num_parameters=out_dim, init=neg_k))
            return layers

        layers = [nn.Flatten(), *block(math.prod(in_shape), hidden_layers[0])]
        for i in range(len(hidden_layers) - 1):
            layers.extend(block(hidden_layers[i], hidden_layers[i + 1]))
        layers.extend(block(hidden_layers[-1], math.prod(out_shape), is_final=True))
        layers.append(nn.Unflatten(1, out_shape))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
