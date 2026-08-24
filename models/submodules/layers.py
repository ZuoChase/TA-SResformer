"""Multi-step spiking layers used by TA-SResformer.

The small wrappers in this module keep backend and ``step_mode`` choices in one
place.  Their class names and nested module names are intentionally stable so
that checkpoints produced by earlier releases remain loadable.
"""

from typing import Literal

import numpy as np
import torch
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from torch import nn
from torch.nn.common_types import _size_2_t


# SpikingJelly 0.0.0.0.14 still accesses this alias on recent NumPy versions.
if not hasattr(np, "int"):
    np.int = np.int32


_MULTI_STEP = "m"
_CUPY_BACKEND = "cupy"


class IF(neuron.IFNode):
    """Multi-step integrate-and-fire neuron with an ATan surrogate."""

    def __init__(self) -> None:
        super().__init__(
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode=_MULTI_STEP,
            backend=_CUPY_BACKEND,
            store_v_seq=False,
        )


class LIF(neuron.LIFNode):
    """Multi-step leaky integrate-and-fire neuron."""

    def __init__(self) -> None:
        super().__init__(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode=_MULTI_STEP,
            backend=_CUPY_BACKEND,
            store_v_seq=False,
        )


class PLIF(neuron.ParametricLIFNode):
    """Multi-step parametric LIF neuron."""

    def __init__(self) -> None:
        super().__init__(
            init_tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
            step_mode=_MULTI_STEP,
            backend=_CUPY_BACKEND,
            store_v_seq=False,
        )


class BN(nn.Module):
    """Apply a shared 2-D batch normalization layer at every time step."""

    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                "expected x with shape [T, N, C, H, W], "
                f"but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, self.bn)


class SpikingMatmul(nn.Module):
    """Matrix multiplication marker used by the spiking attention blocks."""

    def __init__(self, spike: Literal["l", "r", "both"]) -> None:
        super().__init__()
        assert spike in {"l", "r", "both"}
        self.spike = spike

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.matmul(left, right)


class Conv3x3(layer.Conv2d):
    """Bias-free-by-default 3 x 3 convolution in multi-step mode."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            step_mode=_MULTI_STEP,
        )


class Conv1x1(layer.Conv2d):
    """Bias-free-by-default 1 x 1 convolution in multi-step mode."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: _size_2_t = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode="zeros",
            step_mode=_MULTI_STEP,
        )


class Linear(layer.Linear):
    """Linear projection in multi-step mode."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            step_mode=_MULTI_STEP,
        )


__all__ = ["BN", "Conv1x1", "Conv3x3", "IF", "LIF", "Linear", "PLIF", "SpikingMatmul"]
