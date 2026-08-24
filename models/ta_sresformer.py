"""TA-SResformer model definition and timm registration."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from spikingjelly.activation_based import layer as sj_layer
from timm.layers import to_2tuple
from timm.models import register_model
from torch import nn

from .submodules.layers import BN, Conv1x1, Conv3x3, LIF, Linear, PLIF, SpikingMatmul


ActivationFactory = Callable[[], nn.Module]
StageLayout = Sequence[Sequence[str]]

_ATTENTION_BLOCK = "FRNSTSA"
_FEED_FORWARD_BLOCK = "SGCFFN"


class SGCFFN(nn.Module):
    """Spike-driven grouped-convolution feed-forward network."""

    def __init__(
        self,
        in_channels: int,
        num_conv: int = 1,
        ratio: int = 4,
        group_size: int = 64,
        activation: ActivationFactory = LIF,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * ratio

        # Attribute names are part of the released checkpoint schema.
        self.up = nn.Sequential(
            activation(),
            Conv1x1(in_channels, hidden_channels),
            BN(hidden_channels),
        )
        self.conv = nn.ModuleList(
            nn.Sequential(
                activation(),
                Conv3x3(
                    hidden_channels,
                    hidden_channels,
                    groups=hidden_channels // group_size,
                ),
                BN(hidden_channels),
            )
            for _ in range(num_conv)
        )
        self.down = nn.Sequential(
            activation(),
            Conv1x1(hidden_channels, in_channels),
            BN(in_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outer_residual = x.clone()
        hidden = self.up(x)
        inner_residual = hidden.clone()

        for convolution in self.conv:
            hidden = convolution(hidden)

        hidden = self.down(hidden + inner_residual)
        return hidden + outer_residual


class FRNSTSA(nn.Module):
    """Firing-rate-normalized spatiotemporal spiking attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        spatial_length: int,
        patch_size: int | tuple[int, int],
        activation: ActivationFactory = PLIF,
        chunk_size: int = 6,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divided by num_heads {num_heads}."
        )

        self.dim = dim
        self.num_heads = num_heads
        self.spatial_length = spatial_length
        self.chunk_size = chunk_size

        self.register_buffer(
            "firing_rate_x",
            torch.zeros(1, 1, num_heads, 1, 1),
        )
        self.init_firing_rate_x = False
        self.momentum = 0.999
        self.activation_in = activation()

        # W/Wproj retain their historic names for checkpoint compatibility.
        self.W = sj_layer.Conv2d(
            dim,
            dim * 2,
            patch_size,
            patch_size,
            bias=False,
            step_mode="m",
        )
        self.norm = BN(dim * 2)
        self.matmul1 = SpikingMatmul("r")
        self.matmul2 = SpikingMatmul("r")

        self.activation_out = activation()
        self.Wproj = Conv1x1(dim, dim)
        self.norm_proj = BN(dim)

    def _update_firing_rate(self, query: torch.Tensor) -> None:
        batch_rate = query.detach().mean((0, 1, 3, 4), keepdim=True)
        if not self.init_firing_rate_x and torch.all(self.firing_rate_x == 0):
            self.firing_rate_x = batch_rate
        self.init_firing_rate_x = True
        self.firing_rate_x = (
            self.firing_rate_x * self.momentum
            + batch_rate * (1.0 - self.momentum)
        )

    def _merge_time_chunks(self, tensor: torch.Tensor) -> torch.Tensor:
        time_steps, batch, heads, channels, tokens = tensor.shape
        chunks = time_steps // self.chunk_size
        return (
            tensor.view(
                chunks,
                self.chunk_size,
                batch,
                heads,
                channels,
                tokens,
            )
            .permute(0, 2, 3, 4, 1, 5)
            .reshape(
                chunks,
                batch,
                heads,
                channels,
                self.chunk_size * tokens,
            )
        )

    def _restore_time_axis(
        self,
        tensor: torch.Tensor,
        *,
        time_steps: int,
        spatial_tokens: int,
    ) -> torch.Tensor:
        chunks, batch, heads, head_dim, _ = tensor.shape
        return (
            tensor.view(
                chunks,
                batch,
                heads,
                head_dim,
                self.chunk_size,
                spatial_tokens,
            )
            .permute(0, 4, 1, 2, 3, 5)
            .reshape(time_steps, batch, heads, head_dim, spatial_tokens)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_steps, batch, channels, height, width = x.shape
        residual = x.clone()
        query_source = self.activation_in(x)

        key_value = self.norm(self.W(query_source))
        reduced_height, reduced_width = key_value.shape[-2:]
        reduced_tokens = reduced_height * reduced_width
        spatial_tokens = height * width
        head_dim = channels // self.num_heads

        key_value = key_value.reshape(
            time_steps,
            batch,
            self.num_heads,
            2 * head_dim,
            reduced_tokens,
        )
        key, value = key_value.split(head_dim, dim=3)
        query = query_source.reshape(
            time_steps,
            batch,
            self.num_heads,
            head_dim,
            spatial_tokens,
        )

        if self.training:
            self._update_firing_rate(query)

        if self.chunk_size > 1:
            assert time_steps % self.chunk_size == 0, (
                "T must be divisible by chunk_size"
            )
            attention_length = self.spatial_length * self.chunk_size
            query = self._merge_time_chunks(query)
            key = self._merge_time_chunks(key)
            value = self._merge_time_chunks(value)
        else:
            attention_length = self.spatial_length

        context = self.matmul1(value, key.transpose(-1, -2))
        context_scale = 1.0 / torch.sqrt(
            torch.tensor(
                attention_length,
                device=query_source.device,
                dtype=query_source.dtype,
            )
        )
        context = context * context_scale
        attended = self.matmul2(context, query)
        output_scale = 1.0 / (
            torch.sqrt(self.firing_rate_x * head_dim) + 1e-6
        )
        attended = attended * output_scale

        if self.chunk_size > 1:
            attended = self._restore_time_axis(
                attended,
                time_steps=time_steps,
                spatial_tokens=spatial_tokens,
            )

        attended = attended.reshape(
            time_steps,
            batch,
            channels,
            height,
            width,
        )
        projected = self.norm_proj(self.Wproj(self.activation_out(attended)))
        return projected + residual


class DownsampleLayer(nn.Module):
    """Activation-first spatial and channel downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        activation: ActivationFactory = LIF,
    ) -> None:
        super().__init__()
        self.conv = Conv3x3(in_channels, out_channels, stride=stride)
        self.norm = BN(out_channels)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(self.activation(x)))


class TASResformer(nn.Module):
    """Hierarchical event-based TA-SResformer classifier."""

    def __init__(
        self,
        layers: StageLayout,
        planes: Sequence[int],
        num_heads: Sequence[int],
        patch_sizes: Sequence[int],
        img_size: int | tuple[int, int] = 224,
        img_size_h: int | None = None,
        img_size_w: int | None = None,
        T: int = 4,
        in_channels: int = 4,
        num_classes: int = 1000,
        prologue: nn.Module | None = None,
        group_size: int = 64,
        activation: ActivationFactory = LIF,
        frn_stsa_chunk_size: int = 4,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs

        self.T = T
        chunk_size = self._compatible_chunk_size(T, frn_stsa_chunk_size)
        self.skip = ["prologue.0", "classifier"]

        assert len(planes) == len(layers) == len(num_heads) == len(patch_sizes)
        image_height, image_width = self._resolve_image_size(
            img_size,
            img_size_h,
            img_size_w,
        )

        if prologue is None:
            self.prologue = nn.Sequential(
                sj_layer.Conv2d(
                    in_channels,
                    planes[0],
                    7,
                    2,
                    3,
                    bias=False,
                    step_mode="m",
                ),
                BN(planes[0]),
                sj_layer.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    step_mode="m",
                ),
            )
            image_height //= 4
            image_width //= 4
        else:
            self.prologue = prologue

        self.layers = nn.Sequential()
        for stage_index, stage_layout in enumerate(layers):
            stage, image_height, image_width = self._build_stage(
                stage_index=stage_index,
                layout=stage_layout,
                planes=planes,
                num_heads=num_heads,
                patch_size=patch_sizes[stage_index],
                image_height=image_height,
                image_width=image_width,
                group_size=group_size,
                activation=activation,
                chunk_size=chunk_size,
            )
            self.layers.append(stage)

        self.avgpool = sj_layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.classifier = Linear(planes[-1], num_classes)
        self.init_weight()

    @staticmethod
    def _compatible_chunk_size(time_steps: int, requested: int) -> int:
        chunk_size = max(1, min(int(requested), int(time_steps)))
        while int(time_steps) % chunk_size != 0 and chunk_size > 1:
            chunk_size -= 1
        return chunk_size

    @staticmethod
    def _resolve_image_size(
        img_size: int | tuple[int, int],
        img_size_h: int | None,
        img_size_w: int | None,
    ) -> tuple[int, int]:
        if img_size_h is None and img_size_w is None:
            height, width = to_2tuple(img_size)
        else:
            height = img_size if img_size_h is None else img_size_h
            width = img_size if img_size_w is None else img_size_w
        return int(height), int(width)

    @staticmethod
    def _build_stage(
        *,
        stage_index: int,
        layout: Sequence[str],
        planes: Sequence[int],
        num_heads: Sequence[int],
        patch_size: int,
        image_height: int,
        image_width: int,
        group_size: int,
        activation: ActivationFactory,
        chunk_size: int,
    ) -> tuple[nn.Sequential, int, int]:
        stage = nn.Sequential()
        channels = planes[stage_index]

        if stage_index > 0:
            stage.append(
                DownsampleLayer(
                    planes[stage_index - 1],
                    channels,
                    stride=2,
                    activation=activation,
                )
            )
            image_height //= 2
            image_width //= 2

        patch_height, patch_width = to_2tuple(patch_size)
        attention_length = max(
            1,
            (image_height // patch_height) * (image_width // patch_width),
        )
        for block_name in layout:
            if block_name == _ATTENTION_BLOCK:
                stage.append(
                    FRNSTSA(
                        channels,
                        num_heads[stage_index],
                        attention_length,
                        patch_size,
                        activation=activation,
                        chunk_size=chunk_size,
                    )
                )
            elif block_name == _FEED_FORWARD_BLOCK:
                stage.append(
                    SGCFFN(
                        channels,
                        group_size=group_size,
                        activation=activation,
                    )
                )
            else:
                raise ValueError(block_name)

        return stage, image_height, image_width

    def init_weight(self) -> None:
        """Initialize trainable ANN layers using the released recipe."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def transfer(self, state_dict: Mapping[str, Any]):
        """Load a checkpoint while discarding its task-specific classifier."""
        transferable = {
            name: value
            for name, value in state_dict.items()
            if "classifier" not in name
        }
        return self.load_state_dict(transferable, strict=False)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 5:
            x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            assert x.dim() == 5
        else:
            # [B, T, C, H, W] -> [T, B, C, H, W]
            x = x.transpose(0, 1)

        encoded = self.layers(self.prologue(x))
        features = torch.flatten(self.avgpool(encoded), 2)
        logits = self.classifier(features)
        if return_features:
            return logits, features.mean(0)
        return logits

    def no_weight_decay(self) -> set[str]:
        return {
            f"{name}.w"
            for name, module in self.named_modules()
            if isinstance(module, PLIF)
        }


@register_model
def ta_sresformer(**kwargs: Any) -> TASResformer:
    """Build the public TA-SResformer architecture."""
    in_channels = kwargs.pop("in_channels", 2)
    return TASResformer(
        layers=(
            (_ATTENTION_BLOCK, _FEED_FORWARD_BLOCK),
            (_ATTENTION_BLOCK, _FEED_FORWARD_BLOCK) * 2,
            (_ATTENTION_BLOCK, _FEED_FORWARD_BLOCK) * 3,
        ),
        planes=(24, 72, 160),
        num_heads=(1, 3, 5),
        patch_sizes=(4, 2, 2),
        in_channels=in_channels,
        prologue=nn.Sequential(
            sj_layer.Conv2d(
                in_channels,
                24,
                3,
                1,
                1,
                bias=False,
                step_mode="m",
            ),
            BN(24),
        ),
        group_size=32,
        activation=PLIF,
        **kwargs,
    )


__all__ = ["FRNSTSA", "SGCFFN", "TASResformer", "ta_sresformer"]
