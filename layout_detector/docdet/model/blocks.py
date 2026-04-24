"""
Reusable building blocks for the DocDet neck and head.

Contents
--------
ConvBNAct   : standard Conv2d + BatchNorm2d + SiLU triplet.
Bottleneck  : residual 1x1 -> 3x3 conv block used inside CSPBlock.
CSPBlock    : Cross-Stage Partial block - splits the input channels,
              runs one half through a stack of Bottlenecks, concatenates
              with the untouched half, and projects back.  ~50% FLOP
              savings vs a plain stack of 3x3 convs at comparable depth.
Scale       : learnable scalar multiplier used per FPN level on the
              regression output so each level learns its own scale.
"""

from typing import Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Conv + BN + activation triplet
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    """
    Conv2d -> BatchNorm2d -> activation.

    Default activation is SiLU (Swish); set ``act=None`` for a pure
    linear projection (used occasionally when the block feeds directly
    into another normalisation layer).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        act: Optional[nn.Module] = None,
    ):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act if act is not None else nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Bottleneck residual block used inside CSPBlock
# ---------------------------------------------------------------------------

class Bottleneck(nn.Module):
    """
    Two-conv bottleneck with optional residual connection.

    Structure: in -> 1x1 conv -> 3x3 conv -> (+ in if shortcut) -> out

    The hidden (bottleneck) channel count is ``hidden_channels`` which
    defaults to ``out_channels`` (no actual squeeze) but may be set
    smaller by CSPBlock for genuine bottlenecking.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        hidden_channels: Optional[int] = None,
    ):
        super().__init__()
        hidden_channels = hidden_channels or out_channels
        self.conv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBNAct(hidden_channels, out_channels, kernel_size=3)
        self.use_shortcut = shortcut and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.conv1(x))
        return x + y if self.use_shortcut else y


# ---------------------------------------------------------------------------
# CSP (Cross-Stage Partial) block
# ---------------------------------------------------------------------------

class CSPBlock(nn.Module):
    """
    Lightweight Cross-Stage Partial block.

    Splits the input projection into two halves of equal channel
    width.  One half passes through ``n`` Bottleneck residual blocks;
    the other is carried straight through.  The two halves are then
    concatenated and projected back to ``out_channels``.

    Parameters
    ----------
    in_channels   : input channel count
    out_channels  : output channel count (neck uses same value in/out)
    n             : number of stacked Bottleneck blocks on the branched half
    shortcut      : whether Bottleneck blocks use residual connections
    expansion     : fraction of ``out_channels`` used by each half
                    (default 0.5 so both halves are ``out_channels/2``)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
    ):
        super().__init__()
        hidden_channels = max(int(out_channels * expansion), 1)

        self.conv_split_a = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.conv_split_b = ConvBNAct(in_channels, hidden_channels, kernel_size=1)

        self.bottlenecks = nn.Sequential(
            *[
                Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut)
                for _ in range(n)
            ]
        )

        self.conv_merge = ConvBNAct(
            hidden_channels * 2, out_channels, kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_a = self.bottlenecks(self.conv_split_a(x))
        branch_b = self.conv_split_b(x)
        return self.conv_merge(torch.cat([branch_a, branch_b], dim=1))


# ---------------------------------------------------------------------------
# Scale: per-FPN-level learnable scalar for regression output
# ---------------------------------------------------------------------------

class Scale(nn.Module):
    """
    Multiply input by a learnable scalar.

    FCOS regressions are output in stride units (distances l/t/r/b
    from each grid cell to box edges), then multiplied by the per-
    level Scale and finally by the level stride to land in pixel
    space.  Giving each level its own learnable scalar lets the
    network pick up different regression magnitude ranges cleanly.

    Initialised to 1.0 so the model starts training as if there were
    no scaling at all.
    """

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale
