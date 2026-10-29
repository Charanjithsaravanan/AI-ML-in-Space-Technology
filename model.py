"""A compact depthwise-separable CNN sized for a CubeSat payload.

Design notes
------------
* 64 -> 64 -> 32 -> 16 -> 8 -> 4 spatial, then global average pooling. GAP keeps
  the classifier head at 192x10 = 1,930 parameters and removes the flatten-to-
  dense layer that normally dominates small-image models.
* Depthwise 3x3 followed by 1x1 pointwise conv (MobileNet-style) cuts the
  multiply-accumulates by roughly 8x versus a plain 3x3 conv at equal width.
* width=1.0 gives 49,434 parameters = 0.19 MiB in fp32, and 6.60 M MACs per
  64x64 patch (13.2 MFLOP) - a comfortable fit for a rad-tolerant Cortex-class
  part or an FPGA soft core at a few frames per second.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _width(channels: int, multiplier: float) -> int:
    return max(8, int(round(channels * multiplier)))


class DepthwiseSeparableBlock(nn.Module):
    """3x3 depthwise -> BN -> ReLU6, then 1x1 pointwise -> BN -> ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        return self.act(self.bn2(self.pointwise(x)))


class TinySatCNN(nn.Module):
    """Input is normalised pixels; see NormalizedModel for the raw-[0,1] wrapper."""

    def __init__(self, num_classes: int = 10, width: float = 1.0, dropout: float = 0.1, in_ch: int = 3):
        super().__init__()
        c1, c2, c3, c4, c5 = (_width(c, width) for c in (24, 48, 96, 128, 192))

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True),
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(c1, c2, stride=2),
            DepthwiseSeparableBlock(c2, c3, stride=2),
            DepthwiseSeparableBlock(c3, c4, stride=2),
            DepthwiseSeparableBlock(c4, c5, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(c5, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self.stem(x))
        x = self.pool(x).flatten(1)
        return self.classifier(self.dropout(x))


class NormalizedModel(nn.Module):
    """Raw [0,1] patch -> logits.

    The mean/std buffers travel inside the traced graph, so benchmark.py and
    infer.py measure and serve exactly the same object, and a downstream user
    cannot forget the normalisation step.
    """

    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model((x - self.mean) / self.std)


def build_model(num_classes: int = 10, width: float = 1.0) -> TinySatCNN:
    return TinySatCNN(num_classes=num_classes, width=width)
