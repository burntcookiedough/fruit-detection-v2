"""Reusable detector components: SPPF, CEM, DFL."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFL(nn.Module):
    """Distribution Focal Loss decoding layer.

    Converts a discrete probability distribution over ``reg_max`` bins into
    a single continuous coordinate value via soft-argmax (expected value).

    Args:
        reg_max: number of discrete bins
    """

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode distribution predictions into coordinate values.

        Args:
            x: ``[B, N, 4 * reg_max]`` raw distribution logits

        Returns:
            ``[B, N, 4]`` decoded LTRB distances
        """
        B, N, _ = x.shape
        x = x.reshape(B, N, 4, self.reg_max)
        assert isinstance(self.proj, torch.Tensor)
        return F.softmax(x, dim=-1).matmul(self.proj.to(dtype=x.dtype))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLO architectures."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 5) -> None:
        super().__init__()
        c_ = in_channels // 2
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True),
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CEM(nn.Module):
    """Context Enhancement Module to boost small object representation."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        c_hidden = max(1, in_channels // 4)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, c_hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_hidden, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y
