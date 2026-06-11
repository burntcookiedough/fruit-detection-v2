"""PANet (Path Aggregation Network) — bidirectional feature fusion neck."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """Standard Conv2d + BatchNorm + SiLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding=kernel_size // 2, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck: two 3×3 convolutions with a skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.cv1 = Conv(channels, channels, 3)
        self.cv2 = Conv(channels, channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))


class CSPBlock(nn.Module):
    """Cross-Stage Partial block — splits channels, processes one half, merges.

    Inspired by YOLOv5's C3 block.

    Args:
        channels: input and output channel count (must be even)
        n_bottlenecks: number of sequential residual bottleneck layers
    """

    def __init__(self, channels: int, n_bottlenecks: int = 2) -> None:
        super().__init__()
        half = channels // 2
        self.cv1 = Conv(channels, half, 1)
        self.cv2 = Conv(channels, half, 1)
        self.bottlenecks = nn.Sequential(*[Bottleneck(half) for _ in range(n_bottlenecks)])
        self.cv3 = Conv(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.bottlenecks(self.cv1(x)), self.cv2(x)], dim=1))


class PANet(nn.Module):
    """Path Aggregation Network for multi-scale feature fusion.

    Performs top-down (FPN) then bottom-up (PAN) feature fusion across
    3 scales. Each fusion point uses a CSPBlock for rich feature mixing.

    Architecture::

        Input:  P3 (stride 8),  P4 (stride 16), P5 (stride 32)

        Top-Down (FPN):
          P5' = lateral(P5)
          P4' = CSP(lateral(P4) + upsample(P5'))
          P3' = CSP(lateral(P3) + upsample(P4'))

        Bottom-Up (PAN):
          N3 = CSP(P3')
          N4 = CSP(P4' + downsample(N3))
          N5 = CSP(P5' + downsample(N4))

        Output: N3 (stride 8), N4 (stride 16), N5 (stride 32)

    Args:
        in_channels: list of backbone output channel counts, e.g. ``[96, 192, 384]``
        out_channels: unified channel count for all neck outputs
        num_csp_blocks: number of bottleneck layers inside each CSPBlock
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int = 128,
        num_csp_blocks: int = 2,
    ) -> None:
        super().__init__()
        if len(in_channels) != 3:
            raise ValueError("PANet expects exactly 3 input scales")
        c3, c4, c5 = in_channels

        # Channel alignment (1×1 lateral convolutions)
        self.lateral_p3 = Conv(c3, out_channels, 1)
        self.lateral_p4 = Conv(c4, out_channels, 1)
        self.lateral_p5 = Conv(c5, out_channels, 1)

        # Top-Down path (FPN)
        self.td_csp_p4 = CSPBlock(out_channels, num_csp_blocks)
        self.td_csp_p3 = CSPBlock(out_channels, num_csp_blocks)

        # Bottom-Up path (PAN)
        self.bu_down_p3 = Conv(out_channels, out_channels, 3, stride=2)
        self.bu_csp_p4 = CSPBlock(out_channels, num_csp_blocks)
        self.bu_down_p4 = Conv(out_channels, out_channels, 3, stride=2)
        self.bu_csp_p5 = CSPBlock(out_channels, num_csp_blocks)

        # Final refinement for N3
        self.bu_csp_p3 = CSPBlock(out_channels, num_csp_blocks)

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            features: ``[P3, P4, P5]`` from backbone

        Returns:
            ``[N3, N4, N5]`` fused feature maps, all with ``out_channels`` channels
        """
        p3, p4, p5 = features

        # Channel alignment
        p3 = self.lateral_p3(p3)
        p4 = self.lateral_p4(p4)
        p5 = self.lateral_p5(p5)

        # Top-Down: P5 → P4 → P3
        p4 = self.td_csp_p4(p4 + F.interpolate(p5, size=p4.shape[2:], mode="nearest"))
        p3 = self.td_csp_p3(p3 + F.interpolate(p4, size=p3.shape[2:], mode="nearest"))

        # Bottom-Up: P3 → P4 → P5
        n3 = self.bu_csp_p3(p3)
        n4 = self.bu_csp_p4(p4 + self.bu_down_p3(n3))
        n5 = self.bu_csp_p5(p5 + self.bu_down_p4(n4))

        return [n3, n4, n5]
