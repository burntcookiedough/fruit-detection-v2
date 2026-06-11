"""Decoupled detection heads for classification and box regression."""

from __future__ import annotations

import torch
import torch.nn as nn

from .neck import Conv


class DecoupledHead(nn.Module):
    """Decoupled detection head — separate branches for classification and regression.

    Unlike a shared head, the cls and reg branches have independent conv stacks.
    This prevents conflicting gradients from classification and localization.

    Args:
        in_ch: input channels from the neck
        num_classes: number of detection classes
        reg_max: DFL bins for box regression
        num_convs: number of convolution layers in each branch
    """

    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        reg_max: int = 16,
        num_convs: int = 1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        # Classification branch
        self.cls_conv = nn.Sequential(*[Conv(in_ch, in_ch, 3) for _ in range(num_convs)])
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        # Box regression branch
        self.reg_conv = nn.Sequential(*[Conv(in_ch, in_ch, 3) for _ in range(num_convs)])
        self.reg_pred = nn.Conv2d(in_ch, 4 * reg_max, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize classification bias for focal loss stability."""
        # Bias init: -log((1 - prior) / prior) where prior = 0.01
        if self.cls_pred.bias is not None:
            nn.init.constant_(self.cls_pred.bias, -4.595)
        if self.reg_pred.bias is not None:
            nn.init.zeros_(self.reg_pred.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single scale.

        Args:
            x: ``[B, C, H, W]`` feature map from neck

        Returns:
            cls: ``[B, H*W, num_classes]`` classification logits
            reg: ``[B, H*W, 4*reg_max]`` box distribution logits
        """
        B, _, H, W = x.shape

        cls = self.cls_conv(x)
        cls = self.cls_pred(cls)
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)

        reg = self.reg_conv(x)
        reg = self.reg_pred(reg)
        reg = reg.permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max)

        return cls, reg
