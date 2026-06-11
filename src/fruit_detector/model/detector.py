"""FruitDetector v2 — Anchor-free detector with ConvNeXt backbone + PANet neck."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from ..ops.anchor_points import generate_anchor_points_and_strides
from .backbone import ConvNeXtBackbone
from .components import CEM, DFL, SPPF
from .heads import DecoupledHead
from .neck import PANet

logger = logging.getLogger(__name__)


class FruitDetectorV2(nn.Module):
    """Anchor-free fruit detector with ConvNeXt backbone + PANet neck + decoupled head.

    Architecture::

        ConvNeXt-Femto backbone (ImageNet pre-trained)
          → PANet bidirectional feature fusion
          → Decoupled classification + DFL regression heads

    Args:
        num_classes: number of fruit classes
        img_size: input image size
        backbone_name: timm model name
        pretrained: whether to use ImageNet pre-trained backbone weights
        neck_channels: unified channel count in the PANet neck
        reg_max: DFL bins for box regression
        strides: detection strides
        num_head_convs: convolution layers per head branch
        use_sppf: whether to apply SPPF to the deepest feature map
        use_cem: whether to apply Context Enhancement Modules
        use_grn: whether to inject GRN into the backbone
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 416,
        backbone_name: str = "convnext_femto.d1_in1k",
        pretrained: bool = True,
        neck_channels: int = 128,
        reg_max: int = 16,
        strides: list[int] | None = None,
        num_head_convs: int = 2,
        use_sppf: bool = True,
        use_cem: bool = True,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.reg_max = reg_max
        self.strides = strides or [8, 16, 32]
        self.use_sppf = use_sppf
        self.use_cem = use_cem

        # Backbone: ConvNeXt → 3 feature maps at strides [8, 16, 32]
        self.backbone = ConvNeXtBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=(1, 2, 3),
            use_grn=use_grn,
        )

        # SPPF on the highest-level feature map
        self.sppf: nn.Module
        if self.use_sppf:
            self.sppf = SPPF(
                self.backbone.out_channels[-1],
                self.backbone.out_channels[-1],
                k=5,
            )
        else:
            self.sppf = nn.Identity()

        # Neck: PANet bidirectional feature fusion
        self.neck = PANet(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
        )

        # CEM: Context Enhancement Modules after neck
        self.cem: nn.ModuleList | None
        if self.use_cem:
            self.cem = nn.ModuleList([CEM(neck_channels) for _ in range(3)])
        else:
            self.cem = None

        # Detection heads: one per scale, shared architecture
        self.heads = nn.ModuleList(
            [
                DecoupledHead(neck_channels, num_classes, reg_max, num_head_convs)
                for _ in self.strides
            ]
        )

        # DFL decoder (shared across scales)
        self.dfl = DFL(reg_max)

        # Pre-compute anchor points (registered as buffer — saved with model)
        anchor_points, stride_tensor = generate_anchor_points_and_strides(img_size, self.strides)
        self.register_buffer("anchor_points", anchor_points)
        self.register_buffer("stride_tensor", stride_tensor)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Full forward pass.

        Args:
            x: ``[B, 3, img_size, img_size]`` input images

        Returns:
            cls_pred: ``[B, N, num_classes]`` classification logits
            box_pred_ltrb: ``[B, N, 4]`` decoded LTRB distances
            box_pred_raw: ``[B, N, 4*reg_max]`` raw distribution logits
            anchor_points: ``[N, 2]`` anchor centers
            stride_tensor: ``[N, 1]`` stride per point
        """
        # Backbone → multi-scale features
        features = self.backbone(x)

        # Apply SPPF to P5
        features = list(features)
        features[-1] = self.sppf(features[-1])

        # Neck → fused features
        fused = self.neck(features)

        # Apply CEM
        if self.use_cem and self.cem is not None:
            fused = [cem(f) for cem, f in zip(self.cem, fused)]

        # Heads → per-scale predictions
        cls_list: list[torch.Tensor] = []
        reg_list: list[torch.Tensor] = []
        for head, feat in zip(self.heads, fused):
            cls, reg = head(feat)
            cls_list.append(cls)
            reg_list.append(reg)

        # Concatenate across scales
        cls_pred = torch.cat(cls_list, dim=1)
        box_pred_raw = torch.cat(reg_list, dim=1)

        # Decode DFL distributions into LTRB distances
        box_pred_ltrb = self.dfl(box_pred_raw)

        # Expected registered buffers as tensors
        assert isinstance(self.anchor_points, torch.Tensor)
        assert isinstance(self.stride_tensor, torch.Tensor)

        # Dynamically generate anchor points if image size changed (multi-scale training)
        if cls_pred.shape[1] != self.anchor_points.shape[0]:
            anchor_points, stride_tensor = generate_anchor_points_and_strides(
                x.shape[-1], self.strides
            )
            anchor_points = anchor_points.to(x.device)
            stride_tensor = stride_tensor.to(x.device)
        else:
            anchor_points = self.anchor_points
            stride_tensor = self.stride_tensor

        return cls_pred, box_pred_ltrb, box_pred_raw, anchor_points, stride_tensor

    def freeze_backbone(self) -> None:
        """Freeze backbone weights (for transfer learning warmup)."""
        self.backbone.freeze_all()

    def freeze_early_backbone(self) -> None:
        """Freeze stem and stage 0, 1 for extreme speedup."""
        for name, param in self.backbone.named_parameters():
            if "stem" in name or "stages.0" in name or "stages.1" in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze stages 2 and 3, leaving stem and stage 0, 1 frozen."""
        for name, param in self.backbone.named_parameters():
            if not ("stem" in name or "stages.0" in name or "stages.1" in name):
                param.requires_grad = True

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:  # type: ignore[override]
        """Custom load_state_dict to handle anchor buffer shape changes gracefully."""
        for key in ["anchor_points", "stride_tensor"]:
            if key in state_dict and hasattr(self, key):
                model_shape = getattr(self, key).shape
                ckpt_shape = state_dict[key].shape
                if model_shape != ckpt_shape:
                    logger.info(
                        "Overriding buffer %s from checkpoint shape %s to model shape %s",
                        key,
                        ckpt_shape,
                        model_shape,
                    )
                    state_dict[key] = getattr(self, key).clone()
        return super().load_state_dict(state_dict, strict=strict)
