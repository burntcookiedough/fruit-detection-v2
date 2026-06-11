"""ConvNeXt-Pico backbone via timm — produces multi-scale features for detection."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    import timm
except ImportError as exc:
    raise ImportError("timm is required for the ConvNeXt backbone: pip install timm") from exc


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2.

    Adapted for ``(B, C, H, W)`` format used by timm's ConvMlp.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.linalg.vector_norm(x, ord=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class GRNWrapper(nn.Module):
    """Wraps a linear layer with GRN normalization."""

    def __init__(self, dim: int, fc2: nn.Module) -> None:
        super().__init__()
        self.grn = GRN(dim)
        self.fc2 = fc2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.grn(x))


def inject_grn_into_timm(model: nn.Module) -> None:
    """Inject GRN layers into timm ConvNeXt blocks to upgrade them to V2."""
    for module in model.modules():
        if module.__class__.__name__ == "Mlp" and hasattr(module, "fc2"):
            m: Any = module
            dim = m.fc1.out_features if hasattr(m.fc1, "out_features") else m.fc1.out_channels
            m.fc2 = GRNWrapper(dim, m.fc2)


class ConvNeXtBackbone(nn.Module):
    """Wraps a timm ConvNeXt model as a multi-scale feature extractor.

    Outputs feature maps at 3 scales (stride 8, 16, 32) for detection.
    Supports ImageNet pre-trained weights for transfer learning.

    Args:
        model_name: timm model name (default: ``'convnext_femto.d1_in1k'``)
        pretrained: whether to load ImageNet-1K pre-trained weights
        out_indices: which stages to extract features from (0-indexed)
        freeze_stem: if True, freezes stem + first stage
        use_grn: if True, injects GRN layers for ConvNeXt V2 upgrade
    """

    def __init__(
        self,
        model_name: str = "convnext_femto.d1_in1k",
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (1, 2, 3),
        freeze_stem: bool = False,
        use_grn: bool = True,
    ) -> None:
        super().__init__()
        self.model: Any = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=list(out_indices),
        )
        if use_grn:
            inject_grn_into_timm(self.model)
        feature_info: Any = self.model.feature_info
        self.out_channels: list[int] = feature_info.channels()
        self.out_strides: list[int] = feature_info.reduction()

        if freeze_stem:
            self._freeze_stem()

    def _freeze_stem(self) -> None:
        """Freeze the stem (patch embedding) and first stage."""
        for name, param in self.model.named_parameters():
            if "stem" in name or "stages.0" in name:
                param.requires_grad_(False)

    def freeze_all(self) -> None:
        """Freeze the complete backbone during transfer-learning warmup."""
        for param in self.model.parameters():
            param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (call after warmup epochs)."""
        for param in self.model.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return list of feature maps ``[P3, P4, P5]`` at strides ``[8, 16, 32]``."""
        return self.model(x)

    @property
    def num_output_features(self) -> int:
        return len(self.out_channels)
