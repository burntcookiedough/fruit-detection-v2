"""ConvNeXt-Pico backbone via timm — produces multi-scale features for detection."""
import torch
import torch.nn as nn
from typing import Tuple

try:
    import timm
except ImportError:
    raise ImportError("timm is required for the ConvNeXt backbone: pip install timm")


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2.
    Adapted for (B, C, H, W) format used by timm's ConvMlp.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, C, H, W)
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class GRNWrapper(nn.Module):
    def __init__(self, dim: int, fc2: nn.Module):
        super().__init__()
        self.grn = GRN(dim)
        self.fc2 = fc2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.grn(x))


def inject_grn_into_timm(model: nn.Module) -> None:
    """Injects GRN layers into timm ConvNeXt blocks to upgrade them to V2."""
    for module in model.modules():
        if module.__class__.__name__ == 'Mlp' and hasattr(module, 'fc2'):
            dim = module.fc1.out_features if hasattr(module.fc1, 'out_features') else module.fc1.out_channels
            module.fc2 = GRNWrapper(dim, module.fc2)


class ConvNeXtBackbone(nn.Module):
    """Wraps a timm ConvNeXt model as a multi-scale feature extractor.

    Outputs feature maps at 3 scales (stride 8, 16, 32) for detection.
    Supports ImageNet pre-trained weights for transfer learning.

    Args:
        model_name: timm model name (default: 'convnext_pico')
        pretrained: whether to load ImageNet-1K pre-trained weights
        out_indices: which stages to extract features from (0-indexed)
        freeze_stem: if True, freezes stem + first stage (useful for transfer learning warmup)
    """

    def __init__(self, model_name: str = 'convnext_femto.d1_in1k', pretrained: bool = True, out_indices: Tuple[int, ...] = (1, 2, 3),
                 freeze_stem: bool = False, use_grn: bool = True):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=list(out_indices),
        )
        if use_grn:
            inject_grn_into_timm(self.model)
        self.out_channels = self.model.feature_info.channels()
        self.out_strides = self.model.feature_info.reduction()

        if freeze_stem:
            self._freeze_stem()

    def _freeze_stem(self) -> None:
        """Freeze the stem (patch embedding) and first stage."""
        for name, param in self.model.named_parameters():
            if 'stem' in name or 'stages.0' in name:
                param.requires_grad_(False)

    def freeze_all(self) -> None:
        """Freeze the complete backbone during transfer-learning warmup."""
        for param in self.model.parameters():
            param.requires_grad_(False)

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (call after warmup epochs)."""
        for param in self.model.parameters():
            param.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> list:
        """Returns list of feature maps [P3, P4, P5] at strides [8, 16, 32]."""
        return self.model(x)

    @property
    def num_output_features(self) -> int:
        return len(self.out_channels)
