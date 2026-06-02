"""ConvNeXt-Pico backbone via timm — produces multi-scale features for detection."""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required for the ConvNeXt backbone: pip install timm")


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

    def __init__(self, model_name='convnext_femto.d1_in1k', pretrained=True, out_indices=(1, 2, 3),
                 freeze_stem=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=list(out_indices),
        )
        self.out_channels = self.model.feature_info.channels()
        self.out_strides = self.model.feature_info.reduction()

        if freeze_stem:
            self._freeze_stem()

    def _freeze_stem(self):
        """Freeze the stem (patch embedding) and first stage."""
        for name, param in self.model.named_parameters():
            if 'stem' in name or 'stages.0' in name:
                param.requires_grad_(False)

    def freeze_all(self):
        """Freeze the complete backbone during transfer-learning warmup."""
        for param in self.model.parameters():
            param.requires_grad_(False)

    def unfreeze_all(self):
        """Unfreeze all parameters (call after warmup epochs)."""
        for param in self.model.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        """Returns list of feature maps [P3, P4, P5] at strides [8, 16, 32]."""
        return self.model(x)

    @property
    def num_output_features(self):
        return len(self.out_channels)
