"""FruitDetector v2 — Anchor-free detector with ConvNeXt backbone + PANet neck."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from .backbone import ConvNeXtBackbone
from .neck import PANet, Conv
from .anchor_points import generate_anchor_points_and_strides


class DFL(nn.Module):
    """Distribution Focal Loss decoding layer.

    Converts a discrete probability distribution over reg_max bins into
    a single continuous coordinate value via soft-argmax (expected value).

    Instead of predicting a box coordinate directly, the model predicts
    a probability distribution over `reg_max` discrete values [0, 1, ..., reg_max-1].
    The expected value of this distribution is the predicted coordinate.

    This allows the model to express uncertainty about box boundaries.

    Args:
        reg_max: number of discrete bins (default: 16)
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer('proj', torch.arange(reg_max, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode distribution predictions into coordinate values.

        Args:
            x: [B, N, 4 * reg_max] raw distribution logits

        Returns:
            [B, N, 4] decoded LTRB distances
        """
        B, N, _ = x.shape
        x = x.reshape(B, N, 4, self.reg_max)
        return F.softmax(x, dim=-1).matmul(self.proj.to(dtype=x.dtype))


class DecoupledHead(nn.Module):
    """Decoupled detection head — separate branches for classification and regression.

    Unlike v1's shared head, the cls and reg branches have independent conv stacks.
    This prevents conflicting gradients from classification and localization.

    Args:
        in_ch: input channels from the neck
        num_classes: number of detection classes
        reg_max: DFL bins for box regression (default: 16)
    """

    def __init__(self, in_ch: int, num_classes: int, reg_max: int = 16, num_convs: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max

        # Classification branch
        self.cls_conv = nn.Sequential(
            *[Conv(in_ch, in_ch, 3) for _ in range(num_convs)]
        )
        self.cls_pred = nn.Conv2d(in_ch, num_classes, 1)

        # Box regression branch
        self.reg_conv = nn.Sequential(
            *[Conv(in_ch, in_ch, 3) for _ in range(num_convs)]
        )
        self.reg_pred = nn.Conv2d(in_ch, 4 * reg_max, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize classification bias for focal loss stability."""
        # Bias init: -log((1 - prior) / prior) where prior = 0.01
        nn.init.constant_(self.cls_pred.bias, -4.595)
        nn.init.zeros_(self.reg_pred.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass for a single scale.

        Args:
            x: [B, C, H, W] feature map from neck

        Returns:
            cls: [B, H*W, num_classes] classification logits
            reg: [B, H*W, 4*reg_max] box distribution logits
        """
        B, _, H, W = x.shape

        cls = self.cls_conv(x)
        cls = self.cls_pred(cls)  # [B, num_classes, H, W]
        cls = cls.permute(0, 2, 3, 1).reshape(B, H * W, self.num_classes)

        reg = self.reg_conv(x)
        reg = self.reg_pred(reg)  # [B, 4*reg_max, H, W]
        reg = reg.permute(0, 2, 3, 1).reshape(B, H * W, 4 * self.reg_max)

        return cls, reg


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLO architectures."""
    def __init__(self, in_channels: int, out_channels: int, k: int = 5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(in_channels, c_, 1, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ * 4, out_channels, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CEM(nn.Module):
    """Context Enhancement Module to boost small object representation."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        c_hidden = max(1, in_channels // 4)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, c_hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_hidden, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class FruitDetectorV2(nn.Module):
    """Anchor-free fruit detector with ConvNeXt backbone + PANet neck + decoupled head.

    Architecture:
        ConvNeXt-Femto backbone (ImageNet pre-trained)
          → PANet bidirectional feature fusion
          → Decoupled classification + DFL regression heads

    Args:
        num_classes: number of fruit classes
        img_size: input image size (default: 416)
        backbone_name: timm model name (default: 'convnext_femto.d1_in1k')
        pretrained: whether to use ImageNet pre-trained backbone weights
        neck_channels: unified channel count in the PANet neck (default: 128)
        reg_max: DFL bins for box regression (default: 16)
        strides: detection strides (default: [8, 16, 32])
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 416,
        backbone_name: str = 'convnext_femto.d1_in1k',
        pretrained: bool = True,
        neck_channels: int = 128,
        reg_max: int = 16,
        strides: Optional[List[int]] = None,
        num_head_convs: int = 1,
        use_sppf: bool = True,
        use_cem: bool = True,
        use_grn: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.reg_max = reg_max
        self.strides = strides or [8, 16, 32]
        self.use_sppf = use_sppf
        self.use_cem = use_cem

        # Backbone: ConvNeXt-Femto → 3 feature maps at strides [8, 16, 32]
        self.backbone = ConvNeXtBackbone(
            model_name=backbone_name,
            pretrained=pretrained,
            out_indices=(1, 2, 3),
            use_grn=use_grn,
        )

        # SPPF on the highest-level feature map
        if self.use_sppf:
            self.sppf = SPPF(self.backbone.out_channels[-1], self.backbone.out_channels[-1], k=5)
        else:
            self.sppf = nn.Identity()

        # Neck: PANet bidirectional feature fusion
        self.neck = PANet(
            in_channels=self.backbone.out_channels,
            out_channels=neck_channels,
        )

        # CEM: Context Enhancement Modules after neck
        if self.use_cem:
            self.cem = nn.ModuleList([CEM(neck_channels) for _ in range(3)])
        else:
            self.cem = None

        # Detection heads: one per scale, shared architecture
        self.heads = nn.ModuleList([
            DecoupledHead(neck_channels, num_classes, reg_max, num_head_convs)
            for _ in self.strides
        ])

        # DFL decoder (shared across scales)
        self.dfl = DFL(reg_max)

        # Pre-compute anchor points (registered as buffer — saved with model)
        anchor_points, stride_tensor = generate_anchor_points_and_strides(
            img_size, self.strides
        )
        self.register_buffer('anchor_points', anchor_points)  # [N, 2]
        self.register_buffer('stride_tensor', stride_tensor)  # [N, 1]

    def forward(self, x: torch.Tensor) -> tuple:
        """Full forward pass.

        Args:
            x: [B, 3, img_size, img_size] input images

        Returns:
            cls_pred: [B, N, num_classes] classification logits
            box_pred_ltrb: [B, N, 4] decoded LTRB distances (for loss/inference)
            box_pred_raw: [B, N, 4*reg_max] raw distribution logits (for DFL loss)
            anchor_points: [N, 2] anchor centers
            stride_tensor: [N, 1] stride per point
        """
        # Backbone → multi-scale features
        features = self.backbone(x)  # [P3, P4, P5]
        
        # Apply SPPF to P5
        features = list(features)
        features[-1] = self.sppf(features[-1])

        # Neck → fused features
        fused = self.neck(features)  # [N3, N4, N5]
        
        # Apply CEM
        if self.use_cem:
            fused = [cem(f) for cem, f in zip(self.cem, fused)]

        # Heads → per-scale predictions
        cls_list = []
        reg_list = []
        for head, feat in zip(self.heads, fused):
            cls, reg = head(feat)
            cls_list.append(cls)
            reg_list.append(reg)

        # Concatenate across scales
        cls_pred = torch.cat(cls_list, dim=1)  # [B, N, num_classes]
        box_pred_raw = torch.cat(reg_list, dim=1)  # [B, N, 4*reg_max]

        # Decode DFL distributions into LTRB distances
        box_pred_ltrb = self.dfl(box_pred_raw)  # [B, N, 4]

        # Dynamically generate anchor points if image size changed (e.g. multi-scale training)
        if cls_pred.shape[1] != self.anchor_points.shape[0]:
            from .anchor_points import generate_anchor_points_and_strides
            anchor_points, stride_tensor = generate_anchor_points_and_strides(x.shape[-1], self.strides)
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
            if 'stem' in name or 'stages.0' in name or 'stages.1' in name:
                param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone weights."""
        self.backbone.unfreeze_all()
