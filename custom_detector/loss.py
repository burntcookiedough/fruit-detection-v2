"""Detection loss v2 — Task-Aligned Assignment + CIoU + Distribution Focal Loss."""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from .assigner import TaskAlignedAssigner
from .anchor_points import decode_boxes


def compute_class_weights(label_dir: str, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    """Compute inverse-frequency class weights from YOLO label directory.

    Returns a tensor of shape [num_classes] where rare classes get higher weight.
    Uses sqrt-inverse-frequency with Laplace smoothing for stability.
    """
    from collections import Counter
    counts = Counter()
    for fname in os.listdir(label_dir):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    counts[int(parts[0])] += 1
    total = sum(counts.values())
    if total == 0:
        return torch.ones(num_classes)
    freq = torch.tensor([counts.get(i, 0) + smoothing for i in range(num_classes)])
    weights = torch.sqrt(freq.sum() / (num_classes * freq))
    weights = weights / weights.mean()
    return weights


def compute_class_weights_from_label_entries(label_entries: list, num_classes: int,
                                             smoothing: float = 0.1) -> torch.Tensor:
    """Compute class weights from FruitDataset's pre-loaded label entries."""
    counts = torch.full((num_classes,), smoothing, dtype=torch.float32)
    for entry in label_entries:
        labels = entry.get('labels')
        if labels is None or len(labels) == 0:
            continue
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        counts += torch.bincount(labels_tensor, minlength=num_classes).float()[:num_classes]
    if counts.sum() <= num_classes * smoothing:
        return torch.ones(num_classes)
    weights = torch.sqrt(counts.sum() / (num_classes * counts))
    return weights / weights.mean()


@torch.amp.autocast('cuda', enabled=False)
def ciou_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    """Complete IoU loss between predicted and target boxes (both xyxy).

    CIoU = IoU - distance_penalty - aspect_ratio_penalty
    Returns per-box loss = 1 - CIoU, clamped to [0, 4].

    Args:
        pred_xyxy: [N, 4] predicted boxes
        target_xyxy: [N, 4] target boxes

    Returns:
        loss: [N] per-box CIoU loss
    """
    pred_xyxy = pred_xyxy.float()
    target_xyxy = target_xyxy.float()

    # Intersection
    inter_x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    inter_y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    inter_x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    inter_y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

    # Areas
    pred_area = (pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1])
    target_area = (target_xyxy[:, 2] - target_xyxy[:, 0]) * (target_xyxy[:, 3] - target_xyxy[:, 1])
    union = pred_area + target_area - inter + 1e-7
    iou = inter / union

    # Enclosing box diagonal
    enc_x1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    enc_y1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    enc_x2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    enc_y2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enc_diag_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2 + 1e-7

    # Center distance
    pred_cx = (pred_xyxy[:, 0] + pred_xyxy[:, 2]) / 2
    pred_cy = (pred_xyxy[:, 1] + pred_xyxy[:, 3]) / 2
    tgt_cx = (target_xyxy[:, 0] + target_xyxy[:, 2]) / 2
    tgt_cy = (target_xyxy[:, 1] + target_xyxy[:, 3]) / 2
    center_dist_sq = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2

    # Aspect ratio penalty
    pred_w = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=1e-6)
    pred_h = (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=1e-6)
    tgt_w = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=1e-6)
    tgt_h = (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=1e-6)
    v = (4.0 / (math.pi ** 2)) * (torch.atan(tgt_w / tgt_h) - torch.atan(pred_w / pred_h)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - center_dist_sq / enc_diag_sq - alpha * v
    return (1 - ciou).clamp(min=0, max=4.0)


def dfl_loss(pred_dist: torch.Tensor, target_ltrb: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Distribution Focal Loss for bounding box regression.

    Instead of predicting a single coordinate, the model predicts a probability
    distribution over `reg_max` discrete bins. DFL applies cross-entropy between
    the predicted distribution and a soft two-bin target derived from the true coordinate.

    Args:
        pred_dist: [N, 4 * reg_max] raw distribution logits for positive anchors
        target_ltrb: [N, 4] target LTRB distances (in stride units) for positive anchors
        reg_max: number of distribution bins

    Returns:
        loss: scalar DFL loss
    """
    N = pred_dist.shape[0]
    if N == 0:
        return pred_dist.sum() * 0.0

    # Reshape: [N, 4*reg_max] → [N*4, reg_max]
    pred_dist = pred_dist.reshape(N * 4, reg_max)
    target = target_ltrb.reshape(N * 4)  # [N*4]

    # Clamp targets to valid range [0, reg_max - 1]
    target = target.clamp(min=0, max=reg_max - 1 - 0.01)

    # Soft two-bin target: interpolate between floor and ceil bins
    tl = target.long()  # lower bin
    tr = tl + 1  # upper bin
    wl = tr.float() - target  # weight for lower bin
    wr = 1.0 - wl  # weight for upper bin

    # Cross-entropy for both bins, weighted
    loss = (F.cross_entropy(pred_dist, tl, reduction='none') * wl +
            F.cross_entropy(pred_dist, tr.clamp(max=reg_max - 1), reduction='none') * wr)

    return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for dense object detection to mitigate class imbalance.
    Works correctly with soft targets (e.g., from Task-Aligned Assignment).
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(inputs, targets)
        prob = torch.sigmoid(inputs)
        
        # focal weight for soft targets: |target - prob|^gamma
        focal_weight = torch.abs(targets - prob) ** self.gamma
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
            
        return focal_weight * bce_loss


class DetectionLossV2(nn.Module):
    """Detection loss with Task-Aligned Assignment + CIoU + DFL.

    Loss = cls_weight * L_cls + box_weight * L_ciou + dfl_weight * L_dfl

    Args:
        num_classes: number of detection classes
        reg_max: DFL bins (default: 16)
        cls_weight: classification loss weight (default: 1.0)
        box_weight: CIoU box loss weight (default: 2.5)
        dfl_weight: DFL loss weight (default: 0.5)
        tal_topk: TAL top-K candidates per GT (default: 10)
        class_weights: optional [C] tensor for per-class weighting
    """

    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        cls_weight: float = 1.0,
        box_weight: float = 2.5,
        dfl_weight: float = 0.5,
        tal_topk: int = 10,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.assigner = TaskAlignedAssigner(topk=tal_topk)
        self.bce = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        cls_pred: torch.Tensor,
        box_pred_ltrb: torch.Tensor,
        box_pred_raw: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_boxes_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            cls_pred: [B, N, C] classification logits
            box_pred_ltrb: [B, N, 4] decoded LTRB distances
            box_pred_raw: [B, N, 4*reg_max] raw DFL logits
            anchor_points: [N, 2] anchor centers
            stride_tensor: [N, 1] stride per point
            gt_boxes_list: list of [M_i, 4] GT boxes in cxcywh (YOLO format, pixel coords)
            gt_labels_list: list of [M_i] GT labels

        Returns:
            dict with keys: total, cls, box, dfl, num_pos
        """
        device = cls_pred.device
        B = cls_pred.shape[0]

        # Convert GT boxes from cxcywh to xyxy for the assigner
        gt_bboxes_xyxy = []
        for b in gt_boxes_list:
            if b.numel() > 0:
                x1 = b[:, 0] - b[:, 2] / 2
                y1 = b[:, 1] - b[:, 3] / 2
                x2 = b[:, 0] + b[:, 2] / 2
                y2 = b[:, 1] + b[:, 3] / 2
                gt_bboxes_xyxy.append(torch.stack([x1, y1, x2, y2], dim=1))
            else:
                gt_bboxes_xyxy.append(torch.zeros((0, 4), device=device))

        # Decode predicted boxes to xyxy for the assigner
        pred_bboxes_xyxy = decode_boxes(anchor_points, box_pred_ltrb, stride_tensor)  # [B, N, 4]

        # Run Task-Aligned Assignment
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner.assign(
            pred_scores=cls_pred.detach().sigmoid(),
            pred_bboxes=pred_bboxes_xyxy.detach(),
            anchor_points=anchor_points,
            stride_tensor=stride_tensor,
            gt_labels_list=gt_labels_list,
            gt_bboxes_list=gt_bboxes_xyxy,
            num_classes=self.num_classes,
        )

        num_pos = fg_mask.sum().item()

        # --- Classification loss (Focal Loss with soft targets from TAL) ---
        cls_targets = target_scores  # [B, N, C] soft targets
        cls_loss = self.bce(
            cls_pred, cls_targets
        )  # [B, N, C]

        # Apply per-class weights if provided
        if self.class_weights is not None:
            cls_loss = cls_loss * self.class_weights.to(device).unsqueeze(0).unsqueeze(0)

        cls_loss = cls_loss.sum() / max(num_pos, 1)

        if num_pos == 0:
            zero = torch.tensor(0.0, device=device)
            return {
                'total': cls_loss * self.cls_weight,
                'cls': cls_loss,
                'box': zero,
                'dfl': zero,
                'num_pos': 0,
            }

        # --- Box regression loss (CIoU) on foreground anchors ---
        fg_pred_xyxy = pred_bboxes_xyxy[fg_mask]  # [P, 4]
        fg_target_xyxy = target_bboxes[fg_mask]  # [P, 4]
        box_loss = ciou_loss(fg_pred_xyxy, fg_target_xyxy).mean()

        # --- DFL loss on foreground anchors ---
        # Compute target LTRB in stride units
        fg_anchor_pts = anchor_points.unsqueeze(0).expand(B, -1, -1)[fg_mask]  # [P, 2]
        fg_strides = stride_tensor.unsqueeze(0).expand(B, -1, -1)[fg_mask]  # [P, 1]

        # Target LTRB: distances from anchor point to box edges, in stride units
        target_left = (fg_anchor_pts[:, 0] - fg_target_xyxy[:, 0]) / fg_strides[:, 0]
        target_top = (fg_anchor_pts[:, 1] - fg_target_xyxy[:, 1]) / fg_strides[:, 0]
        target_right = (fg_target_xyxy[:, 2] - fg_anchor_pts[:, 0]) / fg_strides[:, 0]
        target_bottom = (fg_target_xyxy[:, 3] - fg_anchor_pts[:, 1]) / fg_strides[:, 0]
        target_ltrb = torch.stack([target_left, target_top, target_right, target_bottom], dim=1)

        fg_box_raw = box_pred_raw[fg_mask]  # [P, 4*reg_max]
        dfl_loss_val = dfl_loss(fg_box_raw, target_ltrb, self.reg_max)

        # --- Total loss ---
        total = (self.cls_weight * cls_loss +
                 self.box_weight * box_loss +
                 self.dfl_weight * dfl_loss_val)

        return {
            'total': total,
            'cls': cls_loss,
            'box': box_loss,
            'dfl': dfl_loss_val,
            'num_pos': num_pos,
        }
