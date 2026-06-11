"""Task-Aligned Label Assignment (TAL) for anchor-free detection."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (both xyxy format).

    Args:
        boxes1: ``[N, 4]`` xyxy
        boxes2: ``[M, 4]`` xyxy

    Returns:
        ``[N, M]`` IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def _is_point_in_box(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Check if anchor points fall inside GT boxes.

    Args:
        points: ``[N, 2]`` ``(x, y)`` anchor centers
        boxes: ``[M, 4]`` ``(x1, y1, x2, y2)`` GT boxes

    Returns:
        ``[N, M]`` bool mask
    """
    px = points[:, None, 0]
    py = points[:, None, 1]
    x1 = boxes[None, :, 0]
    y1 = boxes[None, :, 1]
    x2 = boxes[None, :, 2]
    y2 = boxes[None, :, 3]
    return (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)


class TaskAlignedAssigner:
    """Task-Aligned Label Assignment for anchor-free object detection.

    Assigns ground truth boxes to anchor points based on a combination of
    classification score and IoU, rather than just geometric overlap.

    Args:
        topk: number of candidate anchors per GT box
        alpha: weight for classification score in alignment metric
        beta: weight for IoU in alignment metric
    """

    def __init__(
        self,
        topk: int = 10,
        alpha: float = 0.5,
        beta: float = 6.0,
    ) -> None:
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=False)
    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_labels_list: list[torch.Tensor],
        gt_bboxes_list: list[torch.Tensor],
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assign GT boxes to predictions for a batch.

        Args:
            pred_scores: ``[B, N, num_classes]`` predicted class scores (after sigmoid)
            pred_bboxes: ``[B, N, 4]`` predicted boxes in xyxy format
            anchor_points: ``[N, 2]`` anchor center points
            stride_tensor: ``[N, 1]`` stride per point
            gt_labels_list: list of B tensors, each ``[M_i]``
            gt_bboxes_list: list of B tensors, each ``[M_i, 4]`` in xyxy
            num_classes: total number of classes

        Returns:
            target_labels: ``[B, N]`` assigned class labels (0 = background)
            target_bboxes: ``[B, N, 4]`` assigned GT boxes
            target_scores: ``[B, N, num_classes]`` soft classification targets
            fg_mask: ``[B, N]`` bool, True for positive assignments
        """
        device = pred_scores.device
        B, N, C = pred_scores.shape

        pred_scores = pred_scores.float()
        pred_bboxes = pred_bboxes.float()

        target_labels = torch.zeros(B, N, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(B, N, 4, dtype=torch.float32, device=device)
        target_scores = torch.zeros(B, N, C, dtype=torch.float32, device=device)
        fg_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            gt_labels = gt_labels_list[b]
            gt_bboxes = gt_bboxes_list[b]

            M = gt_labels.shape[0]
            if M == 0:
                continue

            # Step 1: Center prior — only consider points inside GT boxes
            in_gt_mask = _is_point_in_box(anchor_points, gt_bboxes)
            candidate_mask = in_gt_mask.any(dim=1)
            if not candidate_mask.any():
                continue

            candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
            candidate_in_gt = in_gt_mask[candidate_idx]

            # Step 2: Compute alignment metric
            gt_cls_scores = pred_scores[b, candidate_idx][:, gt_labels]
            iou = _batch_iou(pred_bboxes[b, candidate_idx], gt_bboxes)
            iou = iou.clamp(min=0, max=1.0)

            alignment = (gt_cls_scores.clamp(min=0) ** self.alpha) * (iou**self.beta)
            alignment = alignment * candidate_in_gt.float()

            # Step 3: Select top-K anchors per GT
            topk = min(self.topk, candidate_idx.numel())
            topk_values, topk_indices = alignment.T.topk(topk, dim=1)

            is_topk = torch.zeros_like(alignment, dtype=torch.bool)
            valid_topk = topk_values > 0
            if valid_topk.any():
                gt_idx = torch.arange(M, device=device).unsqueeze(1).expand_as(topk_indices)
                is_topk[topk_indices[valid_topk], gt_idx[valid_topk]] = True

            fg = candidate_in_gt & is_topk

            # Step 4: Handle conflicts
            fg_any = fg.any(dim=1)
            if not fg_any.any():
                continue

            alignment_masked = alignment.clone()
            alignment_masked[~fg] = -1.0
            best_gt_idx = alignment_masked.argmax(dim=1)

            fg_anchor_idx = candidate_idx[fg_any]
            fg_mask[b, fg_anchor_idx] = True
            target_labels[b, fg_anchor_idx] = gt_labels[best_gt_idx[fg_any]] + 1
            target_bboxes[b, fg_anchor_idx] = gt_bboxes[best_gt_idx[fg_any]]

            # Soft classification targets
            fg_indices = fg_any.nonzero(as_tuple=True)[0]
            best_alignment = alignment_masked[fg_indices, best_gt_idx[fg_indices]]
            gt_max_align = alignment.max(dim=0).values.clamp(min=1e-7)
            norm_align = best_alignment / gt_max_align[best_gt_idx[fg_indices]]
            norm_align = norm_align.clamp(min=0, max=1.0)

            one_hot = F.one_hot(gt_labels[best_gt_idx[fg_indices]], num_classes=C).float()
            target_scores[b, fg_anchor_idx] = one_hot * norm_align.unsqueeze(1)

        return target_labels, target_bboxes, target_scores, fg_mask
