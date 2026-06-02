"""Task-Aligned Label Assignment (TAL) for anchor-free detection."""
import torch
import torch.nn.functional as F


def _batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes (both xyxy format).

    Args:
        boxes1: [N, 4] xyxy
        boxes2: [M, 4] xyxy

    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # [N]
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # [M]

    inter_x1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])  # [N, M]
    inter_y1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


def _is_point_in_box(points: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Check if anchor points fall inside GT boxes.

    Args:
        points: [N, 2] (x, y) anchor centers
        boxes: [M, 4] (x1, y1, x2, y2) GT boxes

    Returns:
        mask: [N, M] bool — True if point i is inside box j
    """
    # points[:, None, :] → [N, 1, 2],  boxes[None, :, :] → [1, M, 4]
    px = points[:, None, 0]  # [N, 1]
    py = points[:, None, 1]  # [N, 1]
    x1 = boxes[None, :, 0]  # [1, M]
    y1 = boxes[None, :, 1]
    x2 = boxes[None, :, 2]
    y2 = boxes[None, :, 3]

    return (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)  # [N, M]


class TaskAlignedAssigner:
    """Task-Aligned Label Assignment for anchor-free object detection.

    Assigns ground truth boxes to anchor points based on a combination of
    classification score and IoU, rather than just geometric overlap.
    This creates a self-reinforcing training loop — the model assigns labels
    to predictions it is already good at, which accelerates learning.

    Args:
        topk: number of candidate anchors per GT box (default: 10)
        alpha: weight for classification score in alignment metric (default: 0.5)
        beta: weight for IoU in alignment metric (default: 6.0)
    """

    def __init__(self, topk: int = 10, alpha: float = 0.5, beta: float = 6.0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    @torch.amp.autocast('cuda', enabled=False)
    def assign(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        stride_tensor: torch.Tensor,
        gt_labels_list: list,
        gt_bboxes_list: list,
        num_classes: int,
    ) -> tuple:
        """Assign GT boxes to predictions for a batch.

        Args:
            pred_scores: [B, N, num_classes] predicted class scores (after sigmoid)
            pred_bboxes: [B, N, 4] predicted boxes in xyxy format
            anchor_points: [N, 2] anchor center points
            stride_tensor: [N, 1] stride per point
            gt_labels_list: list of B tensors, each [M_i] with class indices
            gt_bboxes_list: list of B tensors, each [M_i, 4] in xyxy format
            num_classes: int

        Returns:
            target_labels: [B, N] assigned class labels (0 = background)
            target_bboxes: [B, N, 4] assigned GT boxes (zeros for negatives)
            target_scores: [B, N, num_classes] soft classification targets
            fg_mask: [B, N] bool, True for positive (foreground) assignments
        """
        device = pred_scores.device
        B, N, C = pred_scores.shape

        # Ensure float32 for numerical stability
        pred_scores = pred_scores.float()
        pred_bboxes = pred_bboxes.float()

        target_labels = torch.zeros(B, N, dtype=torch.long, device=device)
        target_bboxes = torch.zeros(B, N, 4, dtype=torch.float32, device=device)
        target_scores = torch.zeros(B, N, C, dtype=torch.float32, device=device)
        fg_mask = torch.zeros(B, N, dtype=torch.bool, device=device)

        for b in range(B):
            gt_labels = gt_labels_list[b]  # [M]
            gt_bboxes = gt_bboxes_list[b]  # [M, 4] xyxy

            M = gt_labels.shape[0]
            if M == 0:
                # No GT boxes — all anchors are background
                continue

            # Step 1: Center prior — only consider points inside GT boxes
            in_gt_mask = _is_point_in_box(anchor_points, gt_bboxes)  # [N, M]
            candidate_mask = in_gt_mask.any(dim=1)  # [N]
            if not candidate_mask.any():
                continue

            candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
            candidate_in_gt = in_gt_mask[candidate_idx]  # [K, M]

            # Step 2: Compute alignment metric for all (anchor, GT) pairs
            # Get predicted score for each GT's class
            gt_cls_scores = pred_scores[b, candidate_idx][:, gt_labels]  # [K, M]
            # Compute IoU between predicted boxes and GT boxes
            iou = _batch_iou(pred_bboxes[b, candidate_idx], gt_bboxes)  # [K, M]
            iou = iou.clamp(min=0, max=1.0)

            # alignment = score^alpha * iou^beta
            alignment = (gt_cls_scores.clamp(min=0) ** self.alpha) * (iou ** self.beta)  # [K, M]

            # Only keep alignments for points inside GT boxes
            alignment = alignment * candidate_in_gt.float()

            # Step 3: Select top-K anchors per GT by alignment score
            topk = min(self.topk, candidate_idx.numel())
            # topk_values: [M, topk], topk_indices: [M, topk]
            topk_values, topk_indices = alignment.T.topk(topk, dim=1)  # transpose to [M, K] first

            # Build is_topk mask [K, M]
            is_topk = torch.zeros_like(alignment, dtype=torch.bool)
            valid_topk = topk_values > 0
            if valid_topk.any():
                gt_idx = torch.arange(M, device=device).unsqueeze(1).expand_as(topk_indices)
                is_topk[topk_indices[valid_topk], gt_idx[valid_topk]] = True

            # Final foreground mask: must be inside GT AND in top-K
            fg = candidate_in_gt & is_topk  # [K, M]

            # Step 4: Handle conflicts — if one anchor is assigned to multiple GTs,
            # pick the GT with the highest alignment metric
            fg_any = fg.any(dim=1)  # [K] — is this candidate assigned to any GT?
            if not fg_any.any():
                continue

            # For each anchor, find the GT with highest alignment
            alignment_masked = alignment.clone()
            alignment_masked[~fg] = -1.0  # mask out non-assigned pairs
            best_gt_idx = alignment_masked.argmax(dim=1)  # [K]

            # Extract targets for foreground anchors
            fg_anchor_idx = candidate_idx[fg_any]
            fg_mask[b, fg_anchor_idx] = True
            target_labels[b, fg_anchor_idx] = gt_labels[best_gt_idx[fg_any]] + 1  # +1 so 0 = background
            target_bboxes[b, fg_anchor_idx] = gt_bboxes[best_gt_idx[fg_any]]

            # Soft classification targets: one-hot weighted by normalized alignment
            fg_indices = fg_any.nonzero(as_tuple=True)[0]
            best_alignment = alignment_masked[fg_indices, best_gt_idx[fg_indices]]
            # Normalize per-GT: divide by max alignment for the same GT
            gt_max_align = alignment.max(dim=0).values.clamp(min=1e-7)  # [M]
            norm_align = best_alignment / gt_max_align[best_gt_idx[fg_indices]]
            norm_align = norm_align.clamp(min=0, max=1.0)

            # Build soft one-hot: label * normalized_alignment
            one_hot = F.one_hot(gt_labels[best_gt_idx[fg_indices]], num_classes=C).float()
            target_scores[b, fg_anchor_idx] = one_hot * norm_align.unsqueeze(1)

        return target_labels, target_bboxes, target_scores, fg_mask
