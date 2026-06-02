"""Inference utilities — decode raw model outputs into detections."""
import torch
import torchvision.ops as ops


def decode_predictions_v2(cls_pred, box_pred_ltrb, anchor_points, stride_tensor,
                          conf_thresh=0.05, nms_iou=0.45, pre_nms_topk=1000,
                          max_detections=100, img_size=416):
    """Decode one image's predictions from anchor-free outputs.

    Args:
        cls_pred: [N, C] classification logits
        box_pred_ltrb: [N, 4] decoded LTRB distances
        anchor_points: [N, 2]
        stride_tensor: [N, 1]
        conf_thresh: minimum confidence to keep a detection
        nms_iou: IoU threshold for NMS
        pre_nms_topk: max candidates before NMS
        max_detections: max final detections
        img_size: input image size (for clamping)

    Returns:
        boxes_xyxy: [K, 4] detected boxes
        labels: [K] class indices
        scores: [K] confidence scores
    """
    device = cls_pred.device
    scores = torch.sigmoid(cls_pred)  # [N, C]
    max_scores, labels = scores.max(dim=1)  # [N]
    keep = max_scores > conf_thresh
    if keep.sum() == 0:
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,), device=device, dtype=torch.long),
                torch.zeros((0,), device=device))

    max_scores = max_scores[keep]
    labels = labels[keep]
    ltrb = box_pred_ltrb[keep]
    pts = anchor_points[keep]
    st = stride_tensor[keep]

    if max_scores.numel() > pre_nms_topk:
        top_scores, top_idx = max_scores.topk(pre_nms_topk)
        ltrb = ltrb[top_idx]
        labels = labels[top_idx]
        pts = pts[top_idx]
        st = st[top_idx]
        max_scores = top_scores

    # Decode to xyxy
    x1 = pts[:, 0] - ltrb[:, 0] * st[:, 0]
    y1 = pts[:, 1] - ltrb[:, 1] * st[:, 0]
    x2 = pts[:, 0] + ltrb[:, 2] * st[:, 0]
    y2 = pts[:, 1] + ltrb[:, 3] * st[:, 0]
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1).clamp(min=0, max=img_size)

    nms_keep = ops.batched_nms(boxes_xyxy, max_scores, labels, iou_threshold=nms_iou)
    nms_keep = nms_keep[:max_detections]
    return boxes_xyxy[nms_keep], labels[nms_keep], max_scores[nms_keep]
