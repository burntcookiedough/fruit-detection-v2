"""Anchor point generation and box decoding for anchor-free detection."""
import torch


def generate_anchor_points_and_strides(
    img_size: int,
    strides: list = None,
    device: str = 'cpu',
) -> tuple:
    """Generate anchor points (grid centers) and per-point strides.

    For an anchor-free detector, each grid cell has exactly one anchor point
    at its center. No aspect ratio or scale variations — just one point.

    Args:
        img_size: input image size (e.g., 416)
        strides: list of feature map strides (default: [8, 16, 32])
        device: torch device

    Returns:
        anchor_points: [N, 2] tensor of (x, y) center coordinates in pixel space
        stride_tensor: [N, 1] tensor of stride for each point

    Example for img_size=416:
        stride  8: 52×52 = 2704 points
        stride 16: 26×26 =  676 points
        stride 32: 13×13 =  169 points
        Total N = 3549
    """
    if strides is None:
        strides = [8, 16, 32]

    all_points = []
    all_strides = []

    for stride in strides:
        fm_size = img_size // stride
        # Grid centers: (0.5, 0.5), (1.5, 0.5), ... shifted to pixel space
        shifts = (torch.arange(fm_size, dtype=torch.float32, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts, shifts, indexing='ij')
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [fm_size², 2]
        all_points.append(points)
        all_strides.append(torch.full((points.shape[0], 1), stride, dtype=torch.float32, device=device))

    return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)


def decode_boxes(
    anchor_points: torch.Tensor,
    pred_ltrb: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    """Decode predicted left-top-right-bottom distances into xyxy boxes.

    The model predicts 4 non-negative distances from each anchor point to
    the box edges. Multiply by stride to get pixel distances.

    Args:
        anchor_points: [N, 2] center points (x, y) in pixel space
        pred_ltrb: [B, N, 4] predicted distances (left, top, right, bottom)
        stride_tensor: [N, 1] stride per point

    Returns:
        boxes_xyxy: [B, N, 4] decoded boxes in (x1, y1, x2, y2) format
    """
    # anchor_points: [N, 2] → [1, N, 2] for broadcasting with batch dim
    xy = anchor_points.unsqueeze(0)  # [1, N, 2]
    stride = stride_tensor.unsqueeze(0)  # [1, N, 1]

    # pred_ltrb should be non-negative (after softmax+integral in DFL)
    lt = pred_ltrb[..., :2] * stride  # left, top distances  [B, N, 2]
    rb = pred_ltrb[..., 2:] * stride  # right, bottom distances [B, N, 2]

    x1y1 = xy - lt  # top-left corner
    x2y2 = xy + rb  # bottom-right corner

    return torch.cat([x1y1, x2y2], dim=-1)  # [B, N, 4] xyxy
