"""Anchor point generation and box decoding for anchor-free detection."""

from __future__ import annotations

import torch


def generate_anchor_points_and_strides(
    img_size: int,
    strides: list[int] | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate anchor points (grid centers) and per-point strides.

    For an anchor-free detector, each grid cell has exactly one anchor point
    at its center. No aspect ratio or scale variations.

    Args:
        img_size: input image size (e.g., 416)
        strides: list of feature map strides (default: ``[8, 16, 32]``)
        device: torch device

    Returns:
        anchor_points: ``[N, 2]`` tensor of ``(x, y)`` center coordinates
        stride_tensor: ``[N, 1]`` tensor of stride for each point
    """
    if strides is None:
        strides = [8, 16, 32]

    all_points: list[torch.Tensor] = []
    all_strides: list[torch.Tensor] = []

    for stride in strides:
        fm_size = img_size // stride
        shifts = (torch.arange(fm_size, dtype=torch.float32, device=device) + 0.5) * stride
        yy, xx = torch.meshgrid(shifts, shifts, indexing="ij")
        points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        all_points.append(points)
        all_strides.append(
            torch.full(
                (points.shape[0], 1),
                stride,
                dtype=torch.float32,
                device=device,
            )
        )

    return torch.cat(all_points, dim=0), torch.cat(all_strides, dim=0)


def decode_boxes(
    anchor_points: torch.Tensor,
    pred_ltrb: torch.Tensor,
    stride_tensor: torch.Tensor,
) -> torch.Tensor:
    """Decode predicted left-top-right-bottom distances into xyxy boxes.

    Args:
        anchor_points: ``[N, 2]`` center points ``(x, y)`` in pixel space
        pred_ltrb: ``[B, N, 4]`` predicted distances ``(left, top, right, bottom)``
        stride_tensor: ``[N, 1]`` stride per point

    Returns:
        ``[B, N, 4]`` decoded boxes in ``(x1, y1, x2, y2)`` format
    """
    xy = anchor_points.unsqueeze(0)
    stride = stride_tensor.unsqueeze(0)

    lt = pred_ltrb[..., :2] * stride
    rb = pred_ltrb[..., 2:] * stride

    x1y1 = xy - lt
    x2y2 = xy + rb

    return torch.cat([x1y1, x2y2], dim=-1)
