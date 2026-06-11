"""Test-Time Augmentation (TTA) utilities.

Centralizes the horizontal-flip TTA logic that was previously duplicated
across ``run_inference.py``, ``verify.py``, and ``webcam_inference.py``.
"""

from __future__ import annotations

import torch


def unflip_tta_predictions(
    cls_pred_f: torch.Tensor,
    box_ltrb_f: torch.Tensor,
    img_size: int,
    strides: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unflip predictions from a horizontally-flipped input.

    When TTA is used, the model runs on both the original image and its
    horizontal mirror. This function reverses the spatial flip in the
    predictions so they can be averaged with the original predictions.

    Args:
        cls_pred_f: ``[B, N, C]`` class predictions from flipped input
        box_ltrb_f: ``[B, N, 4]`` box predictions from flipped input
        img_size: input image size
        strides: list of detection strides (e.g., ``[8, 16, 32]``)

    Returns:
        Unflipped ``(cls_pred, box_ltrb)``
    """
    cls_f_unflipped: list[torch.Tensor] = []
    box_f_unflipped: list[torch.Tensor] = []
    start = 0

    for stride in strides:
        fm_size = img_size // stride
        num_pts = fm_size * fm_size

        c_chunk = cls_pred_f[:, start : start + num_pts, :]
        c_chunk = c_chunk.view(1, fm_size, fm_size, -1)
        c_chunk = torch.flip(c_chunk, dims=[2])
        cls_f_unflipped.append(c_chunk.view(1, num_pts, -1))

        b_chunk = box_ltrb_f[:, start : start + num_pts, :]
        b_chunk = b_chunk.view(1, fm_size, fm_size, 4)
        b_chunk = torch.flip(b_chunk, dims=[2])
        box_f_unflipped.append(b_chunk.view(1, num_pts, 4))

        start += num_pts

    cls_pred_f = torch.cat(cls_f_unflipped, dim=1)
    box_ltrb_f = torch.cat(box_f_unflipped, dim=1)
    # Swap left↔right in LTRB predictions
    return cls_pred_f, box_ltrb_f[..., [2, 1, 0, 3]]
