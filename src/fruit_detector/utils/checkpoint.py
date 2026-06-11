"""Checkpoint loading helpers shared by inference, verification, and export."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def require_file(path: str | Path, description: str) -> Path:
    """Verify a file exists and return its resolved path."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_detector_state_dict(weights_path: str | Path, map_location: Any = "cpu") -> dict:
    """Load detector weights, preferring EMA weights when available."""
    path = require_file(weights_path, "Weights file")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if "ema_state_dict" in checkpoint:
        ema_state = checkpoint["ema_state_dict"]
        state_dict = ema_state.get("model", ema_state.get("shadow", ema_state))
        if isinstance(state_dict, dict):
            _normalize_cem_weights(state_dict)
            return state_dict

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    _normalize_cem_weights(state_dict)
    return state_dict


def load_checkpoint_metadata(weights_path: str | Path, map_location: Any = "cpu") -> dict:
    """Load only metadata from a checkpoint (epoch, metrics, config)."""
    path = require_file(weights_path, "Weights file")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        return {}
    return {
        "epoch": checkpoint.get("epoch"),
        "best_map50": checkpoint.get("best_map50"),
        "best_loss": checkpoint.get("best_loss"),
        "config": checkpoint.get("config"),
    }


def infer_model_options(state_dict: dict) -> dict[str, Any]:
    """Infer optional architecture flags and dimensions from checkpoint parameter names/shapes."""
    keys = state_dict.keys()

    use_sppf = any("sppf" in key for key in keys)
    use_cem = any("cem" in key for key in keys)
    use_grn = any(".grn.gamma" in key for key in keys)

    # Infer neck_channels from lateral conv weights (e.g. neck.lateral_p3.conv.weight)
    # Default to 96 if not found (fallback for older checkpoints)
    neck_channels = 96
    for key in keys:
        if "neck.lateral_p3.conv.weight" in key:
            neck_channels = state_dict[key].shape[0]
            break

    # Infer num_head_convs from heads structure
    # The keys in decoupled heads look like:
    # "heads.0.cls_conv.0.conv.weight" (layer 0)
    # "heads.0.cls_conv.1.conv.weight" (layer 1, if num_head_convs=2)
    # So we can count how many modules are in heads.0.cls_conv
    num_head_convs = 1
    cls_conv_indices = []
    for key in keys:
        if "heads.0.cls_conv." in key:
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                cls_conv_indices.append(int(parts[3]))
    if cls_conv_indices:
        num_head_convs = max(cls_conv_indices) + 1

    # Infer reg_max from heads structure
    reg_max = 16
    for key in keys:
        if "heads.0.reg_pred.bias" in key or "heads.0.reg_pred.weight" in key:
            reg_max = state_dict[key].shape[0] // 4
            break

    return {
        "use_sppf": use_sppf,
        "use_cem": use_cem,
        "use_grn": use_grn,
        "neck_channels": neck_channels,
        "num_head_convs": num_head_convs,
        "reg_max": reg_max,
    }


def _normalize_cem_weights(state_dict: dict) -> None:
    """Support older checkpoints where CEM projection weights were 2-D."""
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if "cem." in key and "fc." in key and "weight" in key and hasattr(value, "dim"):
            if value.dim() == 2:
                state_dict[key] = value.unsqueeze(-1).unsqueeze(-1)
