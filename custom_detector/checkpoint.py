"""Checkpoint loading helpers shared by inference, verification, and export."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def require_file(path: str | Path, description: str) -> Path:
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


def infer_model_options(state_dict: dict) -> dict:
    """Infer optional architecture flags from checkpoint parameter names."""
    keys = state_dict.keys()
    return {
        "use_sppf": any("sppf" in key for key in keys),
        "use_cem": any("cem" in key for key in keys),
        "use_grn": any(".grn.gamma" in key for key in keys),
    }


def _normalize_cem_weights(state_dict: dict) -> None:
    """Support older checkpoints where CEM projection weights were 2-D."""
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if "cem." in key and "fc." in key and "weight" in key and hasattr(value, "dim"):
            if value.dim() == 2:
                state_dict[key] = value.unsqueeze(-1).unsqueeze(-1)
