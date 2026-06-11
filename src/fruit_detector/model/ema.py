"""Exponential Moving Average (EMA) of model weights."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain an exponential moving average of model parameters.

    The EMA model is used for validation and inference, producing smoother
    and often more accurate predictions than the raw training model.

    Args:
        model: the model to track
        decay: EMA decay factor (higher = slower update)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters from the training model."""
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.lerp_(model_p.detach(), 1.0 - d)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self) -> dict:
        """Return EMA state for checkpointing."""
        return {"model": self.ema_model.state_dict(), "updates": self.updates}

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore EMA state from a checkpoint."""
        if "model" in state_dict and "updates" in state_dict:
            self.ema_model.load_state_dict(state_dict["model"])
            self.updates = state_dict["updates"]
        else:
            self.ema_model.load_state_dict(state_dict)
