"""Engine subpackage — training and evaluation pipeline."""

from .trainer import run_training, validate

__all__ = ["run_training", "validate"]
