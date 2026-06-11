"""Model subpackage — detector architecture, backbone, neck, heads, and EMA."""

from .detector import FruitDetectorV2
from .ema import ModelEMA

__all__ = ["FruitDetectorV2", "ModelEMA"]
