"""Operations subpackage — anchor points, loss, assignment, and inference decoding."""

from .anchor_points import decode_boxes, generate_anchor_points_and_strides
from .inference import decode_predictions_v2
from .loss import DetectionLossV2

__all__ = [
    "DetectionLossV2",
    "decode_boxes",
    "decode_predictions_v2",
    "generate_anchor_points_and_strides",
]
