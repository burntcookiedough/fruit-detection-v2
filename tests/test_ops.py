"""Tests for ops: anchor points, inference decoding, and TTA."""

from __future__ import annotations

import torch

from fruit_detector.config import NUM_CLASSES
from fruit_detector.ops.anchor_points import decode_boxes, generate_anchor_points_and_strides
from fruit_detector.ops.inference import decode_predictions_v2
from fruit_detector.utils.tta import unflip_tta_predictions


class TestAnchorPoints:
    def test_generate(self) -> None:
        ap, st = generate_anchor_points_and_strides(64, [8, 16, 32])
        # 64/8=8 -> 64pts, 64/16=4 -> 16pts, 64/32=2 -> 4pts = 84
        assert ap.shape == (84, 2)
        assert st.shape == (84, 1)

    def test_decode_boxes(self) -> None:
        ap, st = generate_anchor_points_and_strides(64, [8])
        num_pts = ap.shape[0]
        ltrb = torch.ones(1, num_pts, 4)
        decoded = decode_boxes(ap, ltrb, st)
        assert decoded.shape == (1, num_pts, 4)


class TestInference:
    def test_decode_empty(self) -> None:
        """When all scores are 0, should return empty tensors."""
        cls_pred = torch.zeros(100, NUM_CLASSES)
        box_ltrb = torch.zeros(100, 4)
        ap = torch.rand(100, 2)
        st = torch.ones(100, 1) * 8
        boxes, labels, scores = decode_predictions_v2(
            cls_pred,
            box_ltrb,
            ap,
            st,
            conf_thresh=0.5,
            nms_iou=0.5,
            pre_nms_topk=100,
            max_detections=10,
            img_size=64,
        )
        assert len(boxes) == 0
        assert len(labels) == 0
        assert len(scores) == 0


class TestTTA:
    def test_unflip_roundtrip(self) -> None:
        """Flipping predictions twice should approximately recover originals."""
        strides = [8, 16]
        img_size = 32
        total_pts = sum((img_size // s) ** 2 for s in strides)

        cls_pred = torch.randn(1, total_pts, NUM_CLASSES)
        box_ltrb = torch.randn(1, total_pts, 4)

        # Flip once
        cls_f1, box_f1 = unflip_tta_predictions(cls_pred, box_ltrb, img_size, strides)
        # Flip again
        cls_f2, _box_f2 = unflip_tta_predictions(cls_f1, box_f1, img_size, strides)

        # Classification predictions should be exactly recovered (flip is involution)
        torch.testing.assert_close(cls_f2, cls_pred, atol=1e-6, rtol=1e-6)
