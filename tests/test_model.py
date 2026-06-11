"""Smoke tests for model forward pass and basic components."""

from __future__ import annotations

import torch

from fruit_detector.model import FruitDetectorV2
from fruit_detector.model.components import DFL, SPPF
from fruit_detector.model.ema import ModelEMA


class TestDFL:
    def test_output_shape(self) -> None:
        dfl = DFL(reg_max=8)
        x = torch.randn(2, 100, 4 * 8)
        out = dfl(x)
        assert out.shape == (2, 100, 4)

    def test_expected_range(self) -> None:
        """Softmax output should be within [0, reg_max-1]."""
        dfl = DFL(reg_max=16)
        x = torch.randn(1, 50, 64)
        out = dfl(x)
        assert out.min() >= 0
        assert out.max() < 16


class TestSPPF:
    def test_output_shape(self) -> None:
        sppf = SPPF(64, 64)
        x = torch.randn(1, 64, 8, 8)
        out = sppf(x)
        assert out.shape == (1, 64, 8, 8)


class TestModelForward:
    def test_forward_outputs(self, tiny_model: FruitDetectorV2, random_input: torch.Tensor) -> None:
        """Model forward should return 5 tensors with correct shapes."""
        cls_pred, box_ltrb, _box_raw, anchor_points, stride_tensor = tiny_model(random_input)

        B = random_input.shape[0]
        # Total number of anchor points across all FPN levels
        num_classes = tiny_model.num_classes
        total_points = anchor_points.shape[0]

        assert cls_pred.shape == (B, total_points, num_classes)
        assert box_ltrb.shape == (B, total_points, 4)
        assert anchor_points.ndim == 2 and anchor_points.shape[1] == 2
        assert stride_tensor.ndim == 2 and stride_tensor.shape[0] == total_points

    def test_no_nans(self, tiny_model: FruitDetectorV2, random_input: torch.Tensor) -> None:
        """Model forward should not produce NaN outputs."""
        cls_pred, box_ltrb, _, _, _ = tiny_model(random_input)
        assert not torch.isnan(cls_pred).any()
        assert not torch.isnan(box_ltrb).any()

    def test_freeze_backbone(self, tiny_model: FruitDetectorV2) -> None:
        tiny_model.freeze_backbone()
        for n, p in tiny_model.named_parameters():
            if "backbone" in n:
                assert not p.requires_grad, f"{n} should be frozen"


class TestModelEMA:
    def test_ema_update(self, tiny_model: FruitDetectorV2) -> None:
        ema = ModelEMA(tiny_model, decay=0.999)
        # Modify original params
        with torch.no_grad():
            for p in tiny_model.parameters():
                p.add_(1.0)
        ema.update(tiny_model)
        assert ema.updates == 1

    def test_ema_state_dict_roundtrip(self, tiny_model: FruitDetectorV2) -> None:
        ema = ModelEMA(tiny_model)
        ema.update(tiny_model)
        state = ema.state_dict()
        assert "model" in state
        assert "updates" in state
        ema.load_state_dict(state)
        assert ema.updates == 1
