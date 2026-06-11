"""Test fixtures for the fruit detector test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fruit_detector.config import NUM_CLASSES, STRIDES
from fruit_detector.model import FruitDetectorV2


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing (GPU not required)."""
    return torch.device("cpu")


@pytest.fixture
def tiny_img_size() -> int:
    """Small image size for fast tests."""
    return 64


@pytest.fixture
def tiny_model(tiny_img_size: int) -> FruitDetectorV2:
    """A minimal FruitDetectorV2 for testing (no pretrained weights)."""
    model = FruitDetectorV2(
        num_classes=NUM_CLASSES,
        img_size=tiny_img_size,
        backbone_name="convnext_femto.d1_in1k",
        pretrained=False,
        neck_channels=32,
        reg_max=4,
        strides=STRIDES,
        num_head_convs=1,
        use_sppf=True,
        use_cem=True,
        use_grn=False,
    )
    model.eval()
    return model


@pytest.fixture
def random_input(tiny_img_size: int) -> torch.Tensor:
    """Random input tensor for the model."""
    return torch.randn(1, 3, tiny_img_size, tiny_img_size)


@pytest.fixture
def synthetic_dataset_dir(tmp_path: Path) -> Path:
    """Create a synthetic YOLO-format dataset with 5 tiny images."""
    img_dir = tmp_path / "images"
    lbl_dir = tmp_path / "labels"
    img_dir.mkdir()
    lbl_dir.mkdir()

    rng = np.random.RandomState(42)
    for i in range(5):
        # Create a small random image
        img_arr = rng.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        from PIL import Image

        img = Image.fromarray(img_arr)
        img.save(img_dir / f"img_{i:03d}.jpg")

        # Create YOLO label (class cx cy w h) — normalized coords
        n_boxes = rng.randint(1, 4)
        with open(lbl_dir / f"img_{i:03d}.txt", "w") as f:
            for _ in range(n_boxes):
                cls = rng.randint(0, NUM_CLASSES)
                cx = rng.uniform(0.2, 0.8)
                cy = rng.uniform(0.2, 0.8)
                w = rng.uniform(0.1, 0.4)
                h = rng.uniform(0.1, 0.4)
                f.write(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    return tmp_path
