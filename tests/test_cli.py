"""Tests for the command-line interface entry points."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from fruit_detector.cli import main
from fruit_detector.config import NUM_CLASSES, STRIDES
from fruit_detector.model import FruitDetectorV2


@pytest.fixture
def dummy_checkpoint(tmp_path: Path) -> Path:
    """Create a dummy model checkpoint for testing CLI loading."""
    model = FruitDetectorV2(
        num_classes=NUM_CLASSES,
        img_size=64,
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
    checkpoint_path = tmp_path / "dummy_best.pt"
    torch.save(
        {
            "epoch": 5,
            "best_map50": 0.85,
            "model_state_dict": model.state_dict(),
            "config": {
                "neck_channels": 32,
                "num_head_convs": 1,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def dummy_image(tmp_path: Path) -> Path:
    """Create a dummy image for inference testing."""
    img_path = tmp_path / "test_fruit.jpg"
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    img.save(img_path)
    return img_path


def test_cli_help() -> None:
    """Test that running with no args exits with 1 (prints help)."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 1


def test_cli_train_zero_epochs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that training with 0 epochs exits early and cleanly."""
    # We pass --epochs 0, which should return instantly after validation of env
    # Use monkeypatch to avoid running validation check that might fail on empty directories
    import fruit_detector.engine.trainer as trainer

    monkeypatch.setattr(trainer, "_validate_environment", lambda: None)

    main(["train", "--epochs", "0"])


def test_cli_analyze(tmp_path: Path) -> None:
    """Test the dataset analyzer command."""
    # Create structured folders
    for split in ["train", "valid"]:
        lbl_dir = tmp_path / split / "labels"
        lbl_dir.mkdir(parents=True)
        # Write a dummy label file
        with open(lbl_dir / "001.txt", "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")

    # Call main
    main(["analyze", "--data-dir", str(tmp_path)])


def test_cli_export(dummy_checkpoint: Path, tmp_path: Path) -> None:
    """Test model export command."""
    out_onnx = tmp_path / "model.onnx"
    main(["export", "--weights", str(dummy_checkpoint), "--out", str(out_onnx), "--size", "64"])
    assert out_onnx.is_file()


def test_cli_infer(dummy_checkpoint: Path, dummy_image: Path, tmp_path: Path) -> None:
    """Test single image inference command."""
    out_img = tmp_path / "output.jpg"
    main(
        [
            "infer",
            "--image",
            str(dummy_image),
            "--weights",
            str(dummy_checkpoint),
            "--out",
            str(out_img),
            "--no-tta",
        ]
    )
    assert out_img.is_file()


def test_cli_verify(dummy_checkpoint: Path, dummy_image: Path, tmp_path: Path) -> None:
    """Test validation verify command."""
    # Put our dummy image inside a validation folder structure
    val_images_dir = tmp_path / "valid_images"
    val_images_dir.mkdir()
    img_copy = val_images_dir / "val_img.jpg"
    # Copy dummy image there
    img = Image.open(dummy_image)
    img.save(img_copy)

    main(
        [
            "verify",
            "--weights",
            str(dummy_checkpoint),
            "--image-dir",
            str(val_images_dir),
            "--limit",
            "1",
        ]
    )
