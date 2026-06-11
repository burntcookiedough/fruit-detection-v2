"""Integration tests for the training pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

import fruit_detector.config
import fruit_detector.engine.trainer as trainer


def test_run_training(
    synthetic_dataset_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that the training loop can run for one epoch without crashing."""
    img_dir = synthetic_dataset_dir / "images"
    lbl_dir = synthetic_dataset_dir / "labels"

    # Patch config variables
    monkeypatch.setattr(fruit_detector.config, "TRAIN_IMG_DIR", str(img_dir))
    monkeypatch.setattr(fruit_detector.config, "TRAIN_LBL_DIR", str(lbl_dir))
    monkeypatch.setattr(fruit_detector.config, "VAL_IMG_DIR", str(img_dir))
    monkeypatch.setattr(fruit_detector.config, "VAL_LBL_DIR", str(lbl_dir))
    monkeypatch.setattr(fruit_detector.config, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(fruit_detector.config, "WEIGHTS_DIR", str(tmp_path / "runs" / "weights"))
    monkeypatch.setattr(fruit_detector.config, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(fruit_detector.config, "PRETRAINED", False)
    monkeypatch.setattr(fruit_detector.config, "NUM_WORKERS", 0)
    monkeypatch.setattr(fruit_detector.config, "BATCH_SIZE", 2)
    monkeypatch.setattr(fruit_detector.config, "IMG_SIZE", 64)
    monkeypatch.setattr(fruit_detector.config, "NECK_CHANNELS", 32)
    monkeypatch.setattr(fruit_detector.config, "REG_MAX", 4)
    monkeypatch.setattr(fruit_detector.config, "TAL_TOPK", 2)

    # Patch trainer variables (which are imported at module level)
    monkeypatch.setattr(trainer, "TRAIN_IMG_DIR", str(img_dir))
    monkeypatch.setattr(trainer, "TRAIN_LBL_DIR", str(lbl_dir))
    monkeypatch.setattr(trainer, "VAL_IMG_DIR", str(img_dir))
    monkeypatch.setattr(trainer, "VAL_LBL_DIR", str(lbl_dir))
    monkeypatch.setattr(trainer, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(trainer, "WEIGHTS_DIR", str(tmp_path / "runs" / "weights"))
    monkeypatch.setattr(trainer, "CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(trainer, "PRETRAINED", False)
    monkeypatch.setattr(trainer, "NUM_WORKERS", 0)
    monkeypatch.setattr(trainer, "BATCH_SIZE", 2)
    monkeypatch.setattr(trainer, "IMG_SIZE", 64)
    monkeypatch.setattr(trainer, "NECK_CHANNELS", 32)
    monkeypatch.setattr(trainer, "REG_MAX", 4)
    monkeypatch.setattr(trainer, "TAL_TOPK", 2)

    # Run 1 epoch training with limits on train and validation batches
    trainer.run_training(
        num_epochs=1,
        workers=0,
        batch_size=2,
        skip_val=False,
        limit_train_batches=1,
        limit_val_batches=1,
    )
