"""Tests for data loading and collation."""

from __future__ import annotations

from pathlib import Path

import pytest

from fruit_detector.data import FruitDataset, collate_fn


class TestFruitDataset:
    def test_load(self, synthetic_dataset_dir: Path) -> None:
        ds = FruitDataset(
            img_dir=str(synthetic_dataset_dir / "images"),
            lbl_dir=str(synthetic_dataset_dir / "labels"),
            img_size=32,
            augment=False,
        )
        assert len(ds) == 5

    def test_item_shapes(self, synthetic_dataset_dir: Path) -> None:
        ds = FruitDataset(
            img_dir=str(synthetic_dataset_dir / "images"),
            lbl_dir=str(synthetic_dataset_dir / "labels"),
            img_size=32,
            augment=False,
        )
        img, boxes, labels = ds[0]
        assert img.shape == (3, 32, 32)
        assert boxes.ndim == 2
        assert labels.ndim == 1

    def test_collate(self, synthetic_dataset_dir: Path) -> None:
        ds = FruitDataset(
            img_dir=str(synthetic_dataset_dir / "images"),
            lbl_dir=str(synthetic_dataset_dir / "labels"),
            img_size=32,
            augment=False,
        )
        batch = [ds[i] for i in range(min(3, len(ds)))]
        stacked, boxes_list, labels_list = collate_fn(batch)  # type: ignore[misc]
        assert stacked.shape[0] == len(batch)
        assert len(boxes_list) == len(batch)
        assert len(labels_list) == len(batch)

    def test_augmented_no_crash(self, synthetic_dataset_dir: Path) -> None:
        """Augmentation pipeline should not crash on tiny images."""
        ds = FruitDataset(
            img_dir=str(synthetic_dataset_dir / "images"),
            lbl_dir=str(synthetic_dataset_dir / "labels"),
            img_size=32,
            augment=True,
            mosaic_prob=0.5,
            mixup_prob=0.5,
            copy_paste_prob=0.5,
        )
        # Run 20 random samples to exercise all augmentation paths
        for _ in range(20):
            img, _boxes, _labels = ds[0]
            assert img.shape == (3, 32, 32)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            FruitDataset(
                img_dir=str(tmp_path / "nonexistent"),
                lbl_dir=str(tmp_path / "nonexistent"),
            )
