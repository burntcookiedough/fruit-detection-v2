"""Data subpackage — dataset and collation utilities."""

from .dataset import FruitDataset, collate_fn

__all__ = ["FruitDataset", "collate_fn"]
