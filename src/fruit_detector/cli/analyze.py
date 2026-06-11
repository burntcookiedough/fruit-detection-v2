"""CLI entry point for dataset analysis."""

from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

from ..config import CLASS_NAMES, DATA_DIR

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("analyze", help="Analyze dataset class distribution")
    p.add_argument("--data-dir", type=str, default=DATA_DIR)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    run_analysis(args.data_dir)


def run_analysis(data_dir: str = DATA_DIR) -> None:
    """Print class distribution for each dataset split."""
    for split in ["train", "valid", "test"]:
        lbl_dir = os.path.join(data_dir, split, "labels")
        if not os.path.isdir(lbl_dir):
            continue
        c: Counter[int] = Counter()
        n_files = 0
        n_boxes = 0
        for fname in os.listdir(lbl_dir):
            if not fname.endswith(".txt"):
                continue
            n_files += 1
            with open(os.path.join(lbl_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c[int(parts[0])] += 1
                        n_boxes += 1
        total = sum(c.values())
        logger.info("")
        logger.info("=== %s (%d images, %d boxes) ===", split, n_files, n_boxes)
        for i in range(len(CLASS_NAMES)):
            cnt = c.get(i, 0)
            logger.info(
                "  %d %12s: %6d (%5.1f%%)",
                i,
                CLASS_NAMES[i],
                cnt,
                100 * cnt / max(total, 1),
            )
        if c:
            mx = max(c.values())
            mn = min(c.values())
            logger.info("  Max/Min ratio: %.1fx", mx / max(mn, 1))
