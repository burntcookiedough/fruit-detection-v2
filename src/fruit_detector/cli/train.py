"""CLI entry point for training."""

from __future__ import annotations

import argparse

from ..config import (
    BATCH_SIZE,
    FREEZE_BACKBONE_EPOCHS,
    NUM_EPOCHS,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PREFETCH_FACTOR,
    VAL_EVERY,
)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("train", help="Train the fruit detector")
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--val-every", type=int, default=VAL_EVERY)
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--prefetch-factor", type=int, default=PREFETCH_FACTOR)
    p.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=PERSISTENT_WORKERS,
    )
    p.add_argument("--cache-images", action="store_true")
    p.add_argument("--build-cache", action="store_true")
    p.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=FREEZE_BACKBONE_EPOCHS,
    )
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    from ..engine.trainer import run_training

    run_training(
        num_epochs=args.epochs,
        resume=args.resume,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_every=args.val_every,
        skip_val=args.skip_val,
        workers=args.workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        cache_images=args.cache_images,
        build_cache=args.build_cache,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        batch_size=args.batch_size,
        args=args,
    )
