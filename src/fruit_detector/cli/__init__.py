"""CLI dispatcher — ``fruit-detect`` command entry point."""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    """Top-level CLI dispatcher for the fruit detector."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        prog="fruit-detect",
        description="Anchor-free fruit detection toolkit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train ---
    from .train import add_parser as add_train

    add_train(subparsers)

    # --- infer ---
    from .infer import add_parser as add_infer

    add_infer(subparsers)

    # --- webcam ---
    from .webcam import add_parser as add_webcam

    add_webcam(subparsers)

    # --- verify ---
    from .verify import add_parser as add_verify

    add_verify(subparsers)

    # --- export ---
    from .export import add_parser as add_export

    add_export(subparsers)

    # --- analyze ---
    from .analyze import add_parser as add_analyze

    add_analyze(subparsers)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)
