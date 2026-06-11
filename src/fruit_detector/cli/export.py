"""CLI entry point for ONNX export."""

from __future__ import annotations

import argparse
import logging

import torch

from ..config import (
    BACKBONE_NAME,
    DEFAULT_ONNX_OUTPUT,
    DEFAULT_WEIGHTS,
    IMG_SIZE,
    NUM_CLASSES,
    STRIDES,
)
from ..model import FruitDetectorV2
from ..utils.checkpoint import infer_model_options, load_detector_state_dict

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("export", help="Export model to ONNX format")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=str, default=DEFAULT_ONNX_OUTPUT)
    p.add_argument("--size", type=int, default=IMG_SIZE)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    export_onnx(args.weights, args.out, args.size)


def export_onnx(
    weights_path: str = DEFAULT_WEIGHTS,
    output_path: str = DEFAULT_ONNX_OUTPUT,
    img_size: int = IMG_SIZE,
) -> None:
    """Export the model to ONNX format."""
    logger.info("Loading weights from %s...", weights_path)
    state_dict = load_detector_state_dict(weights_path, map_location="cpu")
    model_options = infer_model_options(state_dict)

    model = FruitDetectorV2(
        num_classes=NUM_CLASSES,
        img_size=img_size,
        backbone_name=BACKBONE_NAME,
        pretrained=False,
        strides=STRIDES,
        **model_options,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)

    logger.info("Exporting to %s...", output_path)
    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=[
            "cls_pred",
            "box_pred_ltrb",
            "box_pred_raw",
            "anchor_points",
            "stride_tensor",
        ],
        dynamic_axes={
            "input": {0: "batch_size"},
            "cls_pred": {0: "batch_size"},
            "box_pred_ltrb": {0: "batch_size"},
            "box_pred_raw": {0: "batch_size"},
        },
    )
    logger.info("Export successful!")
