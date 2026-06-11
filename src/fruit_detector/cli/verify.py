"""CLI entry point for validation verification."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..config import (
    BACKBONE_NAME,
    CLASS_NAMES,
    CONF_THRESH,
    DEFAULT_WEIGHTS,
    IMG_SIZE,
    MAX_DETECTIONS,
    NMS_IOU,
    NUM_CLASSES,
    PRE_NMS_TOPK,
    STRIDES,
    VAL_IMG_DIR,
)
from ..model import FruitDetectorV2
from ..ops.inference import decode_predictions_v2
from ..utils.checkpoint import (
    infer_model_options,
    load_checkpoint_metadata,
    load_detector_state_dict,
)
from ..utils.tta import unflip_tta_predictions

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("verify", help="Run verification on validation images")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--image-dir", type=str, default=VAL_IMG_DIR)
    p.add_argument("--limit", type=int, default=5)
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    run_verify(args.weights, args.image_dir, args.limit)


def run_verify(
    weights_path: str = DEFAULT_WEIGHTS,
    image_dir: str = VAL_IMG_DIR,
    limit: int = 5,
) -> None:
    """Run inference sanity check on validation images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = load_detector_state_dict(weights_path, map_location=device)
    metadata = load_checkpoint_metadata(weights_path, map_location=device)
    model_options = infer_model_options(state_dict)

    model = FruitDetectorV2(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        backbone_name=BACKBONE_NAME,
        pretrained=False,
        strides=STRIDES,
        **model_options,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    logger.info("Loaded model weights.")
    logger.info("Checkpoint epoch: %s", metadata.get("epoch", "?"))
    logger.info("Best mAP50: %s", metadata.get("best_map50", "?"))
    logger.info("Device: %s", device)

    image_dir_path = Path(image_dir).expanduser()
    if not image_dir_path.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir_path}")

    val_images = sorted(
        p for p in image_dir_path.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )[:limit]

    logger.info("Running inference on %d validation images...", len(val_images))

    for img_path in val_images:
        _verify_image(model, img_path, device)

    logger.info("Verification complete.")


def _verify_image(model: FruitDetectorV2, img_path: Path, device: torch.device) -> None:
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)

    with (
        torch.no_grad(),
        torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"),
    ):
        cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
        tensor_flipped = torch.flip(tensor, dims=[3])
        cls_pred_f, box_ltrb_f, _, _, _ = model(tensor_flipped)
        cls_pred_f, box_ltrb_f = unflip_tta_predictions(
            cls_pred_f,
            box_ltrb_f,
            IMG_SIZE,
            STRIDES,
        )
        cls_pred = (cls_pred + cls_pred_f) / 2.0
        box_ltrb = (box_ltrb + box_ltrb_f) / 2.0

    boxes, labels, scores = decode_predictions_v2(
        cls_pred[0],
        box_ltrb[0],
        anchor_points,
        stride_tensor,
        conf_thresh=CONF_THRESH,
        nms_iou=NMS_IOU,
        pre_nms_topk=PRE_NMS_TOPK,
        max_detections=MAX_DETECTIONS,
        img_size=IMG_SIZE,
    )

    logger.info("%s (%dx%d) -> %d detections:", img_path.name, orig_w, orig_h, len(boxes))
    if len(boxes) == 0:
        logger.info("  (no detections above confidence threshold)")
        return

    for i in range(len(boxes)):
        cls_idx = int(labels[i].item())
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
        b = boxes[i].clone()
        b[0] = b[0] * orig_w / IMG_SIZE
        b[1] = b[1] * orig_h / IMG_SIZE
        b[2] = b[2] * orig_w / IMG_SIZE
        b[3] = b[3] * orig_h / IMG_SIZE
        logger.info(
            "  [%12s] conf=%.4f  box=[%.1f, %.1f, %.1f, %.1f]",
            cls_name,
            scores[i].item(),
            b[0].item(),
            b[1].item(),
            b[2].item(),
            b[3].item(),
        )
