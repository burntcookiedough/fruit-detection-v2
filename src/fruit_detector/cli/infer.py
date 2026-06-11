"""CLI entry point for image inference."""

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
    DEFAULT_INPUT_IMAGE,
    DEFAULT_OUTPUT_IMAGE,
    DEFAULT_WEIGHTS,
    IMG_SIZE,
    LABEL_FONT_PATH,
    MAX_DETECTIONS,
    NMS_IOU,
    NUM_CLASSES,
    PRE_NMS_TOPK,
    STRIDES,
)
from ..model import FruitDetectorV2
from ..ops.inference import decode_predictions_v2
from ..utils.checkpoint import infer_model_options, load_detector_state_dict, require_file
from ..utils.tta import unflip_tta_predictions
from ..utils.visualization import draw_detections, load_label_font

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("infer", help="Run inference on an image")
    p.add_argument("--image", type=str, default=DEFAULT_INPUT_IMAGE)
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--out", type=str, default=DEFAULT_OUTPUT_IMAGE)
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--nms-iou", type=float, default=NMS_IOU)
    p.add_argument("--pre-nms-topk", type=int, default=PRE_NMS_TOPK)
    p.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    p.add_argument("--no-tta", action="store_true")
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    run_inference(
        image_path=args.image,
        weights_path=args.weights,
        output_path=args.out,
        conf_thresh=args.conf,
        nms_iou=args.nms_iou,
        pre_nms_topk=args.pre_nms_topk,
        max_detections=args.max_detections,
        use_tta=not args.no_tta,
    )


def run_inference(
    image_path: str,
    weights_path: str,
    output_path: str,
    conf_thresh: float = CONF_THRESH,
    nms_iou: float = NMS_IOU,
    pre_nms_topk: int = PRE_NMS_TOPK,
    max_detections: int = MAX_DETECTIONS,
    use_tta: bool = True,
) -> Path:
    """Run inference on an image and save the annotated result."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    weights_path_resolved = require_file(weights_path, "Weights file")
    image_path_resolved = require_file(image_path, "Input image")
    logger.info("Loading weights from %s...", weights_path_resolved)
    state_dict = load_detector_state_dict(weights_path_resolved, map_location=device)
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
    logger.info("Model loaded successfully.")

    img = Image.open(image_path_resolved).convert("RGB")
    orig_w, orig_h = img.size
    logger.info("Original image size: %dx%d", orig_w, orig_h)

    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)

    with (
        torch.no_grad(),
        torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"),
    ):
        cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
        if use_tta:
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
        conf_thresh=conf_thresh,
        nms_iou=nms_iou,
        pre_nms_topk=pre_nms_topk,
        max_detections=max_detections,
        img_size=IMG_SIZE,
    )

    logger.info("Found %d detections above confidence %.2f:", len(boxes), conf_thresh)

    # Scale boxes to original image size
    scaled_boxes: list[list[float]] = []
    label_names: list[str] = []
    score_values: list[float] = []

    for i in range(len(boxes)):
        b = boxes[i].clone()
        b[0] = b[0] * orig_w / IMG_SIZE
        b[1] = b[1] * orig_h / IMG_SIZE
        b[2] = b[2] * orig_w / IMG_SIZE
        b[3] = b[3] * orig_h / IMG_SIZE
        scaled_boxes.append([b[0].item(), b[1].item(), b[2].item(), b[3].item()])

        cls_idx = int(labels[i].item())
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
        label_names.append(cls_name)
        score_values.append(scores[i].item())
        logger.info(
            "  - %s (%.2f%%): [%.1f, %.1f, %.1f, %.1f]",
            cls_name,
            score_values[-1] * 100,
            *scaled_boxes[-1],
        )

    font = load_label_font(max(12, int(orig_h * 0.02)), LABEL_FONT_PATH)
    draw_detections(img, scaled_boxes, label_names, score_values, font=font)

    out = Path(output_path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    logger.info("Saved annotated image to: %s", out)
    return out
