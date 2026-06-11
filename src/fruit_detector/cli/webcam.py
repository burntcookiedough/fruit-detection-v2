"""CLI entry point for webcam inference."""

from __future__ import annotations

import argparse
import logging
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from ..config import (
    BACKBONE_NAME,
    CLASS_NAMES,
    CONF_THRESH,
    DEFAULT_CAMERA_ID,
    DEFAULT_WEIGHTS,
    IMG_SIZE,
    MAX_DETECTIONS,
    NMS_IOU,
    NUM_CLASSES,
    PRE_NMS_TOPK,
    STRIDES,
)
from ..model import FruitDetectorV2
from ..ops.inference import decode_predictions_v2
from ..utils.checkpoint import infer_model_options, load_detector_state_dict
from ..utils.tta import unflip_tta_predictions

logger = logging.getLogger(__name__)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("webcam", help="Real-time webcam inference")
    p.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    p.add_argument("--conf", type=float, default=CONF_THRESH)
    p.add_argument("--nms-iou", type=float, default=NMS_IOU)
    p.add_argument("--cam", type=int, default=DEFAULT_CAMERA_ID)
    p.add_argument("--tta", action="store_true")
    p.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    run_webcam(
        weights_path=args.weights,
        conf_thresh=args.conf,
        nms_iou=args.nms_iou,
        camera_id=args.cam,
        use_tta=args.tta,
    )


def run_webcam(
    weights_path: str = DEFAULT_WEIGHTS,
    conf_thresh: float = CONF_THRESH,
    nms_iou: float = NMS_IOU,
    camera_id: int = DEFAULT_CAMERA_ID,
    use_tta: bool = False,
) -> None:
    """Run real-time webcam inference."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for webcam inference. Install with: pip install opencv-python"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("Loading weights from %s...", weights_path)

    state_dict = load_detector_state_dict(weights_path, map_location=device)
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

    logger.info("Opening webcam (ID: %d)...", camera_id)
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam with ID {camera_id}.")

    bgr_colors = [
        (48, 59, 255),
        (0, 204, 255),
        (0, 149, 255),
        (10, 214, 255),
        (222, 82, 175),
        (89, 199, 52),
        (250, 200, 90),
        (48, 28, 164),
    ]

    logger.info("Press 'q' to quit, 't' to toggle TTA, '1'-'9' to set confidence.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Could not read frame from webcam.")
            break

        orig_h, orig_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(img_resized)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            if use_tta:
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
            else:
                cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)

        boxes, labels, scores = decode_predictions_v2(
            cls_pred[0],
            box_ltrb[0],
            anchor_points,
            stride_tensor,
            conf_thresh=conf_thresh,
            nms_iou=nms_iou,
            pre_nms_topk=PRE_NMS_TOPK,
            max_detections=MAX_DETECTIONS,
            img_size=IMG_SIZE,
        )

        for i in range(len(boxes)):
            b = boxes[i].clone()
            x1 = int(b[0].item() * orig_w / IMG_SIZE)
            y1 = int(b[1].item() * orig_h / IMG_SIZE)
            x2 = int(b[2].item() * orig_w / IMG_SIZE)
            y2 = int(b[3].item() * orig_h / IMG_SIZE)

            cls_idx = int(labels[i].item())
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
            score = scores[i].item()
            color = bgr_colors[cls_idx % len(bgr_colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{cls_name} {score:.0%}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(
                frame,
                label_text,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time

        overlay_text = (
            f"FPS: {fps:.1f} | Conf: {conf_thresh:.2f} | TTA: {'ON' if use_tta else 'OFF'}"
        )
        cv2.putText(
            frame,
            overlay_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Fruit Detector v2 Webcam Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            use_tta = not use_tta
            logger.info("TTA toggled: %s", use_tta)
        elif ord("1") <= key <= ord("9"):
            conf_thresh = (key - ord("0")) / 10.0
            logger.info("Confidence threshold: %.2f", conf_thresh)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam inference stopped.")
