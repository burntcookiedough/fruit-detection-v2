"""Run inference on an image and save the annotated results."""
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

from config import (IMG_SIZE, NUM_CLASSES, BACKBONE_NAME, NECK_CHANNELS,
                    REG_MAX, STRIDES, CLASS_NAMES, DEFAULT_INPUT_IMAGE,
                    DEFAULT_OUTPUT_IMAGE, DEFAULT_WEIGHTS, CONF_THRESH,
                    NMS_IOU, PRE_NMS_TOPK, MAX_DETECTIONS, LABEL_FONT_PATH)
from custom_detector.checkpoint import (
    infer_model_options,
    load_detector_state_dict,
    require_file,
)
from custom_detector.model import FruitDetectorV2
from custom_detector.inference import decode_predictions_v2


def run_inference(
    image_path,
    weights_path,
    output_path,
    conf_thresh=CONF_THRESH,
    nms_iou=NMS_IOU,
    pre_nms_topk=PRE_NMS_TOPK,
    max_detections=MAX_DETECTIONS,
    use_tta=True,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    weights_path = require_file(weights_path, "Weights file")
    image_path = require_file(image_path, "Input image")
    print(f"Loading weights from {weights_path}...")
    state_dict = load_detector_state_dict(weights_path, map_location=device)
    model_options = infer_model_options(state_dict)

    model = FruitDetectorV2(
        num_classes=NUM_CLASSES, img_size=IMG_SIZE,
        backbone_name=BACKBONE_NAME, pretrained=False,
        neck_channels=NECK_CHANNELS, reg_max=REG_MAX, strides=STRIDES,
        num_head_convs=1,
        **model_options,
    ).to(device)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Model loaded successfully.")

    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original image size: {orig_w}x{orig_h}")

    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
        if use_tta:
            tensor_flipped = torch.flip(tensor, dims=[3])
            cls_pred_f, box_ltrb_f, _, _, _ = model(tensor_flipped)
            cls_pred_f, box_ltrb_f = _unflip_tta_predictions(cls_pred_f, box_ltrb_f)
            cls_pred = (cls_pred + cls_pred_f) / 2.0
            box_ltrb = (box_ltrb + box_ltrb_f) / 2.0

    boxes, labels, scores = decode_predictions_v2(
        cls_pred[0], box_ltrb[0], anchor_points, stride_tensor,
        conf_thresh=conf_thresh, nms_iou=nms_iou,
        pre_nms_topk=pre_nms_topk, max_detections=max_detections,
        img_size=IMG_SIZE
    )

    print(f"Found {len(boxes)} detections above confidence {conf_thresh}:")

    colors = [
        (255, 59, 48), (255, 204, 0), (255, 149, 0), (255, 214, 10),
        (175, 82, 222), (52, 199, 89), (90, 200, 250), (164, 28, 48),
    ]

    draw = ImageDraw.Draw(img)
    font = _load_label_font(max(12, int(orig_h * 0.02)))

    for i in range(len(boxes)):
        b = boxes[i].clone()
        b[0] = b[0] * orig_w / IMG_SIZE
        b[1] = b[1] * orig_h / IMG_SIZE
        b[2] = b[2] * orig_w / IMG_SIZE
        b[3] = b[3] * orig_h / IMG_SIZE

        cls_idx = labels[i].item()
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
        score = scores[i].item()
        color = colors[cls_idx % len(colors)]

        x1, y1, x2, y2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
        print(f"  - {cls_name} ({score:.2%}): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

        thickness = max(2, int(orig_h * 0.005))
        for t in range(thickness):
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)

        label_text = f"{cls_name} {score:.0%}"
        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:4]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)

        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=color)
        draw.text((x1 + 3, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)

    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image to: {output_path}")
    return output_path


def _unflip_tta_predictions(cls_pred_f, box_ltrb_f):
    cls_f_unflipped = []
    box_f_unflipped = []
    start = 0
    for stride in STRIDES:
        fm_size = IMG_SIZE // stride
        num_pts = fm_size * fm_size

        c_chunk = cls_pred_f[:, start:start + num_pts, :]
        c_chunk = c_chunk.view(1, fm_size, fm_size, -1)
        c_chunk = torch.flip(c_chunk, dims=[2])
        cls_f_unflipped.append(c_chunk.view(1, num_pts, -1))

        b_chunk = box_ltrb_f[:, start:start + num_pts, :]
        b_chunk = b_chunk.view(1, fm_size, fm_size, 4)
        b_chunk = torch.flip(b_chunk, dims=[2])
        box_f_unflipped.append(b_chunk.view(1, num_pts, 4))
        start += num_pts

    cls_pred_f = torch.cat(cls_f_unflipped, dim=1)
    box_ltrb_f = torch.cat(box_f_unflipped, dim=1)
    return cls_pred_f, box_ltrb_f[..., [2, 1, 0, 3]]


def _load_label_font(size):
    if LABEL_FONT_PATH:
        try:
            return ImageFont.truetype(LABEL_FONT_PATH, size=size)
        except OSError:
            print(f"Warning: FRUIT_LABEL_FONT could not be loaded: {LABEL_FONT_PATH}")
    return ImageFont.load_default()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default=DEFAULT_INPUT_IMAGE, help='Path to input image')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS, help='Path to checkpoint')
    parser.add_argument('--out', type=str, default=DEFAULT_OUTPUT_IMAGE, help='Path to save annotated image')
    parser.add_argument('--conf', type=float, default=CONF_THRESH, help='Confidence threshold')
    parser.add_argument('--nms-iou', type=float, default=NMS_IOU, help='NMS IoU threshold')
    parser.add_argument('--pre-nms-topk', type=int, default=PRE_NMS_TOPK, help='Maximum candidates before NMS')
    parser.add_argument('--max-detections', type=int, default=MAX_DETECTIONS, help='Maximum final detections')
    parser.add_argument('--no-tta', action='store_true', help='Disable horizontal flip test-time augmentation')
    args = parser.parse_args()

    run_inference(
        args.image,
        args.weights,
        args.out,
        args.conf,
        args.nms_iou,
        args.pre_nms_topk,
        args.max_detections,
        use_tta=not args.no_tta,
    )
