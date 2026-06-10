"""Run a small inference sanity check on validation images."""
import argparse
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms.functional as TF

from config import (IMG_SIZE, NUM_CLASSES, BACKBONE_NAME, NECK_CHANNELS,
                    REG_MAX, STRIDES, VAL_IMG_DIR, CLASS_NAMES,
                    CONF_THRESH, NMS_IOU, PRE_NMS_TOPK, MAX_DETECTIONS,
                    DEFAULT_WEIGHTS)
from custom_detector.checkpoint import (
    infer_model_options,
    load_checkpoint_metadata,
    load_detector_state_dict,
)
from custom_detector.inference import decode_predictions_v2
from custom_detector.model import FruitDetectorV2


def verify(weights_path=DEFAULT_WEIGHTS, image_dir=VAL_IMG_DIR, limit=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = load_detector_state_dict(weights_path, map_location=device)
    metadata = load_checkpoint_metadata(weights_path, map_location=device)
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

    print("Loaded model weights.")
    print(f"Checkpoint epoch: {metadata.get('epoch', '?')}")
    print(f"Best mAP50: {metadata.get('best_map50', '?')}")
    print(f"Device: {device}")
    print()

    image_dir = Path(image_dir).expanduser()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Validation image directory not found: {image_dir}")

    val_images = sorted([
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    ])[:limit]

    print(f"Running inference on {len(val_images)} validation images...")
    print("=" * 70)

    for img_path in val_images:
        _verify_image(model, img_path, device)

    print("\n" + "=" * 70)
    print("Verification complete.")


def _verify_image(model, img_path, device):
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
        tensor_flipped = torch.flip(tensor, dims=[3])
        cls_pred_f, box_ltrb_f, _, _, _ = model(tensor_flipped)
        cls_pred_f, box_ltrb_f = _unflip_tta_predictions(cls_pred_f, box_ltrb_f)
        cls_pred = (cls_pred + cls_pred_f) / 2.0
        box_ltrb = (box_ltrb + box_ltrb_f) / 2.0

    boxes, labels, scores = decode_predictions_v2(
        cls_pred[0], box_ltrb[0], anchor_points, stride_tensor,
        conf_thresh=CONF_THRESH, nms_iou=NMS_IOU,
        pre_nms_topk=PRE_NMS_TOPK, max_detections=MAX_DETECTIONS,
        img_size=IMG_SIZE,
    )

    print(f"\n{img_path.name} ({orig_w}x{orig_h}) -> {len(boxes)} detections:")
    if len(boxes) == 0:
        print("  (no detections above confidence threshold)")
        return

    for i in range(len(boxes)):
        cls_idx = labels[i].item()
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
        b = boxes[i].clone()
        b[0] = b[0] * orig_w / IMG_SIZE
        b[1] = b[1] * orig_h / IMG_SIZE
        b[2] = b[2] * orig_w / IMG_SIZE
        b[3] = b[3] * orig_h / IMG_SIZE
        print(f"  [{cls_name:>12}] conf={scores[i]:.4f}  box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a small validation inference sanity check")
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument('--image-dir', type=str, default=VAL_IMG_DIR)
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    verify(args.weights, args.image_dir, args.limit)
