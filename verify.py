"""Quick verification: load best_map50.pt and run inference on 5 validation images."""
import os, sys, torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
from config import (IMG_SIZE, NUM_CLASSES, BACKBONE_NAME, NECK_CHANNELS,
                        REG_MAX, STRIDES, WEIGHTS_DIR, VAL_IMG_DIR, CLASS_NAMES,
                        CONF_THRESH, NMS_IOU, PRE_NMS_TOPK, MAX_DETECTIONS)
from custom_detector.model import FruitDetectorV2
from custom_detector.inference import decode_predictions_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = FruitDetectorV2(
    num_classes=NUM_CLASSES, img_size=IMG_SIZE,
    backbone_name=BACKBONE_NAME, pretrained=False,
    neck_channels=NECK_CHANNELS, reg_max=REG_MAX, strides=STRIDES,
    num_head_convs=1
).to(device)

ckpt_path = os.path.join(WEIGHTS_DIR, 'best_map50.pt')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

# Try EMA weights first, fall back to model weights
if 'ema_state_dict' in ckpt:
    ema_sd = ckpt['ema_state_dict']
    # EMA state dict may have 'ema_model.' prefix or 'shadow' key
    if 'shadow' in ema_sd:
        model.load_state_dict(ema_sd['shadow'])
        print("Loaded EMA shadow weights.")
    else:
        # Try direct load
        try:
            model.load_state_dict(ema_sd)
            print("Loaded EMA weights directly.")
        except RuntimeError:
            model.load_state_dict(ckpt['model_state_dict'])
            print("EMA format mismatch, loaded regular model weights.")
else:
    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded model weights (no EMA found).")

model.eval()
print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
print(f"Best mAP50: {ckpt.get('best_map50', '?')}")
print(f"Device: {device}")
print()

# Get 5 validation images
val_images = sorted([f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:5]

print(f"Running inference on {len(val_images)} validation images...")
print("=" * 70)

for fname in val_images:
    img_path = os.path.join(VAL_IMG_DIR, fname)
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        cls_pred, box_ltrb, box_raw, anchor_points, stride_tensor = model(tensor)

    boxes, labels, scores = decode_predictions_v2(
        cls_pred[0], box_ltrb[0], anchor_points, stride_tensor,
        conf_thresh=CONF_THRESH, nms_iou=NMS_IOU,
        pre_nms_topk=PRE_NMS_TOPK, max_detections=MAX_DETECTIONS,
        img_size=IMG_SIZE,
    )

    print(f"\n{fname} ({orig_w}x{orig_h}) -> {len(boxes)} detections:")
    if len(boxes) == 0:
        print("  (no detections above confidence threshold)")
    else:
        for i in range(len(boxes)):
            cls_name = CLASS_NAMES[labels[i].item()] if labels[i].item() < len(CLASS_NAMES) else f"cls_{labels[i].item()}"
            b = boxes[i]
            print(f"  [{cls_name:>12}] conf={scores[i]:.4f}  box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

print("\n" + "=" * 70)
print("Verification complete.")
