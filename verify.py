"""Quick verification: load best_map50.pt and run inference on 5 validation images."""
import os
import sys
import torch
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(__file__))
from config import (IMG_SIZE, NUM_CLASSES, BACKBONE_NAME, NECK_CHANNELS,
                        REG_MAX, STRIDES, WEIGHTS_DIR, VAL_IMG_DIR, CLASS_NAMES,
                        CONF_THRESH, NMS_IOU, PRE_NMS_TOPK, MAX_DETECTIONS)
from custom_detector.model import FruitDetectorV2
from custom_detector.inference import decode_predictions_v2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ckpt_path = os.path.join(WEIGHTS_DIR, 'best_map50.pt')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

if 'ema_state_dict' in ckpt:
    ema_sd = ckpt['ema_state_dict']
    sd_to_load = ema_sd.get('model', ema_sd.get('shadow', ema_sd))
    if not isinstance(sd_to_load, dict): 
        sd_to_load = ckpt['model_state_dict']
else:
    sd_to_load = ckpt['model_state_dict']

use_sppf = any('sppf' in k for k in sd_to_load.keys())
use_cem = any('cem' in k for k in sd_to_load.keys())
use_grn = any('.grn.gamma' in k for k in sd_to_load.keys())

# Load model
model = FruitDetectorV2(
    num_classes=NUM_CLASSES, img_size=IMG_SIZE,
    backbone_name=BACKBONE_NAME, pretrained=False,
    neck_channels=NECK_CHANNELS, reg_max=REG_MAX, strides=STRIDES,
    num_head_convs=1,
    use_sppf=use_sppf,
    use_cem=use_cem,
    use_grn=use_grn
).to(device)

model.load_state_dict(sd_to_load, strict=True)
print("Loaded model weights.")

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
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        # Base prediction
        cls_pred, box_ltrb, box_raw, anchor_points, stride_tensor = model(tensor)
        
        # Flip prediction
        tensor_flipped = torch.flip(tensor, dims=[3])
        cls_pred_f, box_ltrb_f, _, _, _ = model(tensor_flipped)
        
        # TTA merging: Reverse spatial layout of flipped predictions
        cls_f_unflipped = []
        box_f_unflipped = []
        start = 0
        for stride in STRIDES:
            fm_size = IMG_SIZE // stride
            num_pts = fm_size * fm_size
            
            # Unflip classification map
            c_chunk = cls_pred_f[:, start:start+num_pts, :]
            c_chunk = c_chunk.view(1, fm_size, fm_size, -1)
            c_chunk = torch.flip(c_chunk, dims=[2])
            cls_f_unflipped.append(c_chunk.view(1, num_pts, -1))
            
            # Unflip box map
            b_chunk = box_ltrb_f[:, start:start+num_pts, :]
            b_chunk = b_chunk.view(1, fm_size, fm_size, 4)
            b_chunk = torch.flip(b_chunk, dims=[2])
            box_f_unflipped.append(b_chunk.view(1, num_pts, 4))
            
            start += num_pts
            
        cls_pred_f = torch.cat(cls_f_unflipped, dim=1)
        box_ltrb_f = torch.cat(box_f_unflipped, dim=1)
        
        # Swap left and right distances in box_ltrb_f
        # box_ltrb_f is [..., 4] representing [left, top, right, bottom]
        # Swapping index 0 (left) and 2 (right) maps it back to original image
        box_ltrb_f = box_ltrb_f[..., [2, 1, 0, 3]]
        
        # Average base and flipped predictions
        cls_pred = (cls_pred + cls_pred_f) / 2.0
        box_ltrb = (box_ltrb + box_ltrb_f) / 2.0
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
            b = boxes[i].clone()
            b[0] = b[0] * orig_w / IMG_SIZE
            b[1] = b[1] * orig_h / IMG_SIZE
            b[2] = b[2] * orig_w / IMG_SIZE
            b[3] = b[3] * orig_h / IMG_SIZE
            print(f"  [{cls_name:>12}] conf={scores[i]:.4f}  box=[{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")

print("\n" + "=" * 70)
print("Verification complete.")
