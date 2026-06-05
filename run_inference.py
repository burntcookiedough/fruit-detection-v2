"""Run inference on an image and save the annotated results."""
import os
import sys
import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

# Add root directory to path to import config and custom_detector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (IMG_SIZE, NUM_CLASSES, BACKBONE_NAME, NECK_CHANNELS,
                    REG_MAX, STRIDES, WEIGHTS_DIR, CLASS_NAMES)
from custom_detector.model import FruitDetectorV2
from custom_detector.inference import decode_predictions_v2

def run_inference(image_path, weights_path, output_path, conf_thresh=0.25, nms_iou=0.45):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load weights
    print(f"Loading weights from {weights_path}...")
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'ema_state_dict' in ckpt:
        ema_sd = ckpt['ema_state_dict']
        sd_to_load = ema_sd.get('model', ema_sd.get('shadow', ema_sd))
        if not isinstance(sd_to_load, dict): 
            sd_to_load = ckpt['model_state_dict']
    else:
        sd_to_load = ckpt['model_state_dict']
        
    # Fix for CEM linear to conv2d conversion
    for k in list(sd_to_load.keys()):
        if 'cem.' in k and 'fc.' in k and 'weight' in k:
            if sd_to_load[k].dim() == 2:
                sd_to_load[k] = sd_to_load[k].unsqueeze(-1).unsqueeze(-1)
                
    use_sppf = any('sppf' in k for k in sd_to_load.keys())
    use_cem = any('cem' in k for k in sd_to_load.keys())
    use_grn = any('.grn.gamma' in k for k in sd_to_load.keys())
    
    # Build model
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
    model.eval()
    print("Model loaded successfully.")
    
    # Load and preprocess image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original image size: {orig_w}x{orig_h}")
    
    img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    tensor = TF.to_tensor(img_resized)
    tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    tensor = tensor.unsqueeze(0).to(device)
    
    # Perform TTA (Test Time Augmentation) inference matching verify.py
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
        
        # Flip prediction
        tensor_flipped = torch.flip(tensor, dims=[3])
        cls_pred_f, box_ltrb_f, _, _, _ = model(tensor_flipped)
        
        # TTA merging
        cls_f_unflipped = []
        box_f_unflipped = []
        start = 0
        for stride in STRIDES:
            fm_size = IMG_SIZE // stride
            num_pts = fm_size * fm_size
            
            c_chunk = cls_pred_f[:, start:start+num_pts, :]
            c_chunk = c_chunk.view(1, fm_size, fm_size, -1)
            c_chunk = torch.flip(c_chunk, dims=[2])
            cls_f_unflipped.append(c_chunk.view(1, num_pts, -1))
            
            b_chunk = box_ltrb_f[:, start:start+num_pts, :]
            b_chunk = b_chunk.view(1, fm_size, fm_size, 4)
            b_chunk = torch.flip(b_chunk, dims=[2])
            box_f_unflipped.append(b_chunk.view(1, num_pts, 4))
            
            start += num_pts
            
        cls_pred_f = torch.cat(cls_f_unflipped, dim=1)
        box_ltrb_f = torch.cat(box_f_unflipped, dim=1)
        box_ltrb_f = box_ltrb_f[..., [2, 1, 0, 3]]
        
        cls_pred = (cls_pred + cls_pred_f) / 2.0
        box_ltrb = (box_ltrb + box_ltrb_f) / 2.0
        
    boxes, labels, scores = decode_predictions_v2(
        cls_pred[0], box_ltrb[0], anchor_points, stride_tensor,
        conf_thresh=conf_thresh, nms_iou=nms_iou,
        pre_nms_topk=1000, max_detections=100,
        img_size=IMG_SIZE
    )
    
    print(f"Found {len(boxes)} detections above confidence {conf_thresh}:")
    
    # Class colors: Apple (red), Banana (yellow), Orange (orange), Mango (gold/amber),
    # Pineapple (brown/yellow-green), Watermelon (green), Grapes (purple), Pomegranate (dark red)
    colors = [
        (255, 59, 48),    # apple (red)
        (255, 204, 0),   # banana (yellow)
        (255, 149, 0),   # orange (orange)
        (255, 214, 10),  # mango (amber)
        (175, 82, 222),  # pineapple (purple/violet)
        (52, 199, 89),   # watermelon (green)
        (90, 200, 250),  # grapes (light blue/purple)
        (164, 28, 48)    # pomegranate (crimson)
    ]
    
    # Draw boxes
    draw = ImageDraw.Draw(img)
    
    # Try to load a nicer font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", size=max(12, int(orig_h * 0.02)))
    except IOError:
        font = ImageFont.load_default()
        
    for i in range(len(boxes)):
        b = boxes[i].clone()
        # Scale back to original image size
        b[0] = b[0] * orig_w / IMG_SIZE
        b[1] = b[1] * orig_h / IMG_SIZE
        b[2] = b[2] * orig_w / IMG_SIZE
        b[3] = b[3] * orig_h / IMG_SIZE
        
        cls_idx = labels[i].item()
        cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
        score = scores[i].item()
        
        color = colors[cls_idx % len(colors)]
        
        # Bounding box coordinates
        x1, y1, x2, y2 = b[0].item(), b[1].item(), b[2].item(), b[3].item()
        print(f"  - {cls_name} ({score:.2%}): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        
        # Draw thick box
        thickness = max(2, int(orig_h * 0.005))
        for t in range(thickness):
            draw.rectangle([x1 + t, y1 + t, x2 - t, y2 - t], outline=color)
            
        # Draw label background and text
        label_text = f"{cls_name} {score:.0%}"
        
        # Get text size
        if hasattr(draw, "textbbox"):
            text_w, text_h = draw.textbbox((0, 0), label_text, font=font)[2:4]
        else:
            text_w, text_h = draw.textsize(label_text, font=font)
            
        # Draw label box background
        draw.rectangle([x1, y1 - text_h - 4, x1 + text_w + 6, y1], fill=color)
        # Draw label text
        draw.text((x1 + 3, y1 - text_h - 2), label_text, fill=(255, 255, 255), font=font)
        
    # Save the result
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path)
    print(f"Saved annotated image to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='image.png', help='Path to input image')
    parser.add_argument('--weights', type=str, default=os.path.join(WEIGHTS_DIR, 'best_map50.pt'), help='Path to checkpoint')
    parser.add_argument('--out', type=str, default='output_inference.png', help='Path to save annotated image')
    parser.add_argument('--conf', type=float, default=0.20, help='Confidence threshold')
    args = parser.parse_args()
    
    run_inference(args.image, args.weights, args.out, args.conf)
