"""Real-time webcam inference using the custom fruit detector model."""
import os
import sys
import time
import argparse
import cv2
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Add root directory to path to import config and custom_detector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (IMG_SIZE, NUM_CLASSES, BACKBONE_NAME, NECK_CHANNELS,
                    REG_MAX, STRIDES, WEIGHTS_DIR, CLASS_NAMES)
from custom_detector.model import FruitDetectorV2
from custom_detector.inference import decode_predictions_v2

def run_webcam(weights_path, conf_thresh=0.20, nms_iou=0.45, camera_id=0, use_tta=False):
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
    
    # Open webcam
    print(f"Opening webcam (ID: {camera_id})...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam with ID {camera_id}.")
        return
        
    # Class colors in BGR format for cv2 drawing:
    # Apple (red), Banana (yellow), Orange (orange), Mango (gold/amber),
    # Pineapple (purple), Watermelon (green), Grapes (light blue), Pomegranate (crimson)
    bgr_colors = [
        (48, 59, 255),    # apple (red-ish)
        (0, 204, 255),    # banana (yellow-ish)
        (0, 149, 255),    # orange (orange-ish)
        (10, 214, 255),   # mango (gold/amber)
        (222, 82, 175),   # pineapple (purple/violet)
        (89, 199, 52),    # watermelon (green)
        (250, 200, 90),    # grapes (light blue)
        (48, 28, 164)     # pomegranate (crimson)
    ]
    
    print("\nPress 'q' in the window to quit, or 't' to toggle Test Time Augmentation (TTA).")
    print("Use keys '1' through '9' to adjust confidence threshold (e.g. '1' = 0.10, '3' = 0.30).")
    
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
            
        orig_h, orig_w = frame.shape[:2]
        
        # Preprocess frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        tensor = TF.to_tensor(img_resized)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tensor = tensor.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            if use_tta:
                # Run regular inference + flipped inference for test-time augmentation
                cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
                
                tensor_flipped = torch.flip(tensor, dims=[3])
                cls_pred_f, box_ltrb_f, _, _, _ = model(tensor_flipped)
                
                # Merge flipped predictions
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
            else:
                # Single-pass fast inference
                cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(tensor)
                
        # Decode detections
        boxes, labels, scores = decode_predictions_v2(
            cls_pred[0], box_ltrb[0], anchor_points, stride_tensor,
            conf_thresh=conf_thresh, nms_iou=nms_iou,
            pre_nms_topk=1000, max_detections=100,
            img_size=IMG_SIZE
        )
        
        # Draw detections on OpenCV frame
        for i in range(len(boxes)):
            b = boxes[i].clone()
            # Scale coordinates back to original frame size
            x1 = int(b[0].item() * orig_w / IMG_SIZE)
            y1 = int(b[1].item() * orig_h / IMG_SIZE)
            x2 = int(b[2].item() * orig_w / IMG_SIZE)
            y2 = int(b[3].item() * orig_h / IMG_SIZE)
            
            cls_idx = labels[i].item()
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
            score = scores[i].item()
            
            color = bgr_colors[cls_idx % len(bgr_colors)]
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw text label background and text
            label_text = f"{cls_name} {score:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w + 6, y1), color, -1)
            cv2.putText(frame, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Display overlay info (FPS, confidence threshold, TTA status)
        overlay_text = f"FPS: {fps:.1f} | Conf Thresh: {conf_thresh:.2f} | TTA: {'ON' if use_tta else 'OFF'}"
        cv2.putText(frame, overlay_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show window
        cv2.imshow("Fruit Detector v2 Webcam Demo", frame)
        
        # Key handler
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            use_tta = not use_tta
            print(f"Toggled Test Time Augmentation (TTA): {use_tta}")
        elif ord('1') <= key <= ord('9'):
            conf_thresh = (key - ord('0')) / 10.0
            print(f"Confidence threshold updated to: {conf_thresh:.2f}")
            
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam inference stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=os.path.join(WEIGHTS_DIR, 'best_map50.pt'), help='Path to checkpoint')
    parser.add_argument('--conf', type=float, default=0.15, help='Confidence threshold')
    parser.add_argument('--cam', type=int, default=0, help='Webcam ID')
    parser.add_argument('--tta', action='store_true', help='Use Test Time Augmentation (TTA)')
    args = parser.parse_args()
    
    run_webcam(args.weights, conf_thresh=args.conf, camera_id=args.cam, use_tta=args.tta)
