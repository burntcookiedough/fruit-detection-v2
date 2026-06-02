import torch
import argparse
from custom_detector.model import FruitDetectorV2
from config import (NUM_CLASSES, IMG_SIZE, BACKBONE_NAME, 
                    NECK_CHANNELS, REG_MAX, STRIDES)

def export_onnx(weights_path, output_path, img_size=512):
    print(f"Loading weights from {weights_path}...")
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    
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

    model = FruitDetectorV2(
        num_classes=NUM_CLASSES, img_size=img_size,
        backbone_name=BACKBONE_NAME, pretrained=False,
        neck_channels=NECK_CHANNELS, reg_max=REG_MAX, strides=STRIDES,
        num_head_convs=1,
        use_sppf=use_sppf,
        use_cem=use_cem,
        use_grn=use_grn
    )

    model.load_state_dict(sd_to_load, strict=True)
        
    model.eval()

    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    print(f"Exporting to {output_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=17, 
        do_constant_folding=True,
        input_names=['input'], 
        output_names=['cls_pred', 'box_pred_ltrb', 'box_pred_raw', 'anchor_points', 'stride_tensor'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'cls_pred': {0: 'batch_size'}, 
            'box_pred_ltrb': {0: 'batch_size'},
            'box_pred_raw': {0: 'batch_size'}
        }
    )
    print("Export successful!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--out', type=str, default='fruit_detector_v3.onnx')
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args()
    export_onnx(args.weights, args.out, args.size)
