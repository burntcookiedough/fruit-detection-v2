import argparse
import torch

from custom_detector.model import FruitDetectorV2
from config import (NUM_CLASSES, IMG_SIZE, BACKBONE_NAME,
                    NECK_CHANNELS, REG_MAX, STRIDES, DEFAULT_ONNX_OUTPUT,
                    DEFAULT_WEIGHTS)
from custom_detector.checkpoint import infer_model_options, load_detector_state_dict

def export_onnx(weights_path, output_path, img_size=IMG_SIZE):
    print(f"Loading weights from {weights_path}...")
    state_dict = load_detector_state_dict(weights_path, map_location='cpu')
    model_options = infer_model_options(state_dict)

    model = FruitDetectorV2(
        num_classes=NUM_CLASSES, img_size=img_size,
        backbone_name=BACKBONE_NAME, pretrained=False,
        neck_channels=NECK_CHANNELS, reg_max=REG_MAX, strides=STRIDES,
        num_head_convs=1,
        **model_options,
    )

    model.load_state_dict(state_dict, strict=True)
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
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument('--out', type=str, default=DEFAULT_ONNX_OUTPUT)
    parser.add_argument('--size', type=int, default=IMG_SIZE)
    args = parser.parse_args()
    export_onnx(args.weights, args.out, args.size)
