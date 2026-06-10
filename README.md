# Fruit Detection V2

Anchor-free fruit detection system built with PyTorch. The model combines a
ConvNeXt Femto backbone, SPPF context pooling, PANet multi-scale fusion,
context enhancement modules, decoupled detection heads, Distribution Focal
Loss decoding, Task-Aligned Assignment, CIoU loss, and ONNX export support.

## Features

- Anchor-free dense detection with anchor point decoding.
- ConvNeXt Femto backbone through `timm`.
- PANet neck for top-down and bottom-up multi-scale feature fusion.
- Decoupled classification and box regression heads.
- DFL-based bounding box distance prediction.
- Task-Aligned Assignment for prediction-aware positive matching.
- CIoU, DFL, and focal classification losses.
- Mosaic, MixUp, Copy-Paste, flip, blur, color, contrast, and cutout augmentation.
- Image inference, webcam inference, validation smoke test, training, and ONNX export.
- Cross-platform runners for Linux/macOS shells and Windows.

## Project Structure

```text
fruit-detection-v2/
  config.py                         Environment-driven configuration
  train.py                          Training loop and checkpoint management
  run_inference.py                  Image inference and annotation
  webcam_inference.py               Real-time webcam inference
  verify.py                         Validation-image smoke test
  export.py                         ONNX export
  analyze_dataset.py                Dataset class distribution report
  run.sh                            Linux/macOS one-command runner
  run.ps1                           Windows PowerShell runner
  run.bat                           Windows Command Prompt wrapper
  custom_detector/
    anchor_points.py                Anchor point generation and box decoding
    assigner.py                     Task-Aligned Assigner
    backbone.py                     ConvNeXt backbone wrapper
    checkpoint.py                   Shared checkpoint loading helpers
    dataset.py                      Dataset, caching, and augmentation
    inference.py                    Prediction decoding and NMS
    loss.py                         CIoU, DFL, focal, and composite loss
    model.py                        Detector, SPPF, CEM, DFL, heads
    neck.py                         PANet neck
```

## Installation

Use Python 3.10 or newer. A virtual environment is recommended.

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### CUDA Note

The default `requirements.txt` installs standard PyTorch packages from PyPI.
For a specific CUDA build, install PyTorch and TorchVision from the official
PyTorch index before installing the rest of the requirements. Example:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

## Environment Configuration

All configurable values are controlled through environment variables. The app
also loads a local `.env` file automatically.

```bash
cp .env.example .env
```

Edit `.env` for your machine. The most common variables are:

| Variable | Purpose | Default |
| --- | --- | --- |
| `FRUIT_RUN_MODE` | Runner mode: `image`, `webcam`, `train`, `verify`, `export` | `image` |
| `FRUIT_DATA_DIR` | Dataset root with `train`, `valid`, and `test` splits | `dataset` |
| `FRUIT_WEIGHTS` | Checkpoint path used by inference/export/verify | `runs/fruit_v2/weights/best_map50.pt` |
| `FRUIT_INPUT_IMAGE` | Input image for image inference | `image.png` |
| `FRUIT_OUTPUT_IMAGE` | Annotated image output path | `output_inference.png` |
| `FRUIT_CONF_THRESH` | Inference confidence threshold | `0.05` |
| `FRUIT_NMS_IOU` | NMS IoU threshold | `0.45` |
| `FRUIT_CAMERA_ID` | Webcam index for webcam inference | `0` |
| `FRUIT_IMG_SIZE` | Square model input size | `352` |
| `FRUIT_BACKBONE_NAME` | `timm` backbone identifier | `convnext_femto.d1_in1k` |

Dataset directories default to:

```text
dataset/
  train/images
  train/labels
  valid/images
  valid/labels
  test/images
  test/labels
```

Labels use YOLO format:

```text
class_id center_x center_y width height
```

Coordinates should be normalized to `[0, 1]`.

## One-Command Usage

The runners validate Python and core dependencies, set `PYTHONPATH`, and launch
the selected mode.

### Linux/macOS

```bash
chmod +x run.sh
./run.sh
```

### Windows PowerShell

```powershell
.\run.ps1
```

### Windows Command Prompt

```bat
run.bat
```

## Execution Examples

### Image Inference

```bash
FRUIT_RUN_MODE=image ./run.sh --image image.png --out output_inference.png --conf 0.20
```

Windows PowerShell:

```powershell
$env:FRUIT_RUN_MODE = "image"
.\run.ps1 --image image.png --out output_inference.png --conf 0.20
```

### Webcam Inference

```bash
FRUIT_RUN_MODE=webcam ./run.sh --cam 0 --conf 0.20
```

Windows PowerShell:

```powershell
$env:FRUIT_RUN_MODE = "webcam"
.\run.ps1 --cam 0 --conf 0.20
```

### Training

```bash
FRUIT_RUN_MODE=train ./run.sh --epochs 40
```

Windows PowerShell:

```powershell
$env:FRUIT_RUN_MODE = "train"
.\run.ps1 --epochs 40
```

### Validation Smoke Test

```bash
FRUIT_RUN_MODE=verify ./run.sh --limit 5
```

### ONNX Export

```bash
FRUIT_RUN_MODE=export ./run.sh --weights runs/fruit_v2/weights/best_map50.pt --out fruit_detector_v2.onnx
```

## Model Artifacts

Inference expects a PyTorch checkpoint containing either `model_state_dict` or
`ema_state_dict`. By default the repository looks for:

```text
runs/fruit_v2/weights/best_map50.pt
```

Set `FRUIT_WEIGHTS` or pass `--weights` to use another checkpoint.

## Troubleshooting

### Python was not found

Install Python 3.10+ and ensure it is on `PATH`, or set `PYTHON_BIN`.

Linux/macOS:

```bash
PYTHON_BIN=/path/to/python ./run.sh
```

Windows PowerShell:

```powershell
$env:PYTHON_BIN = "path\to\python.exe"
.\run.ps1
```

### Missing Python packages

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

If you need a CUDA-specific PyTorch build, install PyTorch first from the
official PyTorch index, then install the remaining requirements.

### Weights file not found

Set `FRUIT_WEIGHTS` in `.env` or pass `--weights`:

```bash
./run.sh --weights /path/to/best_map50.pt
```

### Dataset directory not found

Set `FRUIT_DATA_DIR` in `.env`, or set the split-specific variables such as
`FRUIT_TRAIN_IMG_DIR` and `FRUIT_VAL_IMG_DIR`.

### Webcam does not open

Check that OpenCV is installed, the camera is connected, and the camera index
is correct:

```bash
FRUIT_RUN_MODE=webcam FRUIT_CAMERA_ID=1 ./run.sh
```

### ONNX export fails

Ensure `onnx` is installed and the checkpoint matches the configured
architecture variables such as `FRUIT_BACKBONE_NAME`, `FRUIT_NECK_CHANNELS`,
`FRUIT_REG_MAX`, and `FRUIT_STRIDES`.

## Development Notes

- Keep machine-specific paths in `.env`, not source files.
- Keep generated artifacts under ignored folders such as `runs/`, `cache/`, and
  `output/`.
- Use `verify.py` for a quick checkpoint smoke test before sharing a model.
- Use `analyze_dataset.py` to inspect per-class label balance before training.
