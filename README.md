# 🍎 Fruit Detector v2

> Anchor-free object detection for real-time fruit identification, powered by ConvNeXt + PANet + DFL.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

---

## Features

- **Anchor-free architecture** — ConvNeXt-Femto backbone, PANet neck, decoupled heads
- **8-class detection** — Apple, Banana, Orange, Mango, Pineapple, Watermelon, Grapes, Pomegranate
- **Task-Aligned Assignment** with CIoU + Distribution Focal Loss
- **Production training** — EMA, mixed-precision, gradient accumulation, graceful interrupts
- **Advanced augmentations** — Mosaic, MixUp, Copy-Paste (minority-aware), multi-scale
- **CLI toolkit** — `fruit-detect train | infer | webcam | verify | export | analyze`
- **ONNX export** for deployment
- **Docker support** — multi-stage build, GPU training, dev containers
- **Cross-platform** — Linux, macOS, Windows (PowerShell + Bash + Batch)

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Configuration](#configuration)
- [CLI Commands](#cli-commands)
  - [Training](#training)
  - [Inference](#inference)
  - [Webcam](#webcam)
  - [Verification](#verification)
  - [ONNX Export](#onnx-export)
  - [Dataset Analysis](#dataset-analysis)
- [Run Scripts](#run-scripts)
- [Docker](#docker)
- [Development](#development)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [License](#license)

---

## Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| CUDA (optional) | 11.8+ (for GPU training) |
| Docker (optional) | 20.10+ (for containerized runs) |
| NVIDIA Container Toolkit (optional) | For GPU inside Docker |

---

## Installation

### From Source (recommended)

```bash
git clone https://github.com/anshu/fruit-detection-v2.git
cd fruit-detection-v2

# Core package only
pip install -e .

# With development tools (ruff, mypy, pytest)
pip install -e ".[dev]"

# With metrics support (torchmetrics)
pip install -e ".[metrics]"

# With webcam support (OpenCV)
pip install -e ".[webcam]"

# Everything
pip install -e ".[dev,metrics,webcam]"
```

### Verify Installation

```bash
fruit-detect --help
```

---

## Dataset Setup

The dataset should follow the YOLO format with this directory layout:

```
dataset/
├── train/
│   ├── images/       # Training images (.jpg, .png)
│   └── labels/       # YOLO labels (.txt) — one per image
├── valid/
│   ├── images/       # Validation images
│   └── labels/       # Validation labels
└── test/
    ├── images/       # Test images
    └── labels/       # Test labels
```

Each label file contains one line per object: `class_id cx cy w h` (normalized 0–1).

**Class mapping:**

| ID | Fruit |
|---|---|
| 0 | Apple |
| 1 | Banana |
| 2 | Orange |
| 3 | Mango |
| 4 | Pineapple |
| 5 | Watermelon |
| 6 | Grapes |
| 7 | Pomegranate |

---

## Configuration

All settings are driven by environment variables. Copy the example and customize:

```bash
cp .env.example .env
# Edit .env with your paths and hyperparameters
```

### Dataset Paths

| Variable | Default | Description |
|---|---|---|
| `FRUIT_DATA_DIR` | `dataset` | Root dataset directory |
| `FRUIT_TRAIN_IMG_DIR` | `dataset/train/images` | Training images |
| `FRUIT_TRAIN_LBL_DIR` | `dataset/train/labels` | Training labels |
| `FRUIT_VAL_IMG_DIR` | `dataset/valid/images` | Validation images |
| `FRUIT_VAL_LBL_DIR` | `dataset/valid/labels` | Validation labels |
| `FRUIT_TEST_IMG_DIR` | `dataset/test/images` | Test images |
| `FRUIT_TEST_LBL_DIR` | `dataset/test/labels` | Test labels |

### Model Architecture

| Variable | Default | Description |
|---|---|---|
| `FRUIT_BACKBONE_NAME` | `convnext_femto.d1_in1k` | Backbone model (from timm) |
| `FRUIT_PRETRAINED` | `true` | Use ImageNet pretrained weights |
| `FRUIT_NECK_CHANNELS` | `96` | PANet neck channel width |
| `FRUIT_REG_MAX` | `8` | DFL regression bins |
| `FRUIT_STRIDES` | `8,16,32` | Detection strides |
| `FRUIT_IMG_SIZE` | `352` | Input image size |
| `FRUIT_NUM_CLASSES` | `8` | Number of fruit classes |

### Training Hyperparameters

| Variable | Default | Description |
|---|---|---|
| `FRUIT_BATCH_SIZE` | `48` | Training batch size |
| `FRUIT_NUM_EPOCHS` | `40` | Total training epochs |
| `FRUIT_LR_BACKBONE` | `0.001` | Backbone learning rate |
| `FRUIT_LR_HEAD` | `0.005` | Head learning rate |
| `FRUIT_WEIGHT_DECAY` | `0.005` | Weight decay |
| `FRUIT_GRAD_CLIP` | `1.0` | Gradient clipping norm |
| `FRUIT_FREEZE_BACKBONE_EPOCHS` | `2` | Epochs to freeze backbone |
| `FRUIT_PATIENCE` | `20` | Early stopping patience |
| `FRUIT_VAL_EVERY` | `5` | Validate every N epochs |

### Loss Weights

| Variable | Default | Description |
|---|---|---|
| `FRUIT_CLS_WEIGHT` | `1.0` | Classification loss weight |
| `FRUIT_BOX_WEIGHT` | `2.5` | Box regression (CIoU) weight |
| `FRUIT_DFL_WEIGHT` | `0.5` | Distribution Focal Loss weight |
| `FRUIT_TAL_TOPK` | `10` | TAL top-K candidates |

### Augmentation

| Variable | Default | Description |
|---|---|---|
| `FRUIT_MOSAIC_PROB` | `0.5` | Mosaic augmentation probability |
| `FRUIT_MIXUP_PROB` | `0.15` | MixUp probability |
| `FRUIT_COPY_PASTE_PROB` | `0.15` | Copy-Paste probability |
| `FRUIT_MOSAIC_OFF_EPOCHS` | `10` | Disable mosaic for final N epochs |

### Inference / Post-Processing

| Variable | Default | Description |
|---|---|---|
| `FRUIT_CONF_THRESH` | `0.05` | Confidence threshold |
| `FRUIT_NMS_IOU` | `0.45` | NMS IoU threshold |
| `FRUIT_PRE_NMS_TOPK` | `1000` | Pre-NMS top-K |
| `FRUIT_MAX_DETECTIONS` | `100` | Max detections per image |
| `FRUIT_CAMERA_ID` | `0` | Webcam device index |

### DataLoader

| Variable | Default | Description |
|---|---|---|
| `FRUIT_NUM_WORKERS` | `8` | DataLoader worker processes |
| `FRUIT_PREFETCH_FACTOR` | `4` | Prefetch batches per worker |
| `FRUIT_PERSISTENT_WORKERS` | `true` | Keep workers alive between epochs |

### Output Paths

| Variable | Default | Description |
|---|---|---|
| `FRUIT_RUNS_DIR` | `runs/fruit_v2` | Training runs directory |
| `FRUIT_WEIGHTS_DIR` | `runs/fruit_v2/weights` | Saved weights directory |
| `FRUIT_WEIGHTS` | `runs/fruit_v2/weights/best_map50.pt` | Best checkpoint path |
| `FRUIT_INPUT_IMAGE` | `image.png` | Default input image |
| `FRUIT_OUTPUT_IMAGE` | `output_inference.png` | Default output image |
| `FRUIT_ONNX_OUTPUT` | `fruit_detector_v2.onnx` | Default ONNX output |

---

## CLI Commands

All commands are accessed via the `fruit-detect` entry point:

```bash
fruit-detect --help
```

### Training

```bash
# Full training run
fruit-detect train --epochs 100

# Custom batch size and epochs
fruit-detect train --epochs 50 --batch-size 16

# Resume from a checkpoint
fruit-detect train --resume runs/fruit_v2/weights/last.pt

# Build image cache first, then train with caching
fruit-detect train --build-cache --cache-images --epochs 100

# Freeze backbone for more epochs
fruit-detect train --freeze-backbone-epochs 5 --epochs 80

# Quick smoke test (2 epochs, limited batches, no validation)
fruit-detect train --epochs 2 --limit-train-batches 5 --skip-val

# Limit validation batches
fruit-detect train --epochs 50 --limit-val-batches 10 --val-every 3

# Adjust DataLoader workers
fruit-detect train --workers 4 --prefetch-factor 2 --no-persistent-workers
```

**All training CLI options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--epochs` | int | 40 | Number of training epochs |
| `--resume` | str | — | Path to checkpoint to resume from |
| `--limit-train-batches` | int | — | Limit training batches per epoch |
| `--limit-val-batches` | int | — | Limit validation batches |
| `--val-every` | int | 5 | Run validation every N epochs |
| `--skip-val` | flag | — | Skip validation entirely |
| `--workers` | int | 8 | DataLoader workers |
| `--prefetch-factor` | int | 4 | Prefetch factor |
| `--persistent-workers` / `--no-persistent-workers` | bool | true | Keep workers alive |
| `--cache-images` | flag | — | Use cached images from RAM |
| `--build-cache` | flag | — | Build .npy image cache on disk |
| `--freeze-backbone-epochs` | int | 2 | Freeze backbone for N epochs |
| `--batch-size` | int | 48 | Batch size |

### Inference

```bash
# Basic inference
fruit-detect infer --image photo.jpg --weights best_map50.pt --out result.png

# With custom thresholds
fruit-detect infer --image photo.jpg --weights best_map50.pt --conf 0.25 --nms-iou 0.5

# Disable Test-Time Augmentation (faster, slightly less accurate)
fruit-detect infer --image photo.jpg --weights best_map50.pt --no-tta

# Adjust detection limits
fruit-detect infer --image photo.jpg --weights best_map50.pt --max-detections 50 --pre-nms-topk 500
```

**All inference CLI options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--image` | str | `image.png` | Input image path |
| `--weights` | str | `runs/.../best_map50.pt` | Model weights path |
| `--out` | str | `output_inference.png` | Output image path |
| `--conf` | float | 0.05 | Confidence threshold |
| `--nms-iou` | float | 0.45 | NMS IoU threshold |
| `--pre-nms-topk` | int | 1000 | Pre-NMS top-K |
| `--max-detections` | int | 100 | Max detections |
| `--no-tta` | flag | — | Disable Test-Time Augmentation |

### Webcam

```bash
# Real-time webcam inference
fruit-detect webcam --weights best_map50.pt

# Specific camera and confidence threshold
fruit-detect webcam --weights best_map50.pt --cam 1 --conf 0.3

# Enable TTA for webcam (slower but more accurate)
fruit-detect webcam --weights best_map50.pt --tta

# Adjust NMS threshold
fruit-detect webcam --weights best_map50.pt --nms-iou 0.5
```

> **Note:** Requires `opencv-python`. Install with: `pip install -e ".[webcam]"`
>
> **Controls:** Press `q` to quit the webcam window.

**All webcam CLI options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--weights` | str | `runs/.../best_map50.pt` | Model weights |
| `--conf` | float | 0.05 | Confidence threshold |
| `--nms-iou` | float | 0.45 | NMS IoU threshold |
| `--cam` | int | 0 | Camera device index |
| `--tta` | flag | — | Enable Test-Time Augmentation |

### Verification

```bash
# Quick sanity check on validation images
fruit-detect verify --weights best_map50.pt

# Check more images
fruit-detect verify --weights best_map50.pt --limit 20

# Use custom image directory
fruit-detect verify --weights best_map50.pt --image-dir dataset/test/images --limit 10
```

**All verify CLI options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--weights` | str | `runs/.../best_map50.pt` | Model weights |
| `--image-dir` | str | `dataset/valid/images` | Image directory to verify |
| `--limit` | int | 5 | Max images to process |

### ONNX Export

```bash
# Export to ONNX
fruit-detect export --weights best_map50.pt --out model.onnx

# Export with custom input size
fruit-detect export --weights best_map50.pt --out model_416.onnx --size 416
```

**All export CLI options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--weights` | str | `runs/.../best_map50.pt` | Model weights |
| `--out` | str | `fruit_detector_v2.onnx` | Output ONNX path |
| `--size` | int | 352 | Input image size for export |

### Dataset Analysis

```bash
# Analyze class distribution across all splits
fruit-detect analyze

# Analyze a specific dataset directory
fruit-detect analyze --data-dir /path/to/dataset
```

**All analyze CLI options:**

| Flag | Type | Default | Description |
|---|---|---|---|
| `--data-dir` | str | `dataset` | Dataset root directory |

---

## Run Scripts

Cross-platform convenience scripts that auto-load `.env` and support `FRUIT_RUN_MODE` dispatch:

### Linux / macOS

```bash
# Pass arguments directly
./scripts/run.sh train --epochs 50
./scripts/run.sh infer --image photo.jpg
./scripts/run.sh --help

# Or set FRUIT_RUN_MODE in .env and just run:
# FRUIT_RUN_MODE=train
./scripts/run.sh
```

### Windows (PowerShell)

```powershell
# Pass arguments directly
.\scripts\run.ps1 train --epochs 50
.\scripts\run.ps1 infer --image photo.jpg
.\scripts\run.ps1 --help

# Or via FRUIT_RUN_MODE dispatch:
.\scripts\run.ps1
```

### Windows (CMD)

```cmd
scripts\run.bat train --epochs 50
scripts\run.bat --help
```

**Auto-dispatch modes** (set `FRUIT_RUN_MODE` in `.env`):

| Mode | Maps to |
|---|---|
| `train` | `fruit-detect train` |
| `image` / `infer` | `fruit-detect infer` |
| `webcam` | `fruit-detect webcam` |
| `verify` | `fruit-detect verify` |
| `export` | `fruit-detect export` |
| `analyze` | `fruit-detect analyze` |

---

## Docker

### Build

```bash
# Production image
docker build -f docker/Dockerfile -t fruit-detector:latest .

# Development image (with dev tools and tests)
docker build -f docker/Dockerfile --target development -t fruit-detector:dev .
```

### Run with Docker

```bash
# Show help
docker run --rm fruit-detector:latest --help

# Train with GPU
docker run --rm --gpus all \
    -v ./dataset:/app/dataset:ro \
    -v ./runs:/app/runs \
    -v ./cache:/app/cache \
    fruit-detector:latest train --epochs 100

# Inference
docker run --rm \
    -v ./dataset:/app/dataset:ro \
    -v ./runs:/app/runs:ro \
    -v ./output:/app/output \
    fruit-detector:latest infer \
        --image /app/dataset/test/images/sample.jpg \
        --weights /app/runs/fruit_v2/weights/best_map50.pt \
        --out /app/output/result.png

# Export ONNX
docker run --rm \
    -v ./runs:/app/runs:ro \
    -v ./output:/app/output \
    fruit-detector:latest export \
        --weights /app/runs/fruit_v2/weights/best_map50.pt \
        --out /app/output/model.onnx

# Interactive dev shell
docker run --rm -it \
    --gpus all \
    -v ./src:/app/src \
    -v ./tests:/app/tests \
    -v ./dataset:/app/dataset:ro \
    fruit-detector:dev
```

### Run with Docker Compose

```bash
# Train
docker compose -f docker/docker-compose.yml up train

# Inference
docker compose -f docker/docker-compose.yml run infer

# Export
docker compose -f docker/docker-compose.yml run export

# Verify
docker compose -f docker/docker-compose.yml run verify

# Analyze dataset
docker compose -f docker/docker-compose.yml run analyze

# Dev shell
docker compose -f docker/docker-compose.yml --profile dev run dev
```

> **Note:** GPU training requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

---

## Development

### Setup

```bash
# Install all dev dependencies
pip install -e ".[dev,metrics,webcam]"

# Install pre-commit hooks
pre-commit install
```

### Testing

```bash
# Run full test suite (24 tests)
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_model.py -v
python -m pytest tests/test_trainer.py -v
python -m pytest tests/test_ops.py -v
python -m pytest tests/test_data.py -v
python -m pytest tests/test_cli.py -v

# Run with coverage
python -m pytest tests/ -v --cov=fruit_detector --cov-report=html
```

### Linting & Formatting

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix lint errors
ruff check src/ tests/ --fix

# Format code
ruff format src/ tests/

# Type checking
mypy src/fruit_detector/

# Run all pre-commit hooks
pre-commit run --all-files
```

---

## Project Structure

```
fruit-detection-v2/
├── src/fruit_detector/           # Main package
│   ├── __init__.py               # Version
│   ├── config.py                 # Environment-driven configuration
│   ├── model/                    # Model architecture
│   │   ├── detector.py           # FruitDetectorV2 (main model)
│   │   ├── backbone.py           # ConvNeXt backbone (via timm)
│   │   ├── neck.py               # PANet feature pyramid
│   │   ├── heads.py              # Decoupled cls/reg heads
│   │   ├── components.py         # DFL, SPPF, CEM modules
│   │   └── ema.py                # Exponential Moving Average
│   ├── ops/                      # Core operations
│   │   ├── anchor_points.py      # Anchor-free grid generation
│   │   ├── loss.py               # CIoU + Focal + DFL losses
│   │   ├── assigner.py           # Task-Aligned Label Assignment
│   │   └── inference.py          # NMS + decoding
│   ├── data/                     # Data pipeline
│   │   └── dataset.py            # YOLO dataset + augmentations
│   ├── engine/                   # Training engine
│   │   └── trainer.py            # Full training loop
│   ├── utils/                    # Utilities
│   │   ├── checkpoint.py         # Checkpoint I/O + model option inference
│   │   ├── tta.py                # Test-Time Augmentation helpers
│   │   └── visualization.py      # Drawing bounding boxes
│   └── cli/                      # CLI entry points
│       ├── __init__.py            # Dispatcher
│       ├── train.py               # fruit-detect train
│       ├── infer.py               # fruit-detect infer
│       ├── webcam.py              # fruit-detect webcam
│       ├── verify.py              # fruit-detect verify
│       ├── export.py              # fruit-detect export
│       └── analyze.py             # fruit-detect analyze
├── tests/                        # Test suite (24 tests)
│   ├── conftest.py               # Shared fixtures
│   ├── test_model.py             # Model forward pass, DFL, SPPF, EMA
│   ├── test_ops.py               # Anchors, decoding, TTA
│   ├── test_data.py              # Dataset loading, augmentation
│   ├── test_cli.py               # CLI integration tests
│   └── test_trainer.py           # Training loop integration
├── scripts/                      # Cross-platform runners
│   ├── run.sh                    # Bash (Linux/macOS)
│   ├── run.ps1                   # PowerShell (Windows)
│   └── run.bat                   # CMD wrapper → run.ps1
├── docker/                       # Container support
│   ├── Dockerfile                # Multi-stage (deps → production → dev)
│   └── docker-compose.yml        # Train/infer/export/verify/analyze/dev
├── docs/                         # Documentation
│   └── architecture.md           # Architecture overview
├── .dockerignore                 # Docker build context filter
├── .env.example                  # Environment variable template
├── .pre-commit-config.yaml       # Pre-commit hooks
├── pyproject.toml                # Build config, ruff, mypy, pytest
├── requirements.txt              # Pip requirements (mirrors pyproject.toml)
└── README.md                     # This file
```

---

## Architecture

```
Input Image → [ConvNeXt Backbone] → [SPPF] → [PANet Neck] → [Decoupled Heads ×3]
                                                                  ├── cls branch → Focal Loss
                                                                  └── reg branch → DFL → CIoU Loss
                                                              [Task-Aligned Assignment]
```

- **Backbone**: ConvNeXt-Femto (from [timm](https://github.com/huggingface/pytorch-image-models)), ImageNet pretrained
- **Neck**: PANet with bidirectional FPN fusion at 3 scales (P3/P4/P5)
- **Heads**: Decoupled classification + regression branches per scale
- **Assignment**: Task-Aligned Assignment (TAL) with alignment metric
- **Losses**: Focal Loss (cls) + CIoU (box) + Distribution Focal Loss (DFL)
- **Training**: EMA, AMP (mixed precision), gradient accumulation (3-step), multi-scale (352–512), cosine LR schedule with warmup

See [docs/architecture.md](docs/architecture.md) for the full architecture deep-dive.

---

## License

[MIT](LICENSE)
