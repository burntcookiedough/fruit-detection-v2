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

## Quick Start

### Installation

```bash
# From source (recommended for development)
git clone https://github.com/anshu/fruit-detection-v2.git
cd fruit-detection-v2
pip install -e ".[dev]"

# Or just the core package
pip install -e .
```

### Inference

```bash
# Single image
fruit-detect infer --image photo.jpg --weights best_map50.pt --out result.png

# Webcam demo
fruit-detect webcam --weights best_map50.pt

# Verification on validation set
fruit-detect verify --weights best_map50.pt --limit 10
```

### Training

```bash
# Full training
fruit-detect train --epochs 100 --batch-size 16

# Resume from checkpoint
fruit-detect train --resume runs/weights/last.pt

# Build image cache first (speeds up I/O)
fruit-detect train --build-cache --cache-images --epochs 100
```

### Export

```bash
fruit-detect export --weights best_map50.pt --out model.onnx
```

## Project Structure

```
fruit-detection-v2/
├── src/fruit_detector/      # Main package
│   ├── config.py            # Environment-driven configuration
│   ├── model/               # Backbone, neck, heads, detector, EMA
│   ├── ops/                 # Anchor points, loss, assignment, inference
│   ├── data/                # Dataset and augmentation pipeline
│   ├── engine/              # Training loop and validation
│   ├── utils/               # Checkpoint I/O, TTA, visualization
│   └── cli/                 # CLI entry points
├── tests/                   # Pytest test suite
├── scripts/                 # Runner scripts (sh, ps1, bat)
├── docker/                  # Dockerfile and docker-compose
├── docs/                    # Architecture documentation
├── pyproject.toml           # Build config, tool settings
└── .pre-commit-config.yaml  # Pre-commit hooks
```

## Configuration

All configuration is environment-driven via `.env` file (see [.env.example](.env.example)):

```bash
cp .env.example .env
# Edit paths to your dataset
```

Key settings:
| Variable | Default | Description |
|----------|---------|-------------|
| `FRUIT_DATA_DIR` | `./dataset` | Root dataset directory |
| `FRUIT_IMG_SIZE` | `416` | Input image size |
| `FRUIT_BATCH_SIZE` | `12` | Training batch size |
| `FRUIT_NUM_EPOCHS` | `100` | Training epochs |
| `FRUIT_BACKBONE` | `convnext_femto.d1_in1k` | Backbone model |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint and format
ruff check src/ --fix
ruff format src/

# Type check
mypy src/fruit_detector/

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for a detailed architecture overview including the detection pipeline, training strategy, and component descriptions.

## License

[MIT](LICENSE)
