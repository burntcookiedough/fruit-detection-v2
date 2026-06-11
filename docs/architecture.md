# Architecture

## Overview

**fruit-detector** is an anchor-free object detection system built on PyTorch, designed for real-time fruit identification across 8 classes. The architecture follows a standard FPN-based detection paradigm:

```
Input Image (416×416)
       │
       ▼
┌─────────────┐
│  ConvNeXt   │  ← Pre-trained backbone (timm)
│   Femto     │  → Multi-scale features: C3, C4, C5
└─────┬───────┘
      │
      ▼
┌─────────────┐
│   PANet     │  ← Feature Pyramid with bottom-up path
│   Neck      │  → Fused features: P3, P4, P5
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Decoupled  │  ← Separate classification and regression
│   Heads     │  → cls: [B, N, C], reg: [B, N, 4*reg_max]
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  DFL Decode │  ← Distribution Focal Loss → LTRB distances
│  + NMS      │  → Final detections
└─────────────┘
```

## Components

### Backbone (`model/backbone.py`)
- **ConvNeXt-Femto** from `timm` with GRN (Global Response Normalization)
- Extracts multi-scale features at strides 8, 16, 32
- Early-stage freezing for transfer learning efficiency

### Neck (`model/neck.py`)
- **PANet** (Path Aggregation Network)
- Top-down FPN path + bottom-up path for bidirectional feature fusion
- Optional **SPPF** (Spatial Pyramid Pooling Fast) at the deepest level
- Optional **CEM** (Channel Enhancement Module) per scale

### Heads (`model/heads.py`)
- Decoupled classification and regression branches
- Classification: sigmoid focal loss targets
- Regression: DFL-based distribution over `reg_max` bins → LTRB distances

### Loss (`ops/loss.py`)
- **Task-Aligned Assignment** (`ops/assigner.py`): dynamic positive sample selection
- **CIoU Loss**: complete IoU for box regression
- **Distribution Focal Loss**: discrete probability distribution over box coordinates
- **Focal Loss**: class-imbalance aware classification

### Data Pipeline (`data/dataset.py`)
- YOLO-format label loading with in-memory caching
- Augmentations: Mosaic, MixUp, Copy-Paste (minority-aware), HSV jitter, cutout
- Disk cache for resized images (`.npy` format)

### Training Engine (`engine/trainer.py`)
- Dynamic multi-scale training (352–512px)
- Gradient accumulation (3 steps)
- Mixed-precision (AMP) with GradScaler
- Differential learning rates (backbone vs. neck+heads)
- EMA weight averaging
- OneCycleLR scheduler
- Graceful signal handling (SIGINT/SIGTERM)
- Atomic checkpoint writes

## Detection Classes

| Index | Class |
|-------|-------|
| 0 | Apple     |
| 1 | Banana    |
| 2 | Orange    |
| 3 | Mango     |
| 4 | Pineapple |
| 5 | Watermelon|
| 6 | Grapes    |
| 7 | Pomegranate |

## CLI Commands

```
fruit-detect train     # Train the detector
fruit-detect infer     # Single-image inference
fruit-detect webcam    # Real-time webcam demo
fruit-detect verify    # Validation sanity check
fruit-detect export    # ONNX export
fruit-detect analyze   # Dataset class distribution
```
