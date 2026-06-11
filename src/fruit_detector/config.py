"""Runtime configuration for the anchor-free fruit detector.

Every setting can be overridden with an environment variable. A local ``.env``
file is loaded automatically when present, keeping machine-specific paths out
of source code while preserving simple defaults for development.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path = ENV_FILE) -> None:
    """Load simple KEY=VALUE pairs without adding a python-dotenv dependency."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got: {value!r}")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _env_int_list(name: str, default: Iterable[int]) -> list[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _env_str_list(name: str, default: Iterable[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _path(name: str, default: Path) -> str:
    value = os.getenv(name)
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


_load_dotenv()

# --- Dataset paths ---
DATA_DIR: str = _path("FRUIT_DATA_DIR", PROJECT_ROOT / "dataset")
TRAIN_IMG_DIR: str = _path("FRUIT_TRAIN_IMG_DIR", Path(DATA_DIR) / "train" / "images")
TRAIN_LBL_DIR: str = _path("FRUIT_TRAIN_LBL_DIR", Path(DATA_DIR) / "train" / "labels")
VAL_IMG_DIR: str = _path("FRUIT_VAL_IMG_DIR", Path(DATA_DIR) / "valid" / "images")
VAL_LBL_DIR: str = _path("FRUIT_VAL_LBL_DIR", Path(DATA_DIR) / "valid" / "labels")
TEST_IMG_DIR: str = _path("FRUIT_TEST_IMG_DIR", Path(DATA_DIR) / "test" / "images")
TEST_LBL_DIR: str = _path("FRUIT_TEST_LBL_DIR", Path(DATA_DIR) / "test" / "labels")

# --- Output and artifact paths ---
RUNS_DIR: str = _path("FRUIT_RUNS_DIR", PROJECT_ROOT / "runs" / "fruit_v2")
WEIGHTS_DIR: str = _path("FRUIT_WEIGHTS_DIR", Path(RUNS_DIR) / "weights")
CACHE_DIR: str = _path("FRUIT_CACHE_DIR", PROJECT_ROOT / "cache")
DEFAULT_WEIGHTS: str = _path("FRUIT_WEIGHTS", Path(WEIGHTS_DIR) / "best_map50.pt")
DEFAULT_INPUT_IMAGE: str = _path("FRUIT_INPUT_IMAGE", PROJECT_ROOT / "image.png")
DEFAULT_OUTPUT_IMAGE: str = _path("FRUIT_OUTPUT_IMAGE", PROJECT_ROOT / "output_inference.png")
DEFAULT_ONNX_OUTPUT: str = _path("FRUIT_ONNX_OUTPUT", PROJECT_ROOT / "fruit_detector_v2.onnx")
LABEL_FONT_PATH: str = _env("FRUIT_LABEL_FONT", "")
DEFAULT_CAMERA_ID: int = _env_int("FRUIT_CAMERA_ID", 0)

# --- Model architecture ---
BACKBONE_NAME: str = _env("FRUIT_BACKBONE_NAME", "convnext_femto.d1_in1k")
PRETRAINED: bool = _env_bool("FRUIT_PRETRAINED", True)
NECK_CHANNELS: int = _env_int("FRUIT_NECK_CHANNELS", 96)
REG_MAX: int = _env_int("FRUIT_REG_MAX", 8)
STRIDES: list[int] = _env_int_list("FRUIT_STRIDES", [8, 16, 32])

# --- Image / data ---
IMG_SIZE: int = _env_int("FRUIT_IMG_SIZE", 352)
NUM_CLASSES: int = _env_int("FRUIT_NUM_CLASSES", 8)
BATCH_SIZE: int = _env_int("FRUIT_BATCH_SIZE", 48)

# --- Training ---
NUM_EPOCHS: int = _env_int("FRUIT_NUM_EPOCHS", 40)
LR_BACKBONE: float = _env_float("FRUIT_LR_BACKBONE", 1e-3)
LR_HEAD: float = _env_float("FRUIT_LR_HEAD", 5e-3)
WEIGHT_DECAY: float = _env_float("FRUIT_WEIGHT_DECAY", 5e-3)
GRAD_CLIP: float = _env_float("FRUIT_GRAD_CLIP", 1.0)
FREEZE_BACKBONE_EPOCHS: int = _env_int("FRUIT_FREEZE_BACKBONE_EPOCHS", 2)

# --- Loss weights ---
CLS_WEIGHT: float = _env_float("FRUIT_CLS_WEIGHT", 1.0)
BOX_WEIGHT: float = _env_float("FRUIT_BOX_WEIGHT", 2.5)
DFL_WEIGHT: float = _env_float("FRUIT_DFL_WEIGHT", 0.5)
TAL_TOPK: int = _env_int("FRUIT_TAL_TOPK", 10)

# --- Early stopping ---
PATIENCE: int = _env_int("FRUIT_PATIENCE", 20)
VAL_EVERY: int = _env_int("FRUIT_VAL_EVERY", 5)

# --- Post-processing ---
CONF_THRESH: float = _env_float("FRUIT_CONF_THRESH", 0.05)
NMS_IOU: float = _env_float("FRUIT_NMS_IOU", 0.45)
PRE_NMS_TOPK: int = _env_int("FRUIT_PRE_NMS_TOPK", 1000)
MAX_DETECTIONS: int = _env_int("FRUIT_MAX_DETECTIONS", 100)

# --- Augmentation control ---
MOSAIC_PROB: float = _env_float("FRUIT_MOSAIC_PROB", 0.5)
MIXUP_PROB: float = _env_float("FRUIT_MIXUP_PROB", 0.15)
COPY_PASTE_PROB: float = _env_float("FRUIT_COPY_PASTE_PROB", 0.15)
MOSAIC_OFF_EPOCHS: int = _env_int("FRUIT_MOSAIC_OFF_EPOCHS", 10)

# --- DataLoader ---
NUM_WORKERS: int = _env_int("FRUIT_NUM_WORKERS", min(8, os.cpu_count() or 1))
PREFETCH_FACTOR: int = _env_int("FRUIT_PREFETCH_FACTOR", 4)
PERSISTENT_WORKERS: bool = _env_bool("FRUIT_PERSISTENT_WORKERS", True)

CLASS_NAMES: list[str] = _env_str_list(
    "FRUIT_CLASS_NAMES",
    [
        "apple",
        "banana",
        "orange",
        "mango",
        "pineapple",
        "watermelon",
        "grapes",
        "pomegranate",
    ],
)
