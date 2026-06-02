"""All hyperparameters for the anchor-free fruit detector."""
import os

# --- Dataset paths ---
# We share the dataset with the V1 model by default, but allow local override
local_dataset = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
shared_dataset = r"F:\Fruit Detection Model\dataset_v4_balanced"
DATA_DIR = local_dataset if os.path.exists(local_dataset) else shared_dataset

TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATA_DIR, "train", "labels")
VAL_IMG_DIR = os.path.join(DATA_DIR, "valid", "images")
VAL_LBL_DIR = os.path.join(DATA_DIR, "valid", "labels")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test", "images")
TEST_LBL_DIR = os.path.join(DATA_DIR, "test", "labels")

# --- Output directory ---
RUNS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "fruit_v2")
WEIGHTS_DIR = os.path.join(RUNS_DIR, "weights")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# --- Model architecture ---
BACKBONE_NAME = 'convnext_femto.d1_in1k'
PRETRAINED = True
NECK_CHANNELS = 96
REG_MAX = 8
STRIDES = [8, 16, 32]

# --- Image / data ---
IMG_SIZE = 352
NUM_CLASSES = 8
BATCH_SIZE = 48                # Maxed for 6 GB VRAM (measured: 5.3 GB peak)

# --- Training ---
NUM_EPOCHS = 40
LR_BACKBONE = 1e-3             # Stable base LR for AdamW
LR_HEAD = 5e-3                 # Stable base LR for AdamW
WEIGHT_DECAY = 5e-3
GRAD_CLIP = 1.0
FREEZE_BACKBONE_EPOCHS = 2

# --- Loss weights ---
CLS_WEIGHT = 1.0
BOX_WEIGHT = 2.5
DFL_WEIGHT = 0.5
TAL_TOPK = 10

# --- Early stopping ---
PATIENCE = 20
VAL_EVERY = 5                  # default; overridden by smart schedule in train.py

# --- Post-processing ---
CONF_THRESH = 0.05
NMS_IOU = 0.45
PRE_NMS_TOPK = 1000
MAX_DETECTIONS = 100

# --- Augmentation control ---
MOSAIC_PROB = 0.5
MIXUP_PROB = 0.15
COPY_PASTE_PROB = 0.15
MOSAIC_OFF_EPOCHS = 10

# --- DataLoader (maxed for 8-core/16-thread CPU) ---
NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PERSISTENT_WORKERS = True

CLASS_NAMES = [
    "apple", "banana", "orange", "mango",
    "pineapple", "watermelon", "grapes", "pomegranate",
]
