"""Training engine — epoch loop, optimizer, scheduler, checkpointing."""

from __future__ import annotations

import contextlib
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, RandomSampler

from ..config import (
    BACKBONE_NAME,
    BATCH_SIZE,
    BOX_WEIGHT,
    CACHE_DIR,
    CLASS_NAMES,
    CLS_WEIGHT,
    COPY_PASTE_PROB,
    DFL_WEIGHT,
    FREEZE_BACKBONE_EPOCHS,
    GRAD_CLIP,
    IMG_SIZE,
    LR_BACKBONE,
    LR_HEAD,
    MIXUP_PROB,
    MOSAIC_OFF_EPOCHS,
    MOSAIC_PROB,
    NECK_CHANNELS,
    NMS_IOU,
    NUM_CLASSES,
    NUM_EPOCHS,
    NUM_WORKERS,
    PATIENCE,
    PERSISTENT_WORKERS,
    PRE_NMS_TOPK,
    PREFETCH_FACTOR,
    PRETRAINED,
    REG_MAX,
    RUNS_DIR,
    STRIDES,
    TAL_TOPK,
    TRAIN_IMG_DIR,
    TRAIN_LBL_DIR,
    VAL_EVERY,
    VAL_IMG_DIR,
    VAL_LBL_DIR,
    WEIGHT_DECAY,
    WEIGHTS_DIR,
)
from ..data import FruitDataset, collate_fn
from ..model import FruitDetectorV2, ModelEMA
from ..ops import DetectionLossV2
from ..ops.inference import decode_predictions_v2
from ..ops.loss import compute_class_weights_from_label_entries
from ..utils.checkpoint import _normalize_cem_weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def _unpack_batch(
    batch: tuple,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[str] | None]:
    if len(batch) == 4:
        images, boxes_list, labels_list, target_keys = batch
    else:
        images, boxes_list, labels_list = batch
        target_keys = None
    return images, boxes_list, labels_list, target_keys


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: DetectionLossV2,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    max_batches: int | None = None,
    ema: ModelEMA | None = None,
    scaler: torch.GradScaler | None = None,
    scheduler: Any = None,
) -> tuple[float, float, float, float]:
    """Run one epoch of training.

    Returns:
        ``(avg_loss, avg_cls, avg_box, avg_dfl)``
    """
    model.train()
    total_loss = total_cls = total_box = total_dfl = 0.0
    num_batches = 0
    use_amp = scaler is not None
    loader_len = len(loader)

    scales = [352, 384, 416, 448, 480, 512]
    accumulation_steps = 3
    target_sz: int | None = None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, boxes_list, labels_list, _ = _unpack_batch(batch)

        if target_sz is None:
            target_sz = images.shape[-1]

        if batch_idx % 10 == 0:
            target_sz = random.choice(scales)

        if target_sz != images.shape[-1]:
            scale_ratio = target_sz / images.shape[-1]
            images = torch.nn.functional.interpolate(
                images, size=(target_sz, target_sz), mode="bilinear", align_corners=False
            )
            boxes_list = [b * scale_ratio for b in boxes_list]

        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        boxes_list = [b.to(device, non_blocking=True).float() for b in boxes_list]
        labels_list = [lbl.to(device, non_blocking=True) for lbl in labels_list]

        with torch.autocast(device_type="cuda", enabled=use_amp):
            cls_pred, box_ltrb, box_raw, anchor_points, stride_tensor = model(images)
            loss_dict = criterion(
                cls_pred,
                box_ltrb,
                box_raw,
                anchor_points,
                stride_tensor,
                boxes_list,
                labels_list,
            )
            loss = loss_dict["total"]

        loss_scaled = loss / accumulation_steps

        if use_amp and scaler is not None:
            scaler.scale(loss_scaled).backward()
            if (
                (batch_idx + 1) % accumulation_steps == 0
                or (batch_idx + 1) == loader_len
                or (max_batches and batch_idx + 1 == max_batches)
            ):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    scheduler.step()
        else:
            loss_scaled.backward()
            if (
                (batch_idx + 1) % accumulation_steps == 0
                or (batch_idx + 1) == loader_len
                or (max_batches and batch_idx + 1 == max_batches)
            ):
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        total_cls += loss_dict["cls"].item()
        total_box += loss_dict["box"].item()
        total_dfl += loss_dict["dfl"].item()
        num_batches += 1

        if num_batches % 20 == 0 or num_batches == loader_len:
            logger.info(
                "  Batch %d/%d - loss: %.4f",
                num_batches,
                loader_len,
                total_loss / num_batches,
            )

    n = max(num_batches, 1)
    return total_loss / n, total_cls / n, total_box / n, total_dfl / n


def _set_backbone_trainable(model: FruitDetectorV2, trainable: bool) -> None:
    if trainable:
        model.unfreeze_backbone()
    else:
        model.freeze_backbone()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    img_size: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run validation and return torchmetrics MeanAveragePrecision results."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError("Install torchmetrics: pip install torchmetrics") from exc

    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=False)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, boxes_list, labels_list, _ = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        cls_pred, box_ltrb, _, anchor_points, stride_tensor = model(images)
        for b in range(images.shape[0]):
            pred_boxes, pred_labels, pred_scores = decode_predictions_v2(
                cls_pred[b],
                box_ltrb[b],
                anchor_points,
                stride_tensor,
                conf_thresh=0.001,
                nms_iou=NMS_IOU,
                pre_nms_topk=PRE_NMS_TOPK,
                max_detections=300,
                img_size=img_size,
            )
            gt_boxes_b = boxes_list[b]
            if gt_boxes_b.numel() > 0:
                x1 = gt_boxes_b[:, 0] - gt_boxes_b[:, 2] / 2
                y1 = gt_boxes_b[:, 1] - gt_boxes_b[:, 3] / 2
                x2 = gt_boxes_b[:, 0] + gt_boxes_b[:, 2] / 2
                y2 = gt_boxes_b[:, 1] + gt_boxes_b[:, 3] / 2
                gt_xyxy = torch.stack([x1, y1, x2, y2], dim=1)
            else:
                gt_xyxy = torch.zeros((0, 4))
            preds = [
                {
                    "boxes": pred_boxes.cpu(),
                    "scores": pred_scores.cpu(),
                    "labels": pred_labels.cpu(),
                }
            ]
            targets = [{"boxes": gt_xyxy.cpu(), "labels": labels_list[b].cpu()}]
            metric.update(preds, targets)
    return metric.compute()


# ---------------------------------------------------------------------------
# Cache / DataLoader helpers
# ---------------------------------------------------------------------------


def _cache_subdir(split_name: str) -> str:
    return os.path.join(CACHE_DIR, split_name)


def _make_loader(
    dataset: FruitDataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


def _resolve_workers(
    dataset: FruitDataset,
    requested_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
) -> int:
    if requested_workers <= 0:
        return 0
    candidates = [requested_workers]
    if requested_workers > 2:
        candidates.append(2)
    candidates.append(0)
    for workers in candidates:
        if workers == 0:
            return 0
        try:
            probe = _make_loader(
                dataset, 1, False, workers, pin_memory, prefetch_factor, persistent_workers
            )
            next(iter(probe))
            return workers
        except Exception as exc:
            logger.warning("DataLoader workers=%d failed: %s", workers, exc)
    return 0


# ---------------------------------------------------------------------------
# History CSV
# ---------------------------------------------------------------------------

HISTORY_FIELDS = [
    "epoch",
    "lr_backbone",
    "lr_head",
    "loss",
    "cls_loss",
    "box_loss",
    "dfl_loss",
    "map50",
    "map",
    "num_pos_avg",
    "epoch_seconds",
]


def _save_history_row(path: str, row: dict) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _truncate_history_to_epoch(path: str, last_epoch: int) -> None:
    if not os.path.exists(path):
        return
    rows: list[dict] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["epoch"]) <= last_epoch:
                rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def _config_snapshot(args: Any) -> dict:
    raw_args = vars(args) if args is not None else {}
    arg_values = {k: v for k, v in raw_args.items() if not callable(v)}
    return {
        "version": "v2",
        "backbone": BACKBONE_NAME,
        "neck_channels": NECK_CHANNELS,
        "num_head_convs": 2,
        "reg_max": REG_MAX,
        "strides": STRIDES,
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "batch_size": arg_values.get("batch_size", BATCH_SIZE),
        "freeze_backbone_epochs": arg_values.get("freeze_backbone_epochs", FREEZE_BACKBONE_EPOCHS),
        "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD,
        "weight_decay": WEIGHT_DECAY,
        "cls_weight": CLS_WEIGHT,
        "box_weight": BOX_WEIGHT,
        "dfl_weight": DFL_WEIGHT,
        "command_args": arg_values,
    }


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    best_map: float,
    best_loss: float,
    args: Any,
    ema: ModelEMA | None = None,
    scaler: torch.GradScaler | None = None,
    no_improve_count: int = 0,
) -> None:
    """Save a full training checkpoint (atomic write)."""
    save_dict: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_map50": best_map,
        "best_loss": best_loss,
        "no_improve_count": no_improve_count,
        "config": _config_snapshot(args),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
    }
    if torch.cuda.is_available():
        save_dict["rng_cuda"] = torch.cuda.get_rng_state()
    if ema is not None:
        save_dict["ema_state_dict"] = ema.state_dict()
    if scaler is not None:
        save_dict["scaler_state_dict"] = scaler.state_dict()
    tmp_path = path + ".tmp"
    torch.save(save_dict, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    device: torch.device,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ema: ModelEMA | None = None,
    scaler: torch.GradScaler | None = None,
) -> tuple[int, float, float, int]:
    """Load a checkpoint and restore full training state."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        _normalize_cem_weights(ckpt["model_state_dict"])
    if "ema_state_dict" in ckpt:
        ema_state = ckpt["ema_state_dict"]
        if isinstance(ema_state, dict):
            ema_model_dict = ema_state.get("model", ema_state)
            if isinstance(ema_model_dict, dict):
                _normalize_cem_weights(ema_model_dict)

    try:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except (KeyError, ValueError):
            logger.warning("Scheduler state mismatch; scheduler was reset.")
        if ema is not None and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "rng_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_torch"].cpu())
        if "rng_numpy" in ckpt:
            np.random.set_state(ckpt["rng_numpy"])
        if "rng_cuda" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu())
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map50", 0.0)
        best_loss = ckpt.get("best_loss", float("inf"))
        no_improve_count = ckpt.get("no_improve_count", 0)
        return start_epoch, best_map, best_loss, no_improve_count
    except RuntimeError:
        logger.warning("Architecture mismatch. Transfer learning via strict=False.")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        return 0, 0.0, float("inf"), 0


def _find_latest_checkpoint() -> str:
    last_path = os.path.join(WEIGHTS_DIR, "last.pt")
    if os.path.isfile(last_path):
        return last_path
    if not os.path.isdir(WEIGHTS_DIR):
        return ""
    emergency = sorted(
        [f for f in os.listdir(WEIGHTS_DIR) if f.startswith("emergency_") and f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(WEIGHTS_DIR, f)),
        reverse=True,
    )
    if emergency:
        return os.path.join(WEIGHTS_DIR, emergency[0])
    return ""


def _validate_environment() -> None:
    errors: list[str] = []
    for label, path in [
        ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
        ("TRAIN_LBL_DIR", TRAIN_LBL_DIR),
        ("VAL_IMG_DIR", VAL_IMG_DIR),
        ("VAL_LBL_DIR", VAL_LBL_DIR),
    ]:
        if not os.path.isdir(path):
            errors.append(f"  {label} does not exist: {path}")
        elif len(os.listdir(path)) == 0:
            errors.append(f"  {label} is empty: {path}")
    if errors:
        logger.error("Environment validation failed:")
        for e in errors:
            logger.error(e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Differential LR optimizer
# ---------------------------------------------------------------------------


def build_optimizer(
    model: torch.nn.Module,
    lr_backbone: float,
    lr_head: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with differential learning rates.

    - Backbone: lower LR (pre-trained weights need gentle fine-tuning)
    - Neck + heads: higher LR (randomly initialized)
    - Bias and BatchNorm params: no weight decay
    """
    backbone_params_decay: list[torch.nn.Parameter] = []
    backbone_params_nodecay: list[torch.nn.Parameter] = []
    other_params_decay: list[torch.nn.Parameter] = []
    other_params_nodecay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = "backbone" in name
        is_nodecay = ("bias" in name) or ("bn" in name) or ("norm" in name)

        if is_backbone:
            (backbone_params_nodecay if is_nodecay else backbone_params_decay).append(param)
        else:
            (other_params_nodecay if is_nodecay else other_params_decay).append(param)

    param_groups = [
        {"params": backbone_params_decay, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": backbone_params_nodecay, "lr": lr_backbone, "weight_decay": 0.0},
        {"params": other_params_decay, "lr": lr_head, "weight_decay": weight_decay},
        {"params": other_params_nodecay, "lr": lr_head, "weight_decay": 0.0},
    ]
    param_groups = [
        g for g in param_groups if isinstance(g["params"], list) and len(g["params"]) > 0
    ]
    try:
        return torch.optim.AdamW(param_groups, fused=True)
    except TypeError:
        return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def run_training(
    num_epochs: int = NUM_EPOCHS,
    resume: str = "",
    limit_train_batches: int | None = None,
    limit_val_batches: int | None = None,
    val_every: int = VAL_EVERY,
    skip_val: bool = False,
    workers: int = NUM_WORKERS,
    prefetch_factor: int = PREFETCH_FACTOR,
    persistent_workers: bool = PERSISTENT_WORKERS,
    cache_images: bool = False,
    build_cache: bool = False,
    freeze_backbone_epochs: int = FREEZE_BACKBONE_EPOCHS,
    batch_size: int = BATCH_SIZE,
    args: Any = None,
) -> None:
    """Full training pipeline."""
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

    if limit_train_batches is not None and limit_train_batches < 1:
        raise ValueError("--limit-train-batches must be >= 1")
    if limit_val_batches is not None and limit_val_batches < 1:
        raise ValueError("--limit-val-batches must be >= 1")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    _validate_environment()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    logger.info("Device: %s", device)
    logger.info(
        "Architecture: v2 anchor-free | Backbone: %s | Neck: %dch | Image: %dpx",
        BACKBONE_NAME,
        NECK_CHANNELS,
        IMG_SIZE,
    )

    # Build disk cache
    if build_cache:
        logger.info("Building disk image cache (one-time)...")
        for split_name, img_dir, lbl_dir in [
            ("train", TRAIN_IMG_DIR, TRAIN_LBL_DIR),
            ("valid", VAL_IMG_DIR, VAL_LBL_DIR),
        ]:
            ds = FruitDataset(
                img_dir,
                lbl_dir,
                IMG_SIZE,
                augment=False,
                cache_dir=_cache_subdir(split_name),
                cache_images=False,
            )
            cache_result = ds.build_cache(overwrite=False, verbose=True)
            logger.info(
                "  %s: %d written, %d skipped",
                split_name,
                cache_result["written"],
                cache_result["skipped"],
            )
        logger.info("Cache build complete.")
    if num_epochs == 0:
        return

    # Datasets
    train_ds = FruitDataset(
        TRAIN_IMG_DIR,
        TRAIN_LBL_DIR,
        IMG_SIZE,
        augment=True,
        cache_dir=_cache_subdir("train"),
        cache_images=cache_images,
        mosaic_prob=MOSAIC_PROB,
        mixup_prob=MIXUP_PROB,
        copy_paste_prob=COPY_PASTE_PROB,
    )
    val_ds = FruitDataset(
        VAL_IMG_DIR,
        VAL_LBL_DIR,
        IMG_SIZE,
        augment=False,
        cache_dir=_cache_subdir("valid"),
        cache_images=cache_images,
    )

    pin_memory = device.type == "cuda"
    workers = _resolve_workers(train_ds, workers, pin_memory, prefetch_factor, persistent_workers)
    logger.info("DataLoader workers: %d", workers)
    train_sampler = RandomSampler(train_ds, num_samples=len(train_ds) // 3)
    train_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "sampler": train_sampler,
        "num_workers": workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor
        train_kwargs["persistent_workers"] = persistent_workers
    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = _make_loader(
        val_ds, batch_size, False, workers, pin_memory, prefetch_factor, persistent_workers
    )
    mosaic_disabled = False

    # Model
    logger.info("Loading %s backbone...", "pretrained" if PRETRAINED else "random")
    model = FruitDetectorV2(
        num_classes=NUM_CLASSES,
        img_size=IMG_SIZE,
        backbone_name=BACKBONE_NAME,
        pretrained=PRETRAINED,
        neck_channels=NECK_CHANNELS,
        reg_max=REG_MAX,
        strides=STRIDES,
        num_head_convs=2,
    )
    model.freeze_early_backbone()
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for n, p in model.named_parameters() if "backbone" in n)
    logger.info(
        "Parameters: %s total (%s backbone, %s neck+head)",
        f"{total_params:,}",
        f"{backbone_params:,}",
        f"{total_params - backbone_params:,}",
    )

    ema = ModelEMA(model, decay=0.9999)

    # Loss
    class_weights_logged = compute_class_weights_from_label_entries(train_ds._labels, NUM_CLASSES)
    logger.info(
        "Dataset natural class weights: %s", [f"{w:.3f}" for w in class_weights_logged.tolist()]
    )

    criterion = DetectionLossV2(
        num_classes=NUM_CLASSES,
        reg_max=REG_MAX,
        cls_weight=CLS_WEIGHT,
        box_weight=BOX_WEIGHT,
        dfl_weight=DFL_WEIGHT,
        tal_topk=TAL_TOPK,
        class_weights=None,
    ).to(device)

    # Optimizer
    optimizer = build_optimizer(model, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY)
    logger.info("Optimizer param groups: %d", len(optimizer.param_groups))
    for i, g in enumerate(optimizer.param_groups):
        logger.info(
            "  Group %d: %d params, lr=%s, wd=%s",
            i,
            len(g["params"]),
            g["lr"],
            g["weight_decay"],
        )

    # AMP
    scaler = torch.GradScaler("cuda") if device.type == "cuda" else None

    # Scheduler
    steps_per_epoch = math.ceil(len(train_loader) / 3)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1,
    )

    best_map = 0.0
    best_loss = float("inf")
    start_epoch = 0
    no_improve_count = 0

    # Resume
    if not resume:
        resume = _find_latest_checkpoint()
    if resume.lower() in {"none", "false", "null", "fresh"}:
        resume = ""

    if resume and os.path.isfile(resume):
        start_epoch, best_map, best_loss, no_improve_count = load_checkpoint(
            resume,
            device,
            model,
            optimizer,
            scheduler,
            ema=ema,
            scaler=scaler,
        )
        logger.info("[RESUME] Loaded: %s", resume)
        logger.info(
            "[RESUME] Epoch %d, best_mAP50=%.4f, best_loss=%.4f", start_epoch, best_map, best_loss
        )
    else:
        if resume:
            logger.warning("Checkpoint not found at '%s', starting fresh.", resume)
        logger.info("[FRESH] Starting training from scratch.")

    history_path = os.path.join(RUNS_DIR, "history.csv")
    if start_epoch > 0:
        _truncate_history_to_epoch(history_path, start_epoch - 1)
    else:
        if os.path.exists(history_path):
            with contextlib.suppress(OSError):
                os.remove(history_path)

    if args is not None:
        with open(os.path.join(RUNS_DIR, "config_snapshot.json"), "w") as f:
            json.dump(_config_snapshot(args), f, indent=2)

    # Signal handler
    _interrupt_state = {"requested": False}

    def _graceful_shutdown(signum: int, frame: Any) -> None:
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.warning("Received %s. Saving checkpoint...", sig_name)
        _interrupt_state["requested"] = True

    signal.signal(signal.SIGINT, _graceful_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    # Training loop
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch
            if _interrupt_state["requested"]:
                raise KeyboardInterrupt

            if epoch < freeze_backbone_epochs:
                _set_backbone_trainable(model, False)
            elif epoch == freeze_backbone_epochs:
                _set_backbone_trainable(model, True)
                logger.info("[Epoch %d] Backbone unfrozen.", epoch + 1)

            mosaic_off = (num_epochs - epoch) <= MOSAIC_OFF_EPOCHS
            if mosaic_off and not mosaic_disabled:
                train_ds.set_augmentation_probs(mosaic_prob=0.0)
                train_sampler = RandomSampler(train_ds, num_samples=len(train_ds) // 3)
                train_kwargs = {
                    "batch_size": batch_size,
                    "sampler": train_sampler,
                    "num_workers": workers,
                    "collate_fn": collate_fn,
                    "pin_memory": pin_memory,
                }
                if workers > 0:
                    train_kwargs["prefetch_factor"] = prefetch_factor
                    train_kwargs["persistent_workers"] = persistent_workers
                train_loader = DataLoader(train_ds, **train_kwargs)
                mosaic_disabled = True
                logger.info("[Epoch %d] Mosaic disabled for final fine-tuning.", epoch + 1)

            epoch_start = time.perf_counter()
            loss, cls_loss, box_loss, dfl_loss_val = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                GRAD_CLIP,
                max_batches=limit_train_batches,
                ema=ema,
                scaler=scaler,
                scheduler=scheduler,
            )

            effective_val_every = max(val_every * 2, 10) if epoch < num_epochs // 2 else val_every
            should_validate = (not skip_val) and (
                (epoch + 1) % effective_val_every == 0 or epoch == num_epochs - 1
            )
            if should_validate:
                try:
                    val_model = ema.ema_model if ema is not None else model
                    result = validate(
                        val_model, val_loader, device, IMG_SIZE, max_batches=limit_val_batches
                    )
                    map50 = result["map_50"].item()
                    map_val = result["map"].item()
                except (RuntimeError, ModuleNotFoundError) as exc:
                    if "torchmetrics" not in str(exc):
                        raise
                    logger.warning("Validation skipped: install torchmetrics.")
                    skip_val = True
                    map50 = best_map
                    map_val = float("nan")
            else:
                map50 = best_map
                map_val = float("nan")

            epoch_seconds = time.perf_counter() - epoch_start
            lr_bb = optimizer.param_groups[0]["lr"]
            lr_hd = optimizer.param_groups[-1]["lr"] if len(optimizer.param_groups) > 1 else lr_bb
            logger.info(
                "Epoch %d/%d | loss=%.4f cls=%.4f box=%.4f dfl=%.4f | mAP50=%.4f mAP=%.4f | lr=%.6f | time=%.1fs",
                epoch + 1,
                num_epochs,
                loss,
                cls_loss,
                box_loss,
                dfl_loss_val,
                map50,
                map_val,
                lr_hd,
                epoch_seconds,
            )

            _save_history_row(
                history_path,
                {
                    "epoch": epoch + 1,
                    "lr_backbone": lr_bb,
                    "lr_head": lr_hd,
                    "loss": loss,
                    "cls_loss": cls_loss,
                    "box_loss": box_loss,
                    "dfl_loss": dfl_loss_val,
                    "map50": map50,
                    "map": map_val,
                    "num_pos_avg": "",
                    "epoch_seconds": epoch_seconds,
                },
            )

            if loss < best_loss:
                best_loss = loss
                save_checkpoint(
                    os.path.join(WEIGHTS_DIR, "best_loss.pt"),
                    epoch,
                    model,
                    optimizer,
                    scheduler,
                    best_map,
                    best_loss,
                    args,
                    ema=ema,
                    scaler=scaler,
                    no_improve_count=no_improve_count,
                )

            if should_validate:
                best_map_path = os.path.join(WEIGHTS_DIR, "best_map50.pt")
                if map50 > best_map:
                    no_improve_count = 0
                    best_map = map50
                    save_checkpoint(
                        best_map_path,
                        epoch,
                        model,
                        optimizer,
                        scheduler,
                        best_map,
                        best_loss,
                        args,
                        ema=ema,
                        scaler=scaler,
                        no_improve_count=no_improve_count,
                    )
                    logger.info("  -> New best mAP50: %.4f", best_map)
                else:
                    if not os.path.exists(best_map_path):
                        save_checkpoint(
                            best_map_path,
                            epoch,
                            model,
                            optimizer,
                            scheduler,
                            best_map,
                            best_loss,
                            args,
                            ema=ema,
                            scaler=scaler,
                            no_improve_count=no_improve_count,
                        )
                    no_improve_count += 1
                    if no_improve_count >= PATIENCE:
                        logger.info(
                            "Early stopping: no improvement for %d validation cycles.", PATIENCE
                        )
                        break

            last_path = os.path.join(WEIGHTS_DIR, "last.pt")
            save_checkpoint(
                last_path,
                epoch,
                model,
                optimizer,
                scheduler,
                best_map,
                best_loss,
                args,
                ema=ema,
                scaler=scaler,
                no_improve_count=no_improve_count,
            )

    except KeyboardInterrupt:
        logger.warning("Training interrupted at epoch %d.", current_epoch + 1)
        emergency_path = os.path.join(WEIGHTS_DIR, f"emergency_epoch{current_epoch + 1}.pt")
        save_checkpoint(
            emergency_path,
            current_epoch,
            model,
            optimizer,
            scheduler,
            best_map,
            best_loss,
            args,
            ema=ema,
            scaler=scaler,
            no_improve_count=no_improve_count,
        )
        last_path = os.path.join(WEIGHTS_DIR, "last.pt")
        save_checkpoint(
            last_path,
            current_epoch,
            model,
            optimizer,
            scheduler,
            best_map,
            best_loss,
            args,
            ema=ema,
            scaler=scaler,
            no_improve_count=no_improve_count,
        )
        logger.warning("Emergency checkpoint saved: %s", emergency_path)
        sys.exit(130)

    logger.info("Training complete.")
