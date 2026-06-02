"""Training loop — production pipeline with anchor-free detector."""
import argparse
import csv
import json
import math
import numpy as np
import os
import random
import signal
import sys
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from config import (
    TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR,
    IMG_SIZE, NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS,
    LR_BACKBONE, LR_HEAD, WEIGHT_DECAY, GRAD_CLIP,
    RUNS_DIR, WEIGHTS_DIR, CACHE_DIR,
    BACKBONE_NAME, PRETRAINED, NECK_CHANNELS, REG_MAX, STRIDES,
    CLS_WEIGHT, BOX_WEIGHT, DFL_WEIGHT, TAL_TOPK,
    PATIENCE, VAL_EVERY,
    FREEZE_BACKBONE_EPOCHS, MOSAIC_OFF_EPOCHS,
    CONF_THRESH, NMS_IOU, PRE_NMS_TOPK, MAX_DETECTIONS,
    NUM_WORKERS, PREFETCH_FACTOR, PERSISTENT_WORKERS,
    MOSAIC_PROB, MIXUP_PROB, COPY_PASTE_PROB,
    CLASS_NAMES,
)
from custom_detector.dataset import FruitDataset, collate_fn
from custom_detector.model import FruitDetectorV2
from custom_detector.loss import DetectionLossV2, compute_class_weights_from_label_entries
from custom_detector.inference import decode_predictions_v2
from custom_detector.anchor_points import decode_boxes
import torchvision.ops as ops


# ---------------------------------------------------------------------------
# Model EMA
# ---------------------------------------------------------------------------

class ModelEMA:
    """Exponential Moving Average of model weights."""
    def __init__(self, model, decay=0.9999):
        import copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.lerp_(model_p.detach(), 1.0 - d)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self):
        return {'model': self.ema_model.state_dict(), 'updates': self.updates}

    def load_state_dict(self, state_dict):
        if 'model' in state_dict and 'updates' in state_dict:
            self.ema_model.load_state_dict(state_dict['model'])
            self.updates = state_dict['updates']
        else:
            self.ema_model.load_state_dict(state_dict)




# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

def unpack_batch(batch):
    if len(batch) == 4:
        images, boxes_list, labels_list, target_keys = batch
    else:
        images, boxes_list, labels_list = batch
        target_keys = None
    return images, boxes_list, labels_list, target_keys


# ---------------------------------------------------------------------------
# Train / Validate
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip,
                    max_batches=None, ema=None, scaler=None, scheduler=None):
    """Run one epoch of training. Returns (avg_loss, avg_cls, avg_box, avg_dfl)."""
    model.train()
    total_loss = total_cls = total_box = total_dfl = 0
    num_batches = 0
    use_amp = scaler is not None
    loader_len = len(loader)

    # Dynamic Multi-Scale logic
    scales = [352, 384, 416, 448, 480, 512]
    ACCUMULATION_STEPS = 3
    target_sz = None

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, boxes_list, labels_list, _ = unpack_batch(batch)

        if target_sz is None:
            target_sz = images.shape[-1]

        if batch_idx % 10 == 0:  # Change scale every 10 batches
            target_sz = random.choice(scales)

        if target_sz != images.shape[-1]:
            scale_ratio = target_sz / images.shape[-1]
            images = torch.nn.functional.interpolate(images, size=(target_sz, target_sz), mode='bilinear', align_corners=False)
            boxes_list = [b * scale_ratio for b in boxes_list]

        images = images.to(device, non_blocking=True)
        if device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)

        # Centralize CPU->GPU transfers to avoid stalling upstream TAL loops
        boxes_list = [b.to(device, non_blocking=True).float() for b in boxes_list]
        labels_list = [l.to(device, non_blocking=True) for l in labels_list]

        with torch.autocast(device_type='cuda', enabled=use_amp):
            cls_pred, box_ltrb, box_raw, anchor_points, stride_tensor = model(images)
            loss_dict = criterion(cls_pred, box_ltrb, box_raw, anchor_points, stride_tensor,
                                  boxes_list, labels_list)
            loss = loss_dict['total']

        # Gradient Accumulation
        loss_scaled = loss / ACCUMULATION_STEPS
        
        if use_amp:
            scaler.scale(loss_scaled).backward()
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == loader_len or (max_batches and batch_idx + 1 == max_batches):
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
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == loader_len or (max_batches and batch_idx + 1 == max_batches):
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
                if scheduler is not None:
                    scheduler.step()

        total_loss += loss.item()
        total_cls += loss_dict['cls'].item()
        total_box += loss_dict['box'].item()
        total_dfl += loss_dict['dfl'].item()
        num_batches += 1

        if num_batches % 20 == 0 or num_batches == loader_len:
            print(f"  Batch {num_batches}/{loader_len} - loss: {total_loss/num_batches:.4f}", end='\r')

    print()
    n = max(num_batches, 1)
    return total_loss / n, total_cls / n, total_box / n, total_dfl / n


def set_backbone_trainable(model, trainable):
    if trainable:
        model.unfreeze_backbone()
    else:
        model.freeze_backbone()


@torch.no_grad()
def validate(model, loader, device, img_size, max_batches=None):
    """Run validation and return torchmetrics MeanAveragePrecision results."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
    except ImportError as exc:
        raise RuntimeError("Install torchmetrics: pip install torchmetrics") from exc

    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox', class_metrics=False)
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images, boxes_list, labels_list, _ = unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if device.type == 'cuda':
            images = images.contiguous(memory_format=torch.channels_last)
        cls_pred, box_ltrb, box_raw, anchor_points, stride_tensor = model(images)
        for b in range(images.shape[0]):
            pred_boxes, pred_labels, pred_scores = decode_predictions_v2(
                cls_pred[b], box_ltrb[b], anchor_points, stride_tensor,
                conf_thresh=0.001, nms_iou=NMS_IOU,
                pre_nms_topk=PRE_NMS_TOPK, max_detections=300,
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
            preds = [{'boxes': pred_boxes.cpu(), 'scores': pred_scores.cpu(), 'labels': pred_labels.cpu()}]
            targets = [{'boxes': gt_xyxy.cpu(), 'labels': labels_list[b].cpu()}]
            metric.update(preds, targets)
    return metric.compute()


# ---------------------------------------------------------------------------
# Cache / DataLoader helpers
# ---------------------------------------------------------------------------

def cache_subdir(split_name):
    return os.path.join(CACHE_DIR, split_name)


def make_loader(dataset, batch_size, shuffle, workers, pin_memory, prefetch_factor, persistent_workers):
    kwargs = {
        "batch_size": batch_size, "shuffle": shuffle, "num_workers": workers,
        "collate_fn": collate_fn, "pin_memory": pin_memory,
    }
    if workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(dataset, **kwargs)


def resolve_workers(dataset, requested_workers, pin_memory, prefetch_factor, persistent_workers):
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
            probe = make_loader(dataset, 1, False, workers, pin_memory, prefetch_factor, persistent_workers)
            next(iter(probe))
            return workers
        except Exception as exc:
            print(f"DataLoader workers={workers} failed: {exc}")
    return 0


# ---------------------------------------------------------------------------
# History CSV
# ---------------------------------------------------------------------------

HISTORY_FIELDS = ["epoch", "lr_backbone", "lr_head", "loss", "cls_loss", "box_loss", "dfl_loss",
                  "map50", "map", "num_pos_avg", "epoch_seconds"]


def save_history_row(path, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def truncate_history_to_epoch(path, last_epoch):
    if not os.path.exists(path):
        return
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["epoch"]) <= last_epoch:
                rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Checkpoint save / load (atomic writes, full state)
# ---------------------------------------------------------------------------

def config_snapshot(args):
    arg_values = vars(args) if args is not None else {}
    return {
        "version": "v2",
        "backbone": BACKBONE_NAME,
        "neck_channels": NECK_CHANNELS,
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


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_map, best_loss,
                    args, ema=None, scaler=None, no_improve_count=0):
    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_map50': best_map,
        'best_loss': best_loss,
        'no_improve_count': no_improve_count,
        'config': config_snapshot(args),
        'rng_torch': torch.get_rng_state(),
        'rng_numpy': np.random.get_state(),
    }
    if torch.cuda.is_available():
        save_dict['rng_cuda'] = torch.cuda.get_rng_state()
    if ema is not None:
        save_dict['ema_state_dict'] = ema.state_dict()
    if scaler is not None:
        save_dict['scaler_state_dict'] = scaler.state_dict()
    tmp_path = path + ".tmp"
    torch.save(save_dict, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, device, model, optimizer, scheduler, ema=None, scaler=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    try:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (KeyError, ValueError):
            print("Warning: scheduler state mismatch; scheduler was reset.")
        if ema is not None and 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        if scaler is not None and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        if 'rng_torch' in ckpt:
            torch.set_rng_state(ckpt['rng_torch'].cpu())
        if 'rng_numpy' in ckpt:
            np.random.set_state(ckpt['rng_numpy'])
        if 'rng_cuda' in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state(ckpt['rng_cuda'].cpu())
        start_epoch = ckpt['epoch'] + 1
        best_map = ckpt.get('best_map50', 0.0)
        best_loss = ckpt.get('best_loss', float('inf'))
        no_improve_count = ckpt.get('no_improve_count', 0)
        return start_epoch, best_map, best_loss, no_improve_count
    except RuntimeError as e:
        print(f"Architecture mismatch detected. Transfer learning via strict=False and resetting training state.")
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        return 0, 0.0, float('inf'), 0


def find_latest_checkpoint():
    last_path = os.path.join(WEIGHTS_DIR, 'last.pt')
    if os.path.isfile(last_path):
        return last_path
    if not os.path.isdir(WEIGHTS_DIR):
        return ''
    emergency = sorted(
        [f for f in os.listdir(WEIGHTS_DIR) if f.startswith('emergency_') and f.endswith('.pt')],
        key=lambda f: os.path.getmtime(os.path.join(WEIGHTS_DIR, f)),
        reverse=True,
    )
    if emergency:
        return os.path.join(WEIGHTS_DIR, emergency[0])
    return ''


def validate_environment():
    errors = []
    for label, path in [
        ("TRAIN_IMG_DIR", TRAIN_IMG_DIR), ("TRAIN_LBL_DIR", TRAIN_LBL_DIR),
        ("VAL_IMG_DIR", VAL_IMG_DIR), ("VAL_LBL_DIR", VAL_LBL_DIR),
    ]:
        if not os.path.isdir(path):
            errors.append(f"  {label} does not exist: {path}")
        elif len(os.listdir(path)) == 0:
            errors.append(f"  {label} is empty: {path}")
    if errors:
        print("ERROR: Environment validation failed:")
        for e in errors:
            print(e)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Differential LR optimizer
# ---------------------------------------------------------------------------

def build_optimizer(model, lr_backbone, lr_head, weight_decay):
    """Build AdamW optimizer with differential learning rates.

    - Backbone: lower LR (pre-trained weights need gentle fine-tuning)
    - Neck + heads: higher LR (randomly initialized)
    - Bias and BatchNorm params: no weight decay
    """
    backbone_params_decay = []
    backbone_params_nodecay = []
    other_params_decay = []
    other_params_nodecay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_backbone = 'backbone' in name
        is_nodecay = ('bias' in name) or ('bn' in name) or ('norm' in name)

        if is_backbone:
            if is_nodecay:
                backbone_params_nodecay.append(param)
            else:
                backbone_params_decay.append(param)
        else:
            if is_nodecay:
                other_params_nodecay.append(param)
            else:
                other_params_decay.append(param)

    param_groups = [
        {'params': backbone_params_decay, 'lr': lr_backbone, 'weight_decay': weight_decay},
        {'params': backbone_params_nodecay, 'lr': lr_backbone, 'weight_decay': 0.0},
        {'params': other_params_decay, 'lr': lr_head, 'weight_decay': weight_decay},
        {'params': other_params_nodecay, 'lr': lr_head, 'weight_decay': 0.0},
    ]
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    try:
        return torch.optim.AdamW(param_groups, fused=True)
    except TypeError:
        return torch.optim.AdamW(param_groups)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main(num_epochs=NUM_EPOCHS, resume='', limit_train_batches=None, limit_val_batches=None,
         val_every=VAL_EVERY, skip_val=False, workers=NUM_WORKERS,
         prefetch_factor=PREFETCH_FACTOR, persistent_workers=PERSISTENT_WORKERS,
         cache_images=False, build_cache=False, freeze_backbone_epochs=FREEZE_BACKBONE_EPOCHS,
         batch_size=BATCH_SIZE, args=None):
    if limit_train_batches is not None and limit_train_batches < 1:
        raise ValueError("--limit-train-batches must be >= 1")
    if limit_val_batches is not None and limit_val_batches < 1:
        raise ValueError("--limit-val-batches must be >= 1")

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True

    validate_environment()
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device}")
    print(f"Architecture: v2 anchor-free | Backbone: {BACKBONE_NAME} | Neck: {NECK_CHANNELS}ch | Image: {IMG_SIZE}px")

    # --- Build disk cache if requested (one-time operation) ---
    if build_cache:
        print("Building disk image cache (one-time)...")
        for split_name, img_dir, lbl_dir in [
            ("train", TRAIN_IMG_DIR, TRAIN_LBL_DIR),
            ("valid", VAL_IMG_DIR, VAL_LBL_DIR),
        ]:
            ds = FruitDataset(img_dir, lbl_dir, IMG_SIZE, augment=False,
                              cache_dir=cache_subdir(split_name), cache_images=False)
            result = ds.build_cache(overwrite=False, verbose=True)
            print(f"  {split_name}: {result['written']} written, {result['skipped']} skipped")
        print("Cache build complete.")
        if num_epochs == 0:
            return  # build-cache-only mode

    # --- Datasets ---
    train_ds = FruitDataset(TRAIN_IMG_DIR, TRAIN_LBL_DIR, IMG_SIZE, augment=True,
                            cache_dir=cache_subdir("train"), cache_images=cache_images,
                            mosaic_prob=MOSAIC_PROB, mixup_prob=MIXUP_PROB,
                            copy_paste_prob=COPY_PASTE_PROB)
    val_ds = FruitDataset(VAL_IMG_DIR, VAL_LBL_DIR, IMG_SIZE, augment=False,
                          cache_dir=cache_subdir("valid"), cache_images=cache_images)

    pin_memory = device.type == 'cuda'
    workers = resolve_workers(train_ds, workers, pin_memory, prefetch_factor, persistent_workers)
    print(f"DataLoader workers: {workers}")
    train_sampler = RandomSampler(train_ds, num_samples=len(train_ds) // 3)
    train_kwargs = {
        "batch_size": batch_size, "sampler": train_sampler, "num_workers": workers,
        "collate_fn": collate_fn, "pin_memory": pin_memory,
    }
    if workers > 0:
        train_kwargs["prefetch_factor"] = prefetch_factor
        train_kwargs["persistent_workers"] = persistent_workers
    train_loader = DataLoader(train_ds, **train_kwargs)
    val_loader = make_loader(val_ds, batch_size, False, workers, pin_memory, prefetch_factor, persistent_workers)
    mosaic_disabled = False

    # --- Model ---
    print(f"Loading {'pretrained' if PRETRAINED else 'random'} backbone...")
    model = FruitDetectorV2(
        num_classes=NUM_CLASSES, img_size=IMG_SIZE,
        backbone_name=BACKBONE_NAME, pretrained=PRETRAINED,
        neck_channels=NECK_CHANNELS, reg_max=REG_MAX, strides=STRIDES,
        num_head_convs=1
    )
    model.freeze_early_backbone()
    model = model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for n, p in model.named_parameters() if 'backbone' in n)
    print(f"Parameters: {total_params:,} total ({backbone_params:,} backbone, {total_params - backbone_params:,} neck+head)")

    ema = ModelEMA(model, decay=0.9999)

    # --- Loss ---
    class_weights = compute_class_weights_from_label_entries(train_ds._labels, NUM_CLASSES)
    print(f"Class weights: {[f'{w:.3f}' for w in class_weights.tolist()]}")

    criterion = DetectionLossV2(
        num_classes=NUM_CLASSES, reg_max=REG_MAX,
        cls_weight=CLS_WEIGHT, box_weight=BOX_WEIGHT, dfl_weight=DFL_WEIGHT,
        tal_topk=TAL_TOPK, class_weights=class_weights,
    ).to(device)

    # --- Optimizer with differential LR ---
    optimizer = build_optimizer(model, LR_BACKBONE, LR_HEAD, WEIGHT_DECAY)
    print(f"Optimizer param groups: {len(optimizer.param_groups)}")
    for i, g in enumerate(optimizer.param_groups):
        print(f"  Group {i}: {len(g['params'])} params, lr={g['lr']}, wd={g['weight_decay']}")

    # --- AMP ---
    scaler = torch.GradScaler('cuda') if device.type == 'cuda' else None

    # --- Scheduler ---
    steps_per_epoch = math.ceil(len(train_loader) / 3)  # ACCUMULATION_STEPS = 3
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE, LR_BACKBONE, LR_HEAD, LR_HEAD],
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.1
    )

    best_map = 0.0
    best_loss = float('inf')
    start_epoch = 0
    no_improve_count = 0

    # --- Resume ---
    if not resume:
        resume = find_latest_checkpoint()
    if resume and os.path.isfile(resume):
        start_epoch, best_map, best_loss, no_improve_count = load_checkpoint(
            resume, device, model, optimizer, scheduler, ema=ema, scaler=scaler
        )
        print(f"[RESUME] Loaded: {resume}")
        print(f"[RESUME] Epoch {start_epoch}, best_mAP50={best_map:.4f}, best_loss={best_loss:.4f}")
    else:
        if resume:
            print(f"Warning: checkpoint not found at '{resume}', starting fresh.")
        print("[FRESH] Starting training from scratch.")

    history_path = os.path.join(RUNS_DIR, "history.csv")
    if start_epoch > 0:
        truncate_history_to_epoch(history_path, start_epoch - 1)

    if args is not None:
        with open(os.path.join(RUNS_DIR, "config_snapshot.json"), "w") as f:
            json.dump(config_snapshot(args), f, indent=2)

    # --- Signal handler ---
    _interrupt_state = {'requested': False}

    def _graceful_shutdown(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
        print(f"\n[!] Received {sig_name}. Saving checkpoint...")
        _interrupt_state['requested'] = True

    signal.signal(signal.SIGINT, _graceful_shutdown)
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, _graceful_shutdown)

    # --- Training loop ---
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, num_epochs):
            current_epoch = epoch
            if _interrupt_state['requested']:
                raise KeyboardInterrupt

            # Backbone freeze/unfreeze schedule
            if epoch < freeze_backbone_epochs:
                set_backbone_trainable(model, False)
            elif epoch == freeze_backbone_epochs:
                set_backbone_trainable(model, True)
                print(f"[Epoch {epoch+1}] Backbone unfrozen — all parameters now trainable.")

            # Mosaic-off for final epochs
            mosaic_off = (num_epochs - epoch) <= MOSAIC_OFF_EPOCHS
            if mosaic_off and not mosaic_disabled:
                train_ds.set_augmentation_probs(mosaic_prob=0.0)
                # Recreate loader with sampler to preserve epoch step count and prevent worker dataset copies from keeping mosaic
                train_sampler = RandomSampler(train_ds, num_samples=len(train_ds) // 3)
                train_kwargs = {
                    "batch_size": batch_size, "sampler": train_sampler, "num_workers": workers,
                    "collate_fn": collate_fn, "pin_memory": pin_memory,
                }
                if workers > 0:
                    train_kwargs["prefetch_factor"] = prefetch_factor
                    train_kwargs["persistent_workers"] = persistent_workers
                train_loader = DataLoader(train_ds, **train_kwargs)
                mosaic_disabled = True
                print(f"[Epoch {epoch+1}] Mosaic disabled for final fine-tuning.")

            epoch_start = time.perf_counter()
            loss, cls_loss, box_loss, dfl_loss_val = train_one_epoch(
                model, train_loader, criterion, optimizer, device, GRAD_CLIP,
                max_batches=limit_train_batches, ema=ema, scaler=scaler, scheduler=scheduler
            )

            # Smart validation schedule: validate less frequently early on
            effective_val_every = max(val_every * 2, 10) if epoch < num_epochs // 2 else val_every
            should_validate = (not skip_val) and ((epoch + 1) % effective_val_every == 0 or epoch == num_epochs - 1)
            if should_validate:
                try:
                    val_model = ema.ema_model if ema is not None else model
                    result = validate(val_model, val_loader, device, IMG_SIZE, max_batches=limit_val_batches)
                    map50 = result['map_50'].item()
                    map_val = result['map'].item()
                except (RuntimeError, ModuleNotFoundError) as exc:
                    if 'torchmetrics' not in str(exc):
                        raise
                    print("Validation skipped: install torchmetrics.")
                    skip_val = True
                    map50 = best_map
                    map_val = float('nan')
            else:
                map50 = best_map
                map_val = float('nan')

            epoch_seconds = time.perf_counter() - epoch_start
            lr_bb = optimizer.param_groups[0]['lr']
            lr_hd = optimizer.param_groups[-1]['lr'] if len(optimizer.param_groups) > 1 else lr_bb
            print(f"Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} cls={cls_loss:.4f} box={box_loss:.4f} dfl={dfl_loss_val:.4f} | "
                  f"mAP50={map50:.4f} mAP={map_val:.4f} | lr={lr_hd:.6f} | time={epoch_seconds:.1f}s")

            save_history_row(history_path, {
                "epoch": epoch + 1, "lr_backbone": lr_bb, "lr_head": lr_hd,
                "loss": loss, "cls_loss": cls_loss, "box_loss": box_loss, "dfl_loss": dfl_loss_val,
                "map50": map50, "map": map_val, "num_pos_avg": "",
                "epoch_seconds": epoch_seconds,
            })

            if loss < best_loss:
                best_loss = loss
                save_checkpoint(os.path.join(WEIGHTS_DIR, 'best_loss.pt'), epoch, model, optimizer, scheduler,
                                best_map, best_loss, args, ema=ema, scaler=scaler, no_improve_count=no_improve_count)

            if should_validate:
                best_map_path = os.path.join(WEIGHTS_DIR, 'best_map50.pt')
                if map50 > best_map:
                    no_improve_count = 0
                    best_map = map50
                    save_checkpoint(best_map_path, epoch, model, optimizer, scheduler,
                                    best_map, best_loss, args, ema=ema, scaler=scaler, no_improve_count=no_improve_count)
                    print(f"  -> New best mAP50: {best_map:.4f}")
                else:
                    if not os.path.exists(best_map_path):
                        save_checkpoint(best_map_path, epoch, model, optimizer, scheduler,
                                        best_map, best_loss, args, ema=ema, scaler=scaler, no_improve_count=no_improve_count)
                    no_improve_count += 1
                    if no_improve_count >= PATIENCE:
                        print(f"Early stopping: no improvement for {PATIENCE} validation cycles.")
                        break

            last_path = os.path.join(WEIGHTS_DIR, 'last.pt')
            save_checkpoint(last_path, epoch, model, optimizer, scheduler,
                            best_map, best_loss, args, ema=ema, scaler=scaler, no_improve_count=no_improve_count)

    except KeyboardInterrupt:
        print(f"\n[!] Training interrupted at epoch {current_epoch + 1}.")
        emergency_path = os.path.join(WEIGHTS_DIR, f'emergency_epoch{current_epoch + 1}.pt')
        save_checkpoint(emergency_path, current_epoch, model, optimizer, scheduler,
                        best_map, best_loss, args, ema=ema, scaler=scaler, no_improve_count=no_improve_count)
        last_path = os.path.join(WEIGHTS_DIR, 'last.pt')
        save_checkpoint(last_path, current_epoch, model, optimizer, scheduler,
                        best_map, best_loss, args, ema=ema, scaler=scaler, no_improve_count=no_improve_count)
        print(f"[!] Emergency checkpoint saved: {emergency_path}")
        print(f"[!] last.pt updated. Resume with same command.")
        sys.exit(130)

    print("Training complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train v2 anchor-free fruit detector")
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--limit-train-batches', type=int, default=None)
    parser.add_argument('--limit-val-batches', type=int, default=None)
    parser.add_argument('--val-every', type=int, default=VAL_EVERY)
    parser.add_argument('--skip-val', action='store_true')
    parser.add_argument('--workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--prefetch-factor', type=int, default=PREFETCH_FACTOR)
    parser.add_argument('--persistent-workers', action=argparse.BooleanOptionalAction, default=PERSISTENT_WORKERS)
    parser.add_argument('--cache-images', action='store_true',
                        help='Load images from pre-built disk cache (.npy files)')
    parser.add_argument('--build-cache', action='store_true',
                        help='Build disk cache of resized images before training')
    parser.add_argument('--freeze-backbone-epochs', type=int, default=FREEZE_BACKBONE_EPOCHS,
                        help='Keep the pretrained backbone frozen for this many epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help='Override train/validation batch size for throughput tuning')
    args = parser.parse_args()
    main(args.epochs, args.resume, args.limit_train_batches, args.limit_val_batches,
         args.val_every, args.skip_val, args.workers,
         args.prefetch_factor, args.persistent_workers,
         args.cache_images, args.build_cache, args.freeze_backbone_epochs,
         args.batch_size, args)
