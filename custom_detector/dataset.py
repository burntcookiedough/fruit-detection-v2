"""PyTorch Dataset for YOLO-format labels."""
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

# ImageNet channel-wise normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FruitDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, img_size=416, augment=False, cache_dir=None,
                 cache_images=False, return_meta=False, mosaic_prob=0.5,
                 mixup_prob=0.15, copy_paste_prob=0.15):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.img_size = img_size
        self.augment = augment
        self.cache_dir = cache_dir
        self.cache_images = cache_images
        self.return_meta = return_meta
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not os.path.isdir(lbl_dir):
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not self.img_files:
            raise RuntimeError(f"No images found in: {img_dir}")

        # Pre-load all labels into memory (they're tiny text files)
        self._labels = []
        for fname in self.img_files:
            stem = os.path.splitext(fname)[0]
            lbl_path = os.path.join(self.lbl_dir, stem + '.txt')
            boxes = []
            labels = []
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                            labels.append(int(parts[0]))
            self._labels.append({
                'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
                'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            })

    def __len__(self):
        return len(self.img_files)

    def cache_path(self, fname):
        if not self.cache_dir:
            return None
        stem = os.path.splitext(fname)[0]
        return os.path.join(self.cache_dir, f"{stem}.npy")

    def load_image(self, fname):
        img_path = os.path.join(self.img_dir, fname)
        cache_path = self.cache_path(fname)
        if self.cache_images and cache_path and os.path.exists(cache_path):
            arr = np.load(cache_path)
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)

        img = Image.open(img_path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return TF.to_tensor(img)

    def _get_labels(self, idx):
        """Get pre-loaded labels for image at idx, scaled to pixel coordinates."""
        entry = self._labels[idx]
        boxes = torch.tensor(entry['boxes'], dtype=torch.float32)
        labels = torch.tensor(entry['labels'], dtype=torch.long)
        if boxes.numel() > 0:
            boxes[:, 0] *= self.img_size  # cx
            boxes[:, 1] *= self.img_size  # cy
            boxes[:, 2] *= self.img_size  # w
            boxes[:, 3] *= self.img_size  # h
        return boxes, labels

    def set_augmentation_probs(self, mosaic_prob=None, mixup_prob=None, copy_paste_prob=None):
        if mosaic_prob is not None:
            self.mosaic_prob = mosaic_prob
        if mixup_prob is not None:
            self.mixup_prob = mixup_prob
        if copy_paste_prob is not None:
            self.copy_paste_prob = copy_paste_prob

    def build_cache(self, overwrite=False, verbose=True):
        if not self.cache_dir:
            raise ValueError("cache_dir must be set before building image cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        written = 0
        skipped = 0
        for idx, fname in enumerate(self.img_files, start=1):
            cache_path = self.cache_path(fname)
            if os.path.exists(cache_path) and not overwrite:
                skipped += 1
                continue
            img_path = os.path.join(self.img_dir, fname)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
            tmp_path = cache_path + ".tmp.npy"
            np.save(tmp_path, arr)
            os.replace(tmp_path, cache_path)
            written += 1
            if verbose and (idx % 1000 == 0 or idx == len(self.img_files)):
                print(f"  cached {idx}/{len(self.img_files)} images -> {self.cache_dir}")
        return {"written": written, "skipped": skipped, "total": len(self.img_files)}

    def _mixup(self, img1, boxes1, labels1):
        """MixUp augmentation: blends current image with a random other image."""
        idx2 = random.randint(0, len(self) - 1)
        boxes2, labels2 = self._get_labels(idx2)
        img2 = self.load_image(self.img_files[idx2])

        # Mixup ratio
        r = random.uniform(0.4, 0.6)
        img = img1 * r + img2 * (1 - r)

        boxes = torch.cat((boxes1, boxes2), 0)
        labels = torch.cat((labels1, labels2), 0)
        return img, boxes, labels

    def _copy_paste(self, img, boxes, labels):
        """Copy-Paste augmentation: copies random objects from another image into this one."""
        idx2 = random.randint(0, len(self) - 1)
        boxes2, labels2 = self._get_labels(idx2)
        if boxes2.numel() == 0:
            return img, boxes, labels

        img2 = self.load_image(self.img_files[idx2])
        num_objs = min(len(boxes2), random.randint(1, 3))
        indices = torch.randperm(len(boxes2))[:num_objs]

        new_boxes = boxes.tolist() if boxes.numel() > 0 else []
        new_labels = labels.tolist() if labels.numel() > 0 else []

        for idx in indices:
            bx, by, bw, bh = boxes2[idx]
            x1, y1 = int(max(bx - bw / 2, 0)), int(max(by - bh / 2, 0))
            x2, y2 = int(min(bx + bw / 2, self.img_size)), int(min(by + bh / 2, self.img_size))

            # Avoid copying too small patches or whole image
            if x2 - x1 < 10 or y2 - y1 < 10 or (x2-x1)*(y2-y1) > 0.5 * self.img_size**2:
                continue

            patch = img2[:, y1:y2, x1:x2]
            paste_x = random.randint(0, max(0, self.img_size - (x2 - x1)))
            paste_y = random.randint(0, max(0, self.img_size - (y2 - y1)))

            img[:, paste_y:paste_y+(y2-y1), paste_x:paste_x+(x2-x1)] = patch

            new_cx = paste_x + (x2 - x1) / 2
            new_cy = paste_y + (y2 - y1) / 2
            new_boxes.append([new_cx, new_cy, x2 - x1, y2 - y1])
            new_labels.append(labels2[idx].item())

        boxes_out = torch.tensor(new_boxes, dtype=torch.float32) if new_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_out = torch.tensor(new_labels, dtype=torch.long) if new_labels else torch.zeros((0,), dtype=torch.long)
        return img, boxes_out, labels_out

    def _mosaic(self, idx):
        """Combine 4 images into one mosaic composite."""
        img_size = self.img_size
        # Random center point for the mosaic
        cx = random.randint(img_size // 4, 3 * img_size // 4)
        cy = random.randint(img_size // 4, 3 * img_size // 4)

        indices = [idx] + random.choices(range(len(self)), k=3)
        all_boxes = []
        all_labels = []

        # Create canvas
        mosaic_img = torch.zeros(3, img_size, img_size)

        for i, idx_i in enumerate(indices):
            img_i = self.load_image(self.img_files[idx_i])
            boxes_i, labels_i = self._get_labels(idx_i)
            # Convert to lists for iteration
            boxes_list_i = boxes_i.tolist() if boxes_i.numel() > 0 else []
            labels_list_i = labels_i.tolist() if labels_i.numel() > 0 else []

            _, h, w = img_i.shape
            # Place each quadrant
            if i == 0:    # top-left
                x1s, y1s, x1e, y1e = max(cx - w, 0), max(cy - h, 0), cx, cy
                x2s, y2s, x2e, y2e = w - (x1e - x1s), h - (y1e - y1s), w, h
            elif i == 1:  # top-right
                x1s, y1s, x1e, y1e = cx, max(cy - h, 0), min(cx + w, img_size), cy
                x2s, y2s, x2e, y2e = 0, h - (y1e - y1s), min(w, x1e - x1s), h
            elif i == 2:  # bottom-left
                x1s, y1s, x1e, y1e = max(cx - w, 0), cy, cx, min(cy + h, img_size)
                x2s, y2s, x2e, y2e = w - (x1e - x1s), 0, w, min(h, y1e - y1s)
            else:         # bottom-right
                x1s, y1s, x1e, y1e = cx, cy, min(cx + w, img_size), min(cy + h, img_size)
                x2s, y2s, x2e, y2e = 0, 0, min(w, x1e - x1s), min(h, y1e - y1s)

            mosaic_img[:, y1s:y1e, x1s:x1e] = img_i[:, y2s:y2e, x2s:x2e]

            # Shift boxes: convert cxcywh to xyxy, shift, clip, convert back
            for bi, (bx, by, bw, bh) in enumerate(boxes_list_i):
                # Original box in xyxy
                ox1 = bx - bw / 2
                oy1 = by - bh / 2
                ox2 = bx + bw / 2
                oy2 = by + bh / 2
                # Shift to mosaic coords
                shift_x = x1s - x2s
                shift_y = y1s - y2s
                nx1 = max(ox1 + shift_x, x1s)
                ny1 = max(oy1 + shift_y, y1s)
                nx2 = min(ox2 + shift_x, x1e)
                ny2 = min(oy2 + shift_y, y1e)
                # Check if box is still valid (min area)
                if nx2 - nx1 > 2 and ny2 - ny1 > 2:
                    new_cx = (nx1 + nx2) / 2
                    new_cy = (ny1 + ny2) / 2
                    new_w = nx2 - nx1
                    new_h = ny2 - ny1
                    all_boxes.append([new_cx, new_cy, new_w, new_h])
                    all_labels.append(labels_list_i[bi])

        boxes = torch.tensor(all_boxes, dtype=torch.float32) if all_boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(all_labels, dtype=torch.long) if all_labels else torch.zeros((0,), dtype=torch.long)
        return mosaic_img, boxes, labels

    def __getitem__(self, idx):
        fname = self.img_files[idx]

        # Mosaic augmentation: combine 4 images with 50% probability during training
        if self.augment and self.mosaic_prob > 0 and random.random() < self.mosaic_prob:
            img, boxes, labels = self._mosaic(idx)
        else:
            # Use pre-loaded labels (no file I/O)
            boxes, labels = self._get_labels(idx)
            img = self.load_image(fname)

        flipped = False

        if self.augment:
            # MixUp (15% probability)
            if self.mixup_prob > 0 and random.random() < self.mixup_prob:
                img, boxes, labels = self._mixup(img, boxes, labels)

            # CopyPaste (15% probability, mutually exclusive with MixUp for sanity)
            elif self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
                img, boxes, labels = self._copy_paste(img, boxes, labels)

            # Horizontal flip
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                flipped = True
                if boxes.numel() > 0:
                    boxes[:, 0] = self.img_size - boxes[:, 0]

            # HSV / Color Jitter
            if random.random() < 0.3:
                img = TF.adjust_hue(img, random.uniform(-0.1, 0.1))
            if random.random() < 0.3:
                img = TF.adjust_saturation(img, random.uniform(0.5, 1.5))
            if random.random() < 0.3:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))

            # Contrast
            if random.random() < 0.3:
                mean = img.mean(dim=[1, 2], keepdim=True)
                img = ((img - mean) * random.uniform(0.7, 1.3) + mean).clamp(0, 1)

            if random.random() < 0.1:
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.5, 1.5)
                img = TF.gaussian_blur(img, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

            # Cutout: erase a random rectangular patch (30% probability)
            if random.random() < 0.3:
                img_size = self.img_size
                eh = random.randint(img_size // 10, img_size // 4)
                ew = random.randint(img_size // 10, img_size // 4)
                ex = random.randint(0, img_size - ew)
                ey = random.randint(0, img_size - eh)
                img[:, ey:ey+eh, ex:ex+ew] = 0.0

            # Random vertical flip (10% probability)
            if random.random() < 0.1:
                img = torch.flip(img, dims=[1])
                if boxes.numel() > 0:
                    boxes[:, 1] = self.img_size - boxes[:, 1]

        # ImageNet normalization (always applied, after all augmentations)
        img = TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

        if self.return_meta:
            stem = os.path.splitext(fname)[0]
            return img, boxes, labels, stem, flipped
        return img, boxes, labels


def collate_fn(batch):
    images = []
    boxes_list = []
    labels_list = []
    sample_keys = []
    has_meta = len(batch[0]) == 5
    for item in batch:
        if has_meta:
            img, boxes, labels, stem, flipped = item
            sample_keys.append(f"{stem}__flip{int(flipped)}")
        else:
            img, boxes, labels = item
        images.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
    images = torch.stack(images, dim=0)
    if has_meta:
        return images, boxes_list, labels_list, sample_keys
    return images, boxes_list, labels_list
