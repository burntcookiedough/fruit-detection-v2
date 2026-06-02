"""Fast class distribution analysis."""
import os
from collections import Counter

names = ["apple", "banana", "orange", "mango", "pineapple", "watermelon", "grapes", "pomegranate"]
base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_v4_balanced")

for split in ["train", "valid", "test"]:
    lbl_dir = os.path.join(base, split, "labels")
    if not os.path.isdir(lbl_dir):
        continue
    c = Counter()
    n_files = 0
    n_boxes = 0
    for fname in os.listdir(lbl_dir):
        if not fname.endswith('.txt'):
            continue
        n_files += 1
        with open(os.path.join(lbl_dir, fname)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    c[int(parts[0])] += 1
                    n_boxes += 1
    total = sum(c.values())
    print(f"\n=== {split} ({n_files} images, {n_boxes} boxes) ===")
    for i in range(8):
        cnt = c.get(i, 0)
        print(f"  {i} {names[i]:>12s}: {cnt:>6d} ({100*cnt/max(total,1):5.1f}%)")
    if c:
        mx = max(c.values())
        mn = min(c.values())
        print(f"\n  Max/Min ratio: {mx/max(mn,1):.1f}x")
