#!/usr/bin/env python3
"""
Create splits.json for ZooScanNet dataset.

ZooScanNet has images organized in class folders under imgs/
We need to:
1. Filter images by size (remove too small images)
2. Filter classes by population (remove classes with too few samples)
3. Create 60/20/20 stratified splits
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains imgs/ folder)")
    ap.add_argument("--imgs_dir", default="imgs", help="Images folder")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--min_size", type=int, default=32, help="Minimum image dimension")
    ap.add_argument("--min_samples", type=int, default=100, help="Minimum samples per class")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    args = ap.parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    root = Path(args.root)
    imgs_dir = root / args.imgs_dir
    exts = {e.lower() for e in args.exts}

    # Collect all images from class folders
    print("Scanning class folders...")
    class_folders = [d for d in imgs_dir.iterdir() if d.is_dir()]

    image_data = []
    class_counts = {}

    for class_dir in tqdm(class_folders):
        class_name = class_dir.name
        class_counts[class_name] = 0

        for img_path in class_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue

            # Check image size
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if min(w, h) < args.min_size:
                        continue
            except Exception as e:
                print(f"Warning: Could not read {img_path}: {e}")
                continue

            # Store relative path from imgs_dir
            rel_path = f"{class_name}/{img_path.name}"
            image_data.append((rel_path, class_name))
            class_counts[class_name] += 1

    # Filter classes by population
    valid_classes = {c for c, count in class_counts.items() if count >= args.min_samples}
    print(f"Total classes: {len(class_counts)}, Valid classes (>={args.min_samples} samples): {len(valid_classes)}")

    # Filter images
    filtered_data = [(img_id, cls) for img_id, cls in image_data if cls in valid_classes]
    print(f"Total images after filtering: {len(filtered_data)}")

    # Create class-to-index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}

    # Prepare data for splitting
    image_ids = [img_id for img_id, _ in filtered_data]
    labels = np.array([class_to_idx[cls] for _, cls in filtered_data])

    # Stratified split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros_like(labels), labels))

    trainval_ids = [image_ids[i] for i in trainval_idx]
    trainval_labels = labels[trainval_idx]
    test_ids = [image_ids[i] for i in test_idx]

    # Split train vs val
    val_rel = args.val_frac / (args.train_frac + args.val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=args.seed + 1)
    train_idx, val_idx = next(sss2.split(np.zeros_like(trainval_labels), trainval_labels))

    train_ids = [trainval_ids[i] for i in train_idx]
    val_ids = [trainval_ids[i] for i in val_idx]

    # Verify no overlap
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    # Save splits
    splits_out = {"train": train_ids, "val": val_ids, "test": test_ids}
    splits_path = root / args.out_json
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2)

    # Save labels
    labels_dict = {img_id: class_to_idx[cls] for img_id, cls in filtered_data}
    labels_path = root / args.out_labels
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    # Save class mapping
    class_map_path = root / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Wrote {splits_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {class_map_path}")
    print(f"train: {len(train_ids)} images")
    print(f"val:   {len(val_ids)} images")
    print(f"test:  {len(test_ids)} images")
    print(f"Number of classes: {len(class_to_idx)}")


if __name__ == "__main__":
    main()
