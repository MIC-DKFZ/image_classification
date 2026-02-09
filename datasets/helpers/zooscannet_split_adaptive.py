#!/usr/bin/env python3
"""
Create splits.json for ZooScanNet dataset with adaptive size filtering.

Strategy:
- Classes with many samples (>500): Use strict min_size (64px)
- Classes with medium samples (100-500): Use relaxed min_size (48px)
- Classes with few samples (<100): Use lenient min_size (32px)
- Always reject images smaller than absolute_min (e.g., 24px) to avoid extreme upscaling
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
from tqdm import tqdm
from collections import defaultdict


def get_adaptive_min_size(class_count, tier_config):
    """Determine minimum size threshold based on class population."""
    for threshold, min_size in sorted(tier_config.items(), reverse=True):
        if class_count >= threshold:
            return min_size
    return tier_config[min(tier_config.keys())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains imgs/ folder)")
    ap.add_argument("--imgs_dir", default="imgs", help="Images folder")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--absolute_min", type=int, default=24, help="Absolute minimum image size (reject below this)")
    ap.add_argument("--min_samples", type=int, default=50, help="Minimum samples per class after filtering")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    args = ap.parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    # Adaptive tiers: {min_class_size: min_image_size}
    tier_config = {
        500: 64,   # Large classes: strict filtering
        200: 56,   # Medium-large classes
        100: 48,   # Medium classes
        50: 40,    # Small-medium classes
        20: 32,    # Small classes: lenient filtering
    }

    root = Path(args.root)
    imgs_dir = root / args.imgs_dir
    exts = {e.lower() for e in args.exts}

    print("=" * 80)
    print("Phase 1: Initial class population scan")
    print("=" * 80)

    # First pass: count samples per class
    class_folders = [d for d in imgs_dir.iterdir() if d.is_dir()]
    initial_counts = {}

    for class_dir in tqdm(class_folders, desc="Counting samples"):
        class_name = class_dir.name
        count = sum(1 for f in class_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in exts)
        initial_counts[class_name] = count

    print(f"\nFound {len(initial_counts)} classes")
    print(f"Total initial images: {sum(initial_counts.values())}")

    print("\n" + "=" * 80)
    print("Phase 2: Filtering images with adaptive size thresholds")
    print("=" * 80)

    # Second pass: collect images with adaptive filtering
    image_data = []
    class_stats = defaultdict(lambda: {
        "total": 0, "filtered": 0, "rejected_size": 0,
        "min_size_threshold": 0, "sizes": []
    })

    for class_dir in tqdm(class_folders, desc="Filtering images"):
        class_name = class_dir.name
        initial_count = initial_counts[class_name]

        # Determine adaptive min size for this class
        min_size_threshold = get_adaptive_min_size(initial_count, tier_config)
        class_stats[class_name]["min_size_threshold"] = min_size_threshold

        for img_path in class_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue

            class_stats[class_name]["total"] += 1

            # Check image size
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    min_dim = min(w, h)
                    class_stats[class_name]["sizes"].append(min_dim)

                    # Reject if below absolute minimum
                    if min_dim < args.absolute_min:
                        class_stats[class_name]["rejected_size"] += 1
                        continue

                    # Reject if below adaptive threshold
                    if min_dim < min_size_threshold:
                        class_stats[class_name]["rejected_size"] += 1
                        continue

            except Exception as e:
                print(f"\nWarning: Could not read {img_path}: {e}")
                class_stats[class_name]["rejected_size"] += 1
                continue

            # Image passes filters
            rel_path = f"{class_name}/{img_path.name}"
            image_data.append((rel_path, class_name))
            class_stats[class_name]["filtered"] += 1

    print("\n" + "=" * 80)
    print("Filtering Statistics by Class")
    print("=" * 80)
    print(f"{'Class':<30} {'Initial':>7} {'Threshold':>9} {'Kept':>6} {'Rejected':>8} {'Keep%':>6} {'MinSize':>7} {'MeanSize':>8}")
    print("-" * 80)

    for class_name in sorted(class_stats.keys(), key=lambda x: class_stats[x]["filtered"], reverse=True):
        stats = class_stats[class_name]
        keep_pct = 100 * stats["filtered"] / stats["total"] if stats["total"] > 0 else 0
        min_size = min(stats["sizes"]) if stats["sizes"] else 0
        mean_size = np.mean(stats["sizes"]) if stats["sizes"] else 0

        print(f"{class_name:<30} {stats['total']:>7} {stats['min_size_threshold']:>7}px "
              f"{stats['filtered']:>6} {stats['rejected_size']:>8} {keep_pct:>5.1f}% "
              f"{min_size:>7} {mean_size:>8.1f}")

    # Filter classes by minimum samples
    class_counts = defaultdict(int)
    for _, cls in image_data:
        class_counts[cls] += 1

    valid_classes = {c for c, count in class_counts.items() if count >= args.min_samples}
    removed_classes = set(class_counts.keys()) - valid_classes

    if removed_classes:
        print(f"\n⚠ Removing {len(removed_classes)} classes with < {args.min_samples} samples:")
        for cls in sorted(removed_classes):
            print(f"  - {cls}: {class_counts[cls]} samples")

    # Filter images to only valid classes
    filtered_data = [(img_id, cls) for img_id, cls in image_data if cls in valid_classes]

    print(f"\n" + "=" * 80)
    print("Final Dataset Statistics")
    print("=" * 80)
    print(f"Valid classes: {len(valid_classes)}")
    print(f"Total images: {len(filtered_data)}")
    print(f"Images removed: {len(image_data) - len(filtered_data)}")

    # Create class-to-index mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(valid_classes))}

    # Prepare data for splitting
    image_ids = [img_id for img_id, _ in filtered_data]
    labels = np.array([class_to_idx[cls] for _, cls in filtered_data])

    # Stratified split
    print(f"\n" + "=" * 80)
    print("Creating stratified splits (60/20/20)")
    print("=" * 80)

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

    # Save filtering stats
    stats_output = {
        "tier_config": tier_config,
        "absolute_min": args.absolute_min,
        "min_samples_per_class": args.min_samples,
        "total_classes": len(valid_classes),
        "total_images": len(filtered_data),
        "class_stats": {
            cls: {
                "initial_count": class_stats[cls]["total"],
                "kept_count": class_stats[cls]["filtered"],
                "rejected_count": class_stats[cls]["rejected_size"],
                "min_size_threshold": class_stats[cls]["min_size_threshold"],
                "min_image_size": int(min(class_stats[cls]["sizes"])) if class_stats[cls]["sizes"] else 0,
                "mean_image_size": float(np.mean(class_stats[cls]["sizes"])) if class_stats[cls]["sizes"] else 0,
            }
            for cls in valid_classes
        }
    }
    stats_path = root / "filtering_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)

    print(f"\n✓ Wrote {splits_path}")
    print(f"✓ Wrote {labels_path}")
    print(f"✓ Wrote {class_map_path}")
    print(f"✓ Wrote {stats_path}")
    print(f"\nSplit sizes:")
    print(f"  train: {len(train_ids):>6} images")
    print(f"  val:   {len(val_ids):>6} images")
    print(f"  test:  {len(test_ids):>6} images")
    print(f"  total: {len(train_ids) + len(val_ids) + len(test_ids):>6} images")


if __name__ == "__main__":
    main()
