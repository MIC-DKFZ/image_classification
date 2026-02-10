#!/usr/bin/env python3
"""
Create splits.json for Flowers-102 dataset RESPECTING official splits.

Flowers-102 has official train/valid folders. We KEEP these as-is:
- train/ folder → train split
- valid/ folder → split into val + test
- test/ folder is unlabeled (not used)
"""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains dataset/ folder)")
    ap.add_argument("--dataset_dir", default="dataset", help="Dataset folder")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--test_from_valid_frac", type=float, default=0.5, help="Fraction of valid to use as test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"])
    args = ap.parse_args()

    root = Path(args.root)
    dataset_dir = root / args.dataset_dir
    exts = {e.lower() for e in args.exts}

    # Collect images from official train folder
    print("Processing official train folder...")
    train_data = []
    train_dir = dataset_dir / "train"
    if train_dir.exists():
        for class_dir in train_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in exts:
                    continue
                rel_path = f"train/{class_name}/{img_path.name}"
                train_data.append((rel_path, class_name))

    # Collect images from official valid folder (will split into val+test)
    print("Processing official valid folder...")
    valid_data = []
    valid_dir = dataset_dir / "valid"
    if valid_dir.exists():
        for class_dir in valid_dir.iterdir():
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in exts:
                    continue
                rel_path = f"valid/{class_name}/{img_path.name}"
                valid_data.append((rel_path, class_name))

    print(f"Official train images: {len(train_data)}")
    print(f"Official valid images: {len(valid_data)}")

    # Create class-to-index mapping
    all_classes = set(cls for _, cls in train_data + valid_data)
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(all_classes))}
    print(f"Number of classes: {len(class_to_idx)}")

    # Train IDs (use official train as-is)
    train_ids = [img_id for img_id, _ in train_data]

    # Split valid into val + test
    valid_ids = [img_id for img_id, _ in valid_data]
    valid_labels = np.array([class_to_idx[cls] for _, cls in valid_data])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_from_valid_frac, random_state=args.seed)
    val_idx, test_idx = next(sss.split(np.zeros_like(valid_labels), valid_labels))

    val_ids = [valid_ids[i] for i in val_idx]
    test_ids = [valid_ids[i] for i in test_idx]

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
    labels_dict = {img_id: class_to_idx[cls] for img_id, cls in train_data + valid_data}
    labels_path = root / args.out_labels
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    # Save class mapping
    class_map_path = root / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"\nWrote {splits_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {class_map_path}")
    print(f"\n✓ RESPECTING OFFICIAL SPLITS:")
    print(f"  train: {len(train_ids)} images (official train folder)")
    print(f"  val:   {len(val_ids)} images ({(1-args.test_from_valid_frac)*100:.0f}% of official valid)")
    print(f"  test:  {len(test_ids)} images ({args.test_from_valid_frac*100:.0f}% of official valid)")
    print(f"\n⚠️  NO DATA LEAKAGE: Official train/valid boundary preserved!")


if __name__ == "__main__":
    main()
