#!/usr/bin/env python3
"""
Create splits.json for PCam dataset.

PCam has H5 files that need to be extracted first:
- camelyonpatch_level_2_split_train_x.h5 (images)
- camelyonpatch_level_2_split_train_y.h5 (labels)
- camelyonpatch_level_2_split_valid_x.h5
- camelyonpatch_level_2_split_valid_y.h5
- camelyonpatch_level_2_split_test_x.h5.gz (needs unzipping)
- camelyonpatch_level_2_split_test_y.h5.gz (needs unzipping)

This script extracts images from H5 files and creates our 60/20/20 splits.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit


def extract_h5_to_images(h5_path, y_h5_path, output_dir, prefix):
    """Extract images from H5 file and save as PNG files."""
    print(f"Extracting {h5_path}...")

    with h5py.File(h5_path, 'r') as f_x, h5py.File(y_h5_path, 'r') as f_y:
        images = f_x['x'][:]  # Shape: (N, 96, 96, 3)
        labels = f_y['y'][:, 0, 0, 0]  # Shape: (N, 1, 1, 1) -> (N,)

        image_ids = []
        image_labels = []

        output_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(range(len(images))):
            img_array = images[idx]  # (96, 96, 3)
            label = int(labels[idx])

            img_name = f"{prefix}_{idx:06d}.png"
            img_path = output_dir / img_name

            img = Image.fromarray(img_array, mode='RGB')
            img.save(img_path)

            image_ids.append(img_name)
            image_labels.append(label)

    return image_ids, image_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains pcamv1/ folder)")
    ap.add_argument("--pcam_dir", default="pcamv1", help="PCam folder with H5 files")
    ap.add_argument("--images_dir", default="images", help="Output images folder")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_extraction", action="store_true", help="Skip image extraction if already done")
    args = ap.parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    root = Path(args.root)
    pcam_dir = root / args.pcam_dir
    images_dir = root / args.images_dir

    all_image_ids = []
    all_labels = []

    if not args.skip_extraction:
        # Extract train
        train_x_path = pcam_dir / "camelyonpatch_level_2_split_train_x.h5"
        train_y_path = pcam_dir / "camelyonpatch_level_2_split_train_y.h5"
        if train_x_path.exists() and train_y_path.exists():
            train_ids, train_labels = extract_h5_to_images(
                train_x_path, train_y_path, images_dir, "train"
            )
            all_image_ids.extend(train_ids)
            all_labels.extend(train_labels)

        # Extract valid
        valid_x_path = pcam_dir / "camelyonpatch_level_2_split_valid_x.h5"
        valid_y_path = pcam_dir / "camelyonpatch_level_2_split_valid_y.h5"
        if valid_x_path.exists() and valid_y_path.exists():
            valid_ids, valid_labels = extract_h5_to_images(
                valid_x_path, valid_y_path, images_dir, "valid"
            )
            all_image_ids.extend(valid_ids)
            all_labels.extend(valid_labels)

        # Note: test files are .gz and need unzipping first
        # You can unzip them manually: gunzip *.h5.gz
        test_x_path = pcam_dir / "camelyonpatch_level_2_split_test_x.h5"
        test_y_path = pcam_dir / "camelyonpatch_level_2_split_test_y.h5"
        if test_x_path.exists() and test_y_path.exists():
            test_ids, test_labels = extract_h5_to_images(
                test_x_path, test_y_path, images_dir, "test"
            )
            all_image_ids.extend(test_ids)
            all_labels.extend(test_labels)
        else:
            print(f"Warning: Test H5 files not found. Make sure to unzip .gz files first.")
    else:
        # Load from existing images
        print("Loading existing images...")
        for img_path in images_dir.glob("*.png"):
            all_image_ids.append(img_path.name)
            # We'll need to load labels separately if skipping extraction
        print(f"Warning: --skip_extraction requires pre-existing labels.json")

    labels = np.array(all_labels)

    # Stratified split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros_like(labels), labels))

    trainval_ids = [all_image_ids[i] for i in trainval_idx]
    trainval_labels = labels[trainval_idx]
    test_ids = [all_image_ids[i] for i in test_idx]

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
    labels_dict = {img_id: int(label) for img_id, label in zip(all_image_ids, all_labels)}
    labels_path = root / args.out_labels
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    # Save class mapping (binary classification)
    class_map = {"tumor": 1, "normal": 0}
    class_map_path = root / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    print(f"Wrote {splits_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {class_map_path}")
    print(f"train: {len(train_ids)} images")
    print(f"val:   {len(val_ids)} images")
    print(f"test:  {len(test_ids)} images")
    print(f"Class distribution: 0={sum(labels==0)}, 1={sum(labels==1)}")


if __name__ == "__main__":
    main()
