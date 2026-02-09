#!/usr/bin/env python3
"""
Create splits.json for FGVC-Aircraft dataset.

FGVC-Aircraft already has train.csv, val.csv, test.csv with splits.
We'll use these existing splits but convert to our format and create 60/20/20 from scratch
by merging all data and re-splitting.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains train.csv, val.csv, test.csv)")
    ap.add_argument("--aircraft_dir", default="fgvc-aircraft-2013b/fgvc-aircraft-2013b/data",
                    help="FGVC-Aircraft data folder")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    root = Path(args.root)
    aircraft_dir = root / args.aircraft_dir

    # Load existing CSVs
    train_csv = root / "train.csv"
    val_csv = root / "val.csv"
    test_csv = root / "test.csv"

    dfs = []
    for csv_path in [train_csv, val_csv, test_csv]:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            dfs.append(df)
        else:
            print(f"Warning: {csv_path} not found")

    if not dfs:
        raise FileNotFoundError("No CSV files found")

    # Merge all data
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df[["filename", "Classes"]].copy()
    all_df.columns = ["filename", "class"]
    all_df = all_df.dropna()

    # Add .jpg extension if not present
    all_df["filename"] = all_df["filename"].apply(lambda x: x if x.endswith(".jpg") else f"{x}.jpg")

    # Create class mapping
    unique_classes = sorted(all_df["class"].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    all_df["label"] = all_df["class"].map(class_to_idx)

    print(f"Total images: {len(all_df)}")
    print(f"Number of classes: {len(class_to_idx)}")

    # Prepare for splitting
    image_ids = all_df["filename"].tolist()
    labels = all_df["label"].to_numpy()

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
    splits_path = aircraft_dir / args.out_json
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2)

    # Save labels
    labels_dict = dict(zip(all_df["filename"], all_df["label"]))
    labels_path = aircraft_dir / args.out_labels
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    # Save class mapping
    class_map_path = aircraft_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Wrote {splits_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {class_map_path}")
    print(f"train: {len(train_ids)} images")
    print(f"val:   {len(val_ids)} images")
    print(f"test:  {len(test_ids)} images")


if __name__ == "__main__":
    main()
