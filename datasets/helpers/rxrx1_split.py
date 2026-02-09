#!/usr/bin/env python3
"""
Create splits.json for RxRx1 dataset.

RxRx1 has a metadata.csv that contains:
- site_id (unique ID for each image)
- cell_type (HEPG2, RPE, HUVEC, U2OS)
- dataset (train/test) - already has splits!
- sirna_id (treatment label)

We'll use the existing train/test split and create a validation split from train.
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
    ap.add_argument("--root", required=True, help="Dataset root")
    ap.add_argument("--metadata_csv", default="metadata.csv")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--val_frac", type=float, default=0.25, help="Val fraction from train (0.25 gives 60/20/20)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    csv_path = root / args.metadata_csv

    # Load metadata
    df = pd.read_csv(csv_path)
    df = df[["site_id", "dataset", "sirna_id"]].copy()
    df = df.dropna()

    # Filter to only train data (we'll split it ourselves)
    train_df = df[df["dataset"] == "train"].copy()
    test_df = df[df["dataset"] == "test"].copy()

    # Create class mapping (sirna_id is the label)
    unique_sirnas = sorted(train_df["sirna_id"].unique())
    sirna_to_idx = {sirna: idx for idx, sirna in enumerate(unique_sirnas)}

    train_df["label"] = train_df["sirna_id"].map(sirna_to_idx)

    # Split train into train/val
    site_ids = train_df["site_id"].tolist()
    labels = train_df["label"].to_numpy()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=args.seed)
    train_idx, val_idx = next(sss.split(np.zeros_like(labels), labels))

    train_ids = [site_ids[i] for i in train_idx]
    val_ids = [site_ids[i] for i in val_idx]
    test_ids = test_df["site_id"].tolist()

    # Verify no overlap
    assert set(train_ids).isdisjoint(val_ids)

    # Save splits
    splits_out = {"train": train_ids, "val": val_ids, "test": test_ids}
    splits_path = root / args.out_json
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2)

    # Save labels (only for train and val, test might have different labels)
    labels_dict = dict(zip(train_df["site_id"], train_df["label"]))
    # Add test labels if they exist in mapping
    test_df["label"] = test_df["sirna_id"].map(lambda x: sirna_to_idx.get(x, -1))
    test_labels = dict(zip(test_df["site_id"], test_df["label"]))
    labels_dict.update(test_labels)

    labels_path = root / args.out_labels
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    # Save class mapping (convert keys to strings for JSON)
    class_map_path = root / "class_map.json"
    class_map_serializable = {str(k): int(v) for k, v in sirna_to_idx.items()}
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_map_serializable, f, indent=2)

    print(f"Wrote {splits_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {class_map_path}")
    print(f"train: {len(train_ids)} images")
    print(f"val:   {len(val_ids)} images")
    print(f"test:  {len(test_ids)} images")
    print(f"Number of classes: {len(sirna_to_idx)}")


if __name__ == "__main__":
    main()
