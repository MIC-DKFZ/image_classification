#!/usr/bin/env python3
"""
Create a minimal splits.json for AID dataset:
{
  "train": [...image_ids...],
  "val":   [...image_ids...],
  "test":  [...image_ids...]
}

AID dataset has images organized in class folders.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains images/ folder)")
    ap.add_argument("--labels_json", default="labels.json", help="Path to labels.json")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    root = Path(args.root)
    labels_path = root / args.labels_json

    # Load labels
    with open(labels_path, "r", encoding="utf-8") as f:
        labels_dict = json.load(f)

    # Extract image ids and labels
    image_ids = list(labels_dict.keys())
    labels = np.array([labels_dict[img_id] for img_id in image_ids])

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

    out = {"train": train_ids, "val": val_ids, "test": test_ids}

    out_path = root / args.out_json
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_path}")
    print(f"train: {len(train_ids)} images")
    print(f"val:   {len(val_ids)} images")
    print(f"test:  {len(test_ids)} images")


if __name__ == "__main__":
    main()
