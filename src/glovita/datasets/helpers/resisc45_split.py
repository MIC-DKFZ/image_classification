#!/usr/bin/env python3
"""
Create splits.json for RESISC45 dataset RESPECTING official splits.

RESISC45 has official train/validation/test folders. We MUST use these as-is
to prevent test data leakage!

CRITICAL: Do NOT merge and re-split! This would leak test samples into training.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root (contains train/validation/test folders)")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png"])
    args = ap.parse_args()

    root = Path(args.root)
    exts = {e.lower() for e in args.exts}

    # Collect images from OFFICIAL splits (DO NOT MERGE!)
    image_data = {}  # split_name -> list of (path, class)

    for split_name in ["train", "validation", "test"]:
        split_dir = root / split_name
        if not split_dir.exists():
            print(f"ERROR: {split_dir} does not exist!")
            print(f"RESISC45 requires official train/validation/test folders.")
            raise FileNotFoundError(f"Missing {split_name} folder")

        split_data = []
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            for img_path in class_dir.iterdir():
                if not img_path.is_file() or img_path.suffix.lower() not in exts:
                    continue

                # Store as relative path from root
                rel_path = f"{split_name}/{class_name}/{img_path.name}"
                split_data.append((rel_path, class_name))

        image_data[split_name] = split_data
        print(f"Official {split_name}: {len(split_data)} images")

    # Create class-to-index mapping
    all_images = []
    for split_data in image_data.values():
        all_images.extend(split_data)

    unique_classes = sorted(set(cls for _, cls in all_images))
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
    print(f"Number of classes: {len(class_to_idx)}")

    # Create splits using OFFICIAL boundaries
    train_ids = [img_id for img_id, _ in image_data["train"]]
    val_ids = [img_id for img_id, _ in image_data["validation"]]
    test_ids = [img_id for img_id, _ in image_data["test"]]

    # CRITICAL VERIFICATION: Ensure no overlap!
    assert set(train_ids).isdisjoint(val_ids), "Train/Val overlap detected!"
    assert set(train_ids).isdisjoint(test_ids), "Train/Test overlap detected!"
    assert set(val_ids).isdisjoint(test_ids), "Val/Test overlap detected!"
    print("✓ Verified: No data leakage between official splits")

    # Save splits (using official split names)
    splits_out = {
        "train": train_ids,
        "val": val_ids,  # Note: official "validation" → "val" for consistency
        "test": test_ids
    }
    splits_path = root / args.out_json
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2)

    # Save labels
    labels_dict = {img_id: class_to_idx[cls] for img_id, cls in all_images}
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
    print(f"\n✓ USING OFFICIAL RESISC45 SPLITS:")
    print(f"  train: {len(train_ids)} images")
    print(f"  val:   {len(val_ids)} images")
    print(f"  test:  {len(test_ids)} images")
    print(f"\n⚠️  CRITICAL: Official test split preserved - NO DATA LEAKAGE!")


if __name__ == "__main__":
    main()
