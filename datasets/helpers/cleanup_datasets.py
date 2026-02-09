#!/usr/bin/env python3
"""
Clean up datasets to keep only essential files for experiments:
- Images (in appropriate folders)
- splits.json
- labels.json
- class_map.json

Move everything else to backup folder.
"""

import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


DATASETS = {
    "AID": {
        "keep_folders": ["images"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "ZooScanNet/ZooScanNet": {
        "keep_folders": ["imgs"],
        "keep_files": ["splits.json", "labels.json", "class_map.json", "filtering_stats.json"],
    },
    "2025_ChestXray14": {
        "keep_folders": ["images"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "neu-surface-defect-database/NEU-DET": {
        "keep_folders": ["train/images", "validation/images"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "rxrx1_v1.0": {
        "keep_folders": ["images"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "pytorch-challange-flower-dataset": {
        "keep_folders": ["dataset"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "resisc45_images": {
        "keep_folders": ["train", "validation", "test"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "pcamv1-20260120T124959Z-3-001": {
        "keep_folders": ["images", "pcamv1"],  # Keep pcamv1 for H5 files if needed
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
    "diabetic-retinopathy-detection": {
        "keep_folders": ["train", "test"],
        "keep_files": ["splits.json", "trainLabels.csv"],  # Keep original labels CSV
    },
    "fgvc-aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data": {
        "keep_folders": ["images"],
        "keep_files": ["splits.json", "labels.json", "class_map.json"],
    },
}


def cleanup_dataset(data_root, dataset_rel_path, config, backup_root, dry_run=False):
    """Clean up a single dataset."""
    dataset_path = Path(data_root) / dataset_rel_path
    if not dataset_path.exists():
        print(f"⚠️  {dataset_rel_path}: Not found, skipping")
        return

    backup_path = Path(backup_root) / dataset_rel_path
    backup_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Processing: {dataset_rel_path}")
    print(f"{'='*80}")

    keep_folders = set(config["keep_folders"])
    keep_files = set(config["keep_files"])

    moved_count = 0
    kept_count = 0

    # Iterate through all items in dataset folder
    for item in tqdm(list(dataset_path.iterdir()), desc="Scanning"):
        item_name = item.name

        # Check if it's a file we want to keep
        if item.is_file() and item_name in keep_files:
            kept_count += 1
            continue

        # Check if it's a folder we want to keep
        if item.is_dir() and item_name in [f.split('/')[0] for f in keep_folders]:
            kept_count += 1
            continue

        # Move to backup
        backup_item = backup_path / item_name
        if dry_run:
            print(f"  [DRY RUN] Would move: {item_name}")
            moved_count += 1
        else:
            try:
                if backup_item.exists():
                    if backup_item.is_dir():
                        shutil.rmtree(backup_item)
                    else:
                        backup_item.unlink()
                shutil.move(str(item), str(backup_item))
                moved_count += 1
            except Exception as e:
                print(f"  ⚠️  Error moving {item_name}: {e}")

    print(f"✓ Kept: {kept_count} items")
    print(f"→ Moved to backup: {moved_count} items")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root folder containing datasets")
    ap.add_argument("--backup_root", required=True, help="Backup folder")
    ap.add_argument("--dry_run", action="store_true", help="Dry run - don't actually move files")
    ap.add_argument("--datasets", nargs="+", help="Specific datasets to clean (default: all)")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    backup_root = Path(args.backup_root)

    if not data_root.exists():
        print(f"Error: Data root {data_root} does not exist")
        return

    backup_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No files will be moved")
        print("=" * 80)

    datasets_to_process = args.datasets if args.datasets else list(DATASETS.keys())

    for dataset_name in datasets_to_process:
        if dataset_name not in DATASETS:
            print(f"⚠️  Unknown dataset: {dataset_name}")
            continue

        cleanup_dataset(
            data_root,
            dataset_name,
            DATASETS[dataset_name],
            backup_root,
            dry_run=args.dry_run
        )

    print(f"\n{'='*80}")
    print(f"✓ Cleanup complete!")
    print(f"  Data: {data_root}")
    print(f"  Backup: {backup_root}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
