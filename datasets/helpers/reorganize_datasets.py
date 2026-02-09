#!/usr/bin/env python3
"""
Reorganize datasets to have clean folder names and consistent structure:
- Clean dataset name (no prefixes, suffixes, or nesting)
- Single 'images' folder inside
- splits.json, labels.json, class_map.json at root level
"""

import argparse
import shutil
import json
from pathlib import Path
from tqdm import tqdm


DATASET_REORGANIZATION = {
    "AID": {
        "old_path": "AID",
        "new_name": "AID",
        "image_folders": ["images"],  # Already correct
        "action": "keep",  # No change needed
    },
    "ZooScanNet": {
        "old_path": "ZooScanNet/ZooScanNet",
        "new_name": "ZooScanNet",
        "image_folders": ["imgs"],
        "rename_images_to": "images",  # Rename imgs -> images
        "action": "flatten_and_rename",
    },
    "ChestXray14": {
        "old_path": "2025_ChestXray14",
        "new_name": "ChestXray14",
        "image_folders": ["images"],
        "action": "rename",
    },
    "NEUDET": {
        "old_path": "neu-surface-defect-database/NEU-DET",
        "new_name": "NEUDET",
        "image_folders": ["train/images", "validation/images"],
        "merge_to": "images",  # Merge both into single images folder
        "action": "flatten_and_merge",
    },
    "RxRx1": {
        "old_path": "rxrx1_v1.0",
        "new_name": "RxRx1",
        "image_folders": ["images"],
        "action": "rename",
    },
    "Flowers102": {
        "old_path": "pytorch-challange-flower-dataset",
        "new_name": "Flowers102",
        "image_folders": ["dataset"],
        "rename_images_to": "images",  # Rename dataset -> images
        "action": "rename_and_rename_images",
    },
    "RESISC45": {
        "old_path": "resisc45_images",
        "new_name": "RESISC45",
        "image_folders": ["train", "validation", "test"],
        "merge_to": "images",  # Merge all into single images folder
        "action": "rename_and_merge",
    },
    "PCam": {
        "old_path": "pcamv1-20260120T124959Z-3-001",
        "new_name": "PCam",
        "image_folders": ["images", "pcamv1"],  # Keep both for now
        "action": "rename",
    },
    "DiabeticRetinopathy": {
        "old_path": "diabetic-retinopathy-detection",
        "new_name": "DiabeticRetinopathy",
        "image_folders": ["train", "test"],
        "merge_to": "images",  # Merge into single images folder
        "action": "rename_and_merge",
    },
    "FGVCAircraft": {
        "old_path": "fgvc-aircraft/fgvc-aircraft-2013b/fgvc-aircraft-2013b/data",
        "new_name": "FGVCAircraft",
        "image_folders": ["images"],
        "action": "flatten",
    },
}


def merge_image_folders(source_folders, target_folder, data_root, dry_run=False):
    """Merge multiple image folders into a single one, preserving structure."""
    print(f"  Merging image folders into: {target_folder.name}")

    for source_rel in source_folders:
        source = data_root / source_rel
        if not source.exists():
            print(f"    ⚠️  Source not found: {source_rel}")
            continue

        # Count items to merge
        items = list(source.rglob("*"))
        image_files = [f for f in items if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']]

        print(f"    Merging {len(image_files)} images from: {source_rel}")

        if dry_run:
            continue

        # Copy all contents preserving structure
        if source.is_dir():
            for item in tqdm(list(source.rglob("*")), desc=f"    Copying from {source.name}"):
                if item.is_file():
                    rel_path = item.relative_to(source)
                    target_file = target_folder / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_file)


def reorganize_dataset(data_root, dataset_name, config, dry_run=False):
    """Reorganize a single dataset."""
    old_path = data_root / config["old_path"]
    new_name = config["new_name"]
    new_path = data_root / new_name
    action = config["action"]

    if not old_path.exists():
        print(f"⚠️  {dataset_name}: Source not found at {config['old_path']}")
        return

    print(f"\n{'='*80}")
    print(f"Reorganizing: {dataset_name}")
    print(f"  From: {config['old_path']}")
    print(f"  To: {new_name}")
    print(f"  Action: {action}")
    print(f"{'='*80}")

    if action == "keep":
        print("  ✓ Already in correct structure, no changes needed")
        return

    if dry_run:
        print(f"  [DRY RUN] Would reorganize {old_path} -> {new_path}")
        return

    # Create temporary directory for reorganization
    temp_path = data_root / f"_temp_{new_name}"
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True)

    # Handle different actions
    if action == "flatten_and_rename":
        # Move JSON files
        for json_file in ["splits.json", "labels.json", "class_map.json", "filtering_stats.json"]:
            src = old_path / json_file
            if src.exists():
                shutil.copy2(src, temp_path / json_file)

        # Rename image folder
        old_img = old_path / config["image_folders"][0]
        new_img = temp_path / config["rename_images_to"]
        if old_img.exists():
            shutil.copytree(old_img, new_img)

    elif action == "rename":
        # Simple rename - just move everything
        for item in old_path.iterdir():
            shutil.move(str(item), str(temp_path / item.name))

    elif action == "flatten_and_merge":
        # Move JSON files from nested location
        for json_file in ["splits.json", "labels.json", "class_map.json"]:
            src = old_path / json_file
            if src.exists():
                shutil.copy2(src, temp_path / json_file)

        # Merge image folders
        target_img = temp_path / config["merge_to"]
        target_img.mkdir(parents=True)
        merge_image_folders(
            [old_path / f for f in config["image_folders"]],
            target_img,
            old_path.parent,
            dry_run=False
        )

    elif action == "rename_and_rename_images":
        # Move JSON files
        for json_file in ["splits.json", "labels.json", "class_map.json"]:
            src = old_path / json_file
            if src.exists():
                shutil.copy2(src, temp_path / json_file)

        # Rename image folder
        old_img = old_path / config["image_folders"][0]
        new_img = temp_path / config["rename_images_to"]
        if old_img.exists():
            shutil.move(str(old_img), str(new_img))

    elif action == "rename_and_merge":
        # Move JSON files
        for json_file in ["splits.json", "labels.json", "class_map.json", "trainLabels.csv"]:
            src = old_path / json_file
            if src.exists():
                shutil.copy2(src, temp_path / json_file)

        # Merge image folders
        target_img = temp_path / config["merge_to"]
        target_img.mkdir(parents=True)
        merge_image_folders(
            [old_path / f for f in config["image_folders"]],
            target_img,
            old_path.parent,
            dry_run=False
        )

    elif action == "flatten":
        # Extract from nested structure
        for json_file in ["splits.json", "labels.json", "class_map.json"]:
            src = old_path / json_file
            if src.exists():
                shutil.copy2(src, temp_path / json_file)

        # Move image folder
        old_img = old_path / config["image_folders"][0]
        new_img = temp_path / config["image_folders"][0]
        if old_img.exists():
            shutil.copytree(old_img, new_img)

    # Remove old path
    if old_path.exists():
        # For nested paths, only remove the deepest part
        if "/" in config["old_path"]:
            # Remove parent directories if empty
            parts = config["old_path"].split("/")
            for i in range(len(parts), 0, -1):
                path_to_remove = data_root / "/".join(parts[:i])
                if path_to_remove.exists():
                    try:
                        if path_to_remove.is_dir() and not list(path_to_remove.iterdir()):
                            path_to_remove.rmdir()
                        else:
                            shutil.rmtree(path_to_remove)
                    except Exception as e:
                        print(f"    ⚠️  Could not remove {path_to_remove}: {e}")
        else:
            shutil.rmtree(old_path)

    # Move temp to final location
    if new_path.exists():
        shutil.rmtree(new_path)
    shutil.move(str(temp_path), str(new_path))

    print(f"  ✓ Reorganized successfully")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root folder containing datasets")
    ap.add_argument("--dry_run", action="store_true", help="Dry run - don't actually move files")
    ap.add_argument("--datasets", nargs="+", help="Specific datasets to reorganize (default: all)")
    args = ap.parse_args()

    data_root = Path(args.data_root)

    if not data_root.exists():
        print(f"Error: Data root {data_root} does not exist")
        return

    if args.dry_run:
        print("=" * 80)
        print("DRY RUN MODE - No files will be moved")
        print("=" * 80)

    datasets_to_process = args.datasets if args.datasets else list(DATASET_REORGANIZATION.keys())

    for dataset_name in datasets_to_process:
        if dataset_name not in DATASET_REORGANIZATION:
            print(f"⚠️  Unknown dataset: {dataset_name}")
            continue

        reorganize_dataset(
            data_root,
            dataset_name,
            DATASET_REORGANIZATION[dataset_name],
            dry_run=args.dry_run
        )

    print(f"\n{'='*80}")
    print(f"✓ Reorganization complete!")
    print(f"\nFinal structure in {data_root}:")
    if not args.dry_run:
        for dataset_name in sorted(DATASET_REORGANIZATION.keys()):
            new_path = data_root / DATASET_REORGANIZATION[dataset_name]["new_name"]
            if new_path.exists():
                print(f"  ✓ {DATASET_REORGANIZATION[dataset_name]['new_name']}/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
