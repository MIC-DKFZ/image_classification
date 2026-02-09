#!/usr/bin/env python3
"""
Efficiently reorganize datasets to have clean folder names.
Uses move operations to avoid doubling disk space usage.
"""

import argparse
import shutil
import json
from pathlib import Path


def reorganize_datasets(data_root, dry_run=False):
    """Reorganize all datasets efficiently."""
    data_root = Path(data_root)

    print("="*80)
    print("Dataset Reorganization Plan")
    print("="*80)

    # Step 1: Rename top-level folders to clean names
    renames = [
        ("2025_ChestXray14", "ChestXray14"),
        ("rxrx1_v1.0", "RxRx1"),
        ("pytorch-challange-flower-dataset", "Flowers102"),
        ("resisc45_images", "RESISC45"),
        ("pcamv1-20260120T124959Z-3-001", "PCam"),
        ("diabetic-retinopathy-detection", "DiabeticRetinopathy"),
    ]

    for old_name, new_name in renames:
        old_path = data_root / old_name
        new_path = data_root / new_name

        if old_path.exists():
            print(f"\n{old_name} -> {new_name}")
            if not dry_run:
                if new_path.exists():
                    print(f"  ⚠️  Target already exists, skipping")
                else:
                    old_path.rename(new_path)
                    print(f"  ✓ Renamed")
        else:
            print(f"\n{old_name} -> {new_name} (already done or not found)")

    # Step 2: Flatten nested structures
    print("\n" + "="*80)
    print("Flattening nested structures")
    print("="*80)

    # ZooScanNet: ZooScanNet/ZooScanNet -> ZooScanNet
    zooscan_nested = data_root / "ZooScanNet" / "ZooScanNet"
    zooscan_target = data_root / "ZooScanNet_new"
    if zooscan_nested.exists():
        print(f"\nZooScanNet/ZooScanNet -> ZooScanNet")
        if not dry_run:
            zooscan_nested.rename(zooscan_target)
            shutil.rmtree(data_root / "ZooScanNet")
            zooscan_target.rename(data_root / "ZooScanNet")
            print(f"  ✓ Flattened")
    else:
        print(f"\nZooScanNet (already flattened or not found)")

    # NEU-DET: neu-surface-defect-database/NEU-DET -> NEUDET
    neudet_nested = data_root / "neu-surface-defect-database" / "NEU-DET"
    neudet_target = data_root / "NEUDET"
    if neudet_nested.exists():
        print(f"\nneu-surface-defect-database/NEU-DET -> NEUDET")
        if not dry_run:
            neudet_nested.rename(neudet_target)
            # Remove parent folder
            parent = data_root / "neu-surface-defect-database"
            if parent.exists() and not list(parent.iterdir()):
                parent.rmdir()
            print(f"  ✓ Flattened")
    else:
        print(f"\nNEUDET (already flattened or not found)")

    # FGVCAircraft: fgvc-aircraft/.../data -> FGVCAircraft
    aircraft_nested = data_root / "fgvc-aircraft" / "fgvc-aircraft-2013b" / "fgvc-aircraft-2013b" / "data"
    aircraft_target = data_root / "FGVCAircraft"
    if aircraft_nested.exists():
        print(f"\nfgvc-aircraft/.../data -> FGVCAircraft")
        if not dry_run:
            aircraft_nested.rename(aircraft_target)
            # Remove parent folders
            parent = data_root / "fgvc-aircraft"
            if parent.exists():
                shutil.rmtree(parent)
            print(f"  ✓ Flattened")
    else:
        print(f"\nFGVCAircraft (already flattened or not found)")

    # Step 3: Rename image folders
    print("\n" + "="*80)
    print("Renaming image folders")
    print("="*80)

    # ZooScanNet: imgs -> images
    zooscan_imgs = data_root / "ZooScanNet" / "imgs"
    if zooscan_imgs.exists():
        print(f"\nZooScanNet: imgs -> images")
        if not dry_run:
            zooscan_imgs.rename(data_root / "ZooScanNet" / "images")
            print(f"  ✓ Renamed")
    else:
        print(f"\nZooScanNet/images (already renamed or not found)")

    # Flowers102: dataset -> images
    flowers_dataset = data_root / "Flowers102" / "dataset"
    if flowers_dataset.exists():
        print(f"\nFlowers102: dataset -> images")
        if not dry_run:
            flowers_dataset.rename(data_root / "Flowers102" / "images")
            print(f"  ✓ Renamed")
    else:
        print(f"\nFlowers102/images (already renamed or not found)")

    # Step 4: Verify final structure
    print("\n" + "="*80)
    print("Final Dataset Structure")
    print("="*80)

    expected_datasets = [
        "AID",
        "ZooScanNet",
        "ChestXray14",
        "NEUDET",
        "RxRx1",
        "Flowers102",
        "RESISC45",
        "PCam",
        "DiabeticRetinopathy",
        "FGVCAircraft"
    ]

    for dataset in expected_datasets:
        dataset_path = data_root / dataset
        if dataset_path.exists():
            # Find image folders
            image_folders = []
            for item in dataset_path.iterdir():
                if item.is_dir() and item.name in ["images", "imgs", "dataset", "train", "test", "validation"]:
                    image_folders.append(item.name)

            # Find JSON files
            json_files = [f.name for f in dataset_path.glob("*.json")]

            print(f"\n✓ {dataset}/")
            print(f"    Image folders: {', '.join(sorted(image_folders))}")
            print(f"    JSON files: {', '.join(sorted(json_files))}")
        else:
            print(f"\n⚠️  {dataset}/ - NOT FOUND")

    print("\n" + "="*80)
    print("✓ Reorganization complete!")
    print("="*80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Root folder containing datasets")
    ap.add_argument("--dry_run", action="store_true", help="Dry run - don't actually move files")
    args = ap.parse_args()

    if args.dry_run:
        print("DRY RUN MODE - No files will be moved\n")

    reorganize_datasets(args.data_root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
