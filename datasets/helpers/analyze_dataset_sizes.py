#!/usr/bin/env python3
"""
Analyze image size distributions across all datasets.
"""

import argparse
from pathlib import Path
from PIL import Image
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import json


def analyze_directory_structure(root, exts={".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}):
    """Analyze images in a directory structure with class folders."""
    root = Path(root)
    class_stats = defaultdict(lambda: {"sizes": [], "count": 0})

    # Find all class folders
    for class_dir in tqdm(list(root.iterdir()), desc="Scanning classes"):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name

        for img_path in class_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    min_dim = min(w, h)
                    class_stats[class_name]["sizes"].append(min_dim)
                    class_stats[class_name]["count"] += 1
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

    return class_stats


def print_statistics(class_stats, dataset_name):
    """Print statistics about image sizes per class."""
    print(f"\n{'='*80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}\n")

    all_sizes = []
    for class_name, stats in sorted(class_stats.items(), key=lambda x: x[1]["count"], reverse=True):
        sizes = stats["sizes"]
        if not sizes:
            continue

        all_sizes.extend(sizes)
        min_size = min(sizes)
        max_size = max(sizes)
        mean_size = np.mean(sizes)
        median_size = np.median(sizes)

        print(f"Class: {class_name:40s} | Count: {stats['count']:5d} | "
              f"Min: {min_size:4d} | Max: {max_size:5d} | "
              f"Mean: {mean_size:6.1f} | Median: {median_size:5.0f}")

    if all_sizes:
        print(f"\n{'-'*80}")
        print(f"Overall Statistics:")
        print(f"  Total images: {len(all_sizes)}")
        print(f"  Min size: {min(all_sizes)}")
        print(f"  Max size: {max(all_sizes)}")
        print(f"  Mean size: {np.mean(all_sizes):.1f}")
        print(f"  Median size: {np.median(all_sizes):.0f}")
        print(f"  Percentiles:")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            print(f"    {p}th: {np.percentile(all_sizes, p):.0f}")
        print(f"  Images < 32px: {sum(1 for s in all_sizes if s < 32)} ({100*sum(1 for s in all_sizes if s < 32)/len(all_sizes):.2f}%)")
        print(f"  Images < 64px: {sum(1 for s in all_sizes if s < 64)} ({100*sum(1 for s in all_sizes if s < 64)/len(all_sizes):.2f}%)")
        print(f"  Images < 128px: {sum(1 for s in all_sizes if s < 128)} ({100*sum(1 for s in all_sizes if s < 128)/len(all_sizes):.2f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=[
        "zooscannet", "aid", "neudet", "flowers102", "resisc45", "chestxray14", "rxrx1"
    ])
    ap.add_argument("--root", required=True, help="Dataset root directory")
    args = ap.parse_args()

    root = Path(args.root)

    if args.dataset == "zooscannet":
        imgs_dir = root / "imgs"
        class_stats = analyze_directory_structure(imgs_dir)
    elif args.dataset == "aid":
        imgs_dir = root / "images"
        class_stats = analyze_directory_structure(imgs_dir)
    elif args.dataset == "neudet":
        # NEU-DET has train and validation folders
        class_stats = defaultdict(lambda: {"sizes": [], "count": 0})
        for split_dir in ["train", "validation"]:
            imgs_dir = root / "NEU-DET" / split_dir / "images"
            if imgs_dir.exists():
                for img_path in tqdm(list(imgs_dir.glob("*.jpg")), desc=f"Scanning {split_dir}"):
                    try:
                        with Image.open(img_path) as img:
                            w, h = img.size
                            min_dim = min(w, h)
                            # Extract class from filename (e.g., "crazing_1.jpg" -> "crazing")
                            class_name = img_path.stem.rsplit("_", 1)[0]
                            class_stats[class_name]["sizes"].append(min_dim)
                            class_stats[class_name]["count"] += 1
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
    elif args.dataset == "flowers102":
        class_stats = defaultdict(lambda: {"sizes": [], "count": 0})
        dataset_dir = root / "dataset"
        for split_name in ["train", "valid", "test"]:
            split_dir = dataset_dir / split_name
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name
                    for img_path in class_dir.iterdir():
                        if not img_path.is_file():
                            continue
                        try:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                min_dim = min(w, h)
                                class_stats[class_name]["sizes"].append(min_dim)
                                class_stats[class_name]["count"] += 1
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")
    elif args.dataset == "resisc45":
        class_stats = defaultdict(lambda: {"sizes": [], "count": 0})
        for split_name in ["train", "validation", "test"]:
            split_dir = root / split_name
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue
                    class_name = class_dir.name
                    for img_path in class_dir.iterdir():
                        if not img_path.is_file():
                            continue
                        try:
                            with Image.open(img_path) as img:
                                w, h = img.size
                                min_dim = min(w, h)
                                class_stats[class_name]["sizes"].append(min_dim)
                                class_stats[class_name]["count"] += 1
                        except Exception as e:
                            print(f"Error reading {img_path}: {e}")

    print_statistics(class_stats, args.dataset.upper())

    # Save to JSON
    output = {
        class_name: {
            "count": stats["count"],
            "min": int(min(stats["sizes"])) if stats["sizes"] else 0,
            "max": int(max(stats["sizes"])) if stats["sizes"] else 0,
            "mean": float(np.mean(stats["sizes"])) if stats["sizes"] else 0,
            "median": float(np.median(stats["sizes"])) if stats["sizes"] else 0,
        }
        for class_name, stats in class_stats.items()
    }

    output_file = root / f"{args.dataset}_size_analysis.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved analysis to {output_file}")


if __name__ == "__main__":
    main()
