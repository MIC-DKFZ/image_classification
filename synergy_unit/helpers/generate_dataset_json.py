#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_ROOT = Path("synergy_unit/data/datasets")
DEFAULT_OUTPUT_FILENAME = "dataset.json"
REQUIRED_FILE = "splits.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate dataset.json files containing image counts per split and "
            "the total number of unique images for each dataset directory."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Root directory containing dataset folders. "
            f"Defaults to {DEFAULT_ROOT}."
        ),
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help=(
            "Name of the output JSON file to write inside each dataset directory. "
            f"Defaults to {DEFAULT_OUTPUT_FILENAME}."
        ),
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_directories(root):
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def validate_splits(dataset_dir, splits):
    if not isinstance(splits, dict):
        raise ValueError(f"{dataset_dir / REQUIRED_FILE} must contain a JSON object")

    for split_name, sample_ids in splits.items():
        if not isinstance(sample_ids, list):
            raise ValueError(
                f"{dataset_dir / REQUIRED_FILE} split '{split_name}' must contain a JSON array"
            )


def build_dataset_summary(dataset_dir, splits):
    counts_per_split = {}
    unique_samples = set()

    for split_name, sample_ids in splits.items():
        counts_per_split[split_name] = len(sample_ids)
        unique_samples.update(sample_ids)

    return {
        "dataset": dataset_dir.name,
        "num_images_per_split": counts_per_split,
        "total_num_images": len(unique_samples),
    }


def process_dataset(dataset_dir, output_filename):
    splits_path = dataset_dir / REQUIRED_FILE
    if not splits_path.exists():
        return False, f"skipped {dataset_dir.name}: missing {REQUIRED_FILE}"

    splits = load_json(splits_path)
    validate_splits(dataset_dir, splits)

    output = build_dataset_summary(dataset_dir, splits)
    output_path = dataset_dir / output_filename

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=False)
        handle.write("\n")

    return True, f"wrote {output_path}"


def main():
    args = parse_args()
    root = args.root.resolve()

    processed = 0
    skipped = 0

    for dataset_dir in dataset_directories(root):
        success, message = process_dataset(dataset_dir, args.output_filename)
        print(message)
        if success:
            processed += 1
        else:
            skipped += 1

    print(f"processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
