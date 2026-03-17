#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_OUTPUT_FILENAME = "sample_per_class_stats.json"
REQUIRED_FILES = ("labels.json", "class_map.json", "splits.json")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Count the number of samples per class for each split in each "
            "top-level dataset directory."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing the dataset folders. Defaults to the current directory.",
    )
    parser.add_argument(
        "--output-filename",
        default=DEFAULT_OUTPUT_FILENAME,
        help=f"Name of the JSON file to write inside each dataset. Defaults to {DEFAULT_OUTPUT_FILENAME}.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sort_class_items(class_map):
    try:
        return sorted(class_map.items(), key=lambda item: int(item[1]))
    except (TypeError, ValueError):
        return sorted(class_map.items(), key=lambda item: str(item[0]))


def normalize_label(value, id_to_class):
    if isinstance(value, str) and value in id_to_class.values():
        return value
    return id_to_class.get(str(value))


def compute_split_counts(labels, class_map, splits):
    class_items = sort_class_items(class_map)
    id_to_class = {str(class_id): class_name for class_name, class_id in class_items}
    ordered_class_names = [class_name for class_name, _ in class_items]

    split_counts = {}
    missing_samples = {}
    unknown_labels = {}

    for split_name, sample_ids in splits.items():
        counts = {class_name: 0 for class_name in ordered_class_names}
        missing = []
        unknown = []

        for sample_id in sample_ids:
            if sample_id not in labels:
                missing.append(sample_id)
                continue

            class_name = normalize_label(labels[sample_id], id_to_class)
            if class_name is None:
                unknown.append({"sample": sample_id, "label": labels[sample_id]})
                continue

            counts[class_name] += 1

        split_counts[split_name] = counts
        if missing:
            missing_samples[split_name] = missing
        if unknown:
            unknown_labels[split_name] = unknown

    return split_counts, missing_samples, unknown_labels


def dataset_directories(root):
    return sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.name)


def process_dataset(dataset_dir, output_filename):
    required_paths = {name: dataset_dir / name for name in REQUIRED_FILES}
    missing_files = [name for name, path in required_paths.items() if not path.exists()]
    if missing_files:
        return False, f"skipped {dataset_dir.name}: missing {', '.join(missing_files)}"

    labels = load_json(required_paths["labels.json"])
    class_map = load_json(required_paths["class_map.json"])
    splits = load_json(required_paths["splits.json"])

    if not isinstance(labels, dict):
        raise ValueError(f"{dataset_dir / 'labels.json'} must contain a JSON object")
    if not isinstance(class_map, dict):
        raise ValueError(f"{dataset_dir / 'class_map.json'} must contain a JSON object")
    if not isinstance(splits, dict):
        raise ValueError(f"{dataset_dir / 'splits.json'} must contain a JSON object")

    split_counts, missing_samples, unknown_labels = compute_split_counts(labels, class_map, splits)

    output = {
        "dataset": dataset_dir.name,
        "counts_per_split": split_counts,
    }
    if missing_samples:
        output["missing_samples"] = missing_samples
    if unknown_labels:
        output["unknown_labels"] = unknown_labels

    output_path = dataset_dir / output_filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=False)
        handle.write("\n")

    total_counted = {
        split_name: sum(class_counts.values())
        for split_name, class_counts in split_counts.items()
    }
    return True, f"wrote {output_path} ({total_counted})"


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
