#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_FILENAME = "sample_per_class_stats.json"
DEFAULT_OUTPUT_FILENAME = "global_train_class_threshold_counts.json"
DEFAULT_DECIMALS = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate train split class counts across datasets into threshold "
            "steps, reporting how many classes have at least N images."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Root directory containing the dataset folders. Defaults to the current directory.",
    )
    parser.add_argument(
        "--input-filename",
        default=DEFAULT_INPUT_FILENAME,
        help=f"Per-dataset stats filename. Defaults to {DEFAULT_INPUT_FILENAME}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT_FILENAME),
        help=f"Output JSON path. Defaults to {DEFAULT_OUTPUT_FILENAME}.",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=10,
        help="Threshold step size. Defaults to 10.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=DEFAULT_DECIMALS,
        help=f"Number of decimal places for percentages. Defaults to {DEFAULT_DECIMALS}.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_dataset_stats(root, input_filename):
    for dataset_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name):
        stats_path = dataset_dir / input_filename
        if not stats_path.exists():
            continue

        data = load_json(stats_path)
        train_counts = data.get("counts_per_split", {}).get("train")
        if not isinstance(train_counts, dict):
            raise ValueError(f"{stats_path} does not contain counts_per_split.train")

        yield dataset_dir.name, train_counts


def build_threshold_summary(dataset_train_counts, step_size, decimals):
    max_count = max(
        (max(train_counts.values()) for _, train_counts in dataset_train_counts if train_counts),
        default=0,
    )
    max_step = (max_count // step_size) * step_size

    summary = {}
    for step in range(0, max_step + step_size, step_size):
        if step > max_step and step != 0:
            break
        summary[str(step)] = {
            dataset_name: round(
                (
                    sum(1 for count in train_counts.values() if count >= step)
                    / max(len(train_counts), 1)
                )
                * 100.0,
                decimals,
            )
            for dataset_name, train_counts in dataset_train_counts
        }
    return summary


def main():
    args = parse_args()
    if args.step_size <= 0:
        raise ValueError("--step-size must be positive")

    root = args.root.resolve()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = root / output_path

    dataset_train_counts = list(iter_dataset_stats(root, args.input_filename))
    if not dataset_train_counts:
        raise ValueError(f"No datasets with {args.input_filename} were found in {root}")

    summary = build_threshold_summary(dataset_train_counts, args.step_size, args.decimals)

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=False)
        handle.write("\n")

    print(f"wrote {output_path}")
    print(f"datasets={len(dataset_train_counts)} steps={len(summary)} step_size={args.step_size}")


if __name__ == "__main__":
    main()
