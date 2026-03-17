#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MEAN_TIMINGS_PATH = Path("synergy_unit/data/dataset_mean_epoch_times.json")
DEFAULT_DATASETS_ROOT = Path("synergy_unit/data/datasets")
DEFAULT_OUTPUT_PATH = Path("synergy_unit/data/dataset_mean_epoch_split_times.json")
DATASET_NAME_ALIASES = {
    "diabetic_retina": "diabeticretinopathy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split each dataset mean epoch time into train and val components "
            "using the train/val sample ratio from each dataset.json file."
        )
    )
    parser.add_argument(
        "--mean-timings-path",
        type=Path,
        default=DEFAULT_MEAN_TIMINGS_PATH,
        help=f"Path to dataset_mean_epoch_times.json. Defaults to {DEFAULT_MEAN_TIMINGS_PATH}.",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT,
        help=f"Root containing dataset folders with dataset.json files. Defaults to {DEFAULT_DATASETS_ROOT}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the output JSON file. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def index_dataset_jsons(datasets_root: Path) -> dict[str, tuple[Path, dict]]:
    dataset_index: dict[str, tuple[Path, dict]] = {}

    for dataset_json_path in sorted(datasets_root.glob("*/dataset.json")):
        payload = load_json(dataset_json_path)
        dataset_dir_name = dataset_json_path.parent.name
        keys = {
            normalize_name(dataset_dir_name),
            normalize_name(payload["dataset"]),
        }
        for key in keys:
            dataset_index[key] = (dataset_json_path, payload)

    return dataset_index


def build_output(
    mean_timings: dict,
    dataset_index: dict[str, tuple[Path, dict]],
    mean_timings_path: Path,
) -> dict:
    split_timings = {}

    for dataset_name, mean_epoch_seconds in mean_timings["mean_epoch_seconds_per_dataset"].items():
        key = DATASET_NAME_ALIASES.get(dataset_name, normalize_name(dataset_name))
        if key not in dataset_index:
            raise ValueError(f"No dataset.json found for dataset '{dataset_name}'")

        dataset_json_path, dataset_payload = dataset_index[key]
        split_counts = dataset_payload["num_images_per_split"]
        train_count = split_counts["train"]
        val_count = split_counts["val"]
        train_val_total = train_count + val_count

        train_ratio = train_count / train_val_total
        val_ratio = val_count / train_val_total
        train_epoch_seconds = mean_epoch_seconds * train_ratio
        val_epoch_seconds = mean_epoch_seconds * val_ratio

        split_timings[dataset_name] = {
            "mean_epoch_seconds": mean_epoch_seconds,
            "train_epoch_seconds": round(train_epoch_seconds, 4),
            "val_epoch_seconds": round(val_epoch_seconds, 4),
            "mean_epoch_seconds_per_image": round(mean_epoch_seconds / train_val_total, 8),
            "train_epoch_seconds_per_image": round(train_epoch_seconds / train_count, 8),
            "val_epoch_seconds_per_image": round(val_epoch_seconds / val_count, 8),
            "train_val_ratio": {
                "train": round(train_ratio, 6),
                "val": round(val_ratio, 6),
            },
            "train_val_counts": {
                "train": train_count,
                "val": val_count,
            },
            "dataset_json": str(dataset_json_path),
        }

    return {
        "source_mean_timings_file": mean_timings["source_file"],
        "source_dataset_mean_file": str(mean_timings_path),
        "data_fraction": mean_timings["data_fraction"],
        "split_basis": "train_val_ratio_from_dataset_json",
        "mean_epoch_seconds_per_dataset": split_timings,
    }


def main():
    args = parse_args()
    mean_timings = load_json(args.mean_timings_path)
    dataset_index = index_dataset_jsons(args.datasets_root)
    output = build_output(mean_timings, dataset_index, args.mean_timings_path)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
