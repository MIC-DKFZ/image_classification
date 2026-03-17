#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_PATH = Path("synergy_unit/data/epoch_timings.json")
DEFAULT_OUTPUT_PATH = Path("synergy_unit/data/dataset_mean_epoch_times.json")
TARGET_DATA_FRACTION = "1.0"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute one mean epoch time per dataset from epoch_timings.json, "
            "using only entries for data_fraction=1.0."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to epoch_timings.json. Defaults to {DEFAULT_INPUT_PATH}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the output JSON file. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = json.loads(args.input_path.read_text(encoding="utf-8"))

    dataset_means: dict[str, float | None] = {}

    for dataset in source["datasets"]:
        epoch_seconds = []
        for model in source["models"]:
            for peft in source["pefts"]:
                entry = source["timings"][model][dataset][peft][TARGET_DATA_FRACTION]
                if entry is not None:
                    epoch_seconds.append(entry["epoch_seconds"])

        dataset_means[dataset] = (
            round(sum(epoch_seconds) / len(epoch_seconds), 4) if epoch_seconds else None
        )

    output = {
        "source_file": str(args.input_path),
        "data_fraction": TARGET_DATA_FRACTION,
        "mean_epoch_seconds_per_dataset": dataset_means,
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"wrote {args.output_path}")


if __name__ == "__main__":
    main()
