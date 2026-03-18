#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path


DEFAULT_ROOT = Path("synergy_unit/data/datasets")
DEFAULT_OUTPUT_DIRNAME = "subsets"
REQUIRED_FILES = (
    "sample_per_class_stats.json",
    "labels.json",
    "class_map.json",
    "splits.json",
)
SAMPLES_PER_CLASS_VALUES = [10, 20, 30, 40, 50]
NUM_TRIALS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate subset JSON files with actual train-split sample paths for "
            "fixed samples-per-class configurations."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Root directory containing dataset folders. Defaults to {DEFAULT_ROOT}.",
    )
    parser.add_argument(
        "--output-dirname",
        default=DEFAULT_OUTPUT_DIRNAME,
        help=(
            "Name of the output directory created inside each dataset folder. "
            f"Defaults to {DEFAULT_OUTPUT_DIRNAME}."
        ),
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_directories(root):
    return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)


def sort_class_items(class_map):
    try:
        return sorted(class_map.items(), key=lambda item: int(item[1]))
    except (TypeError, ValueError):
        return sorted(class_map.items(), key=lambda item: str(item[0]))


def normalize_label(value, id_to_class):
    if isinstance(value, str) and value in id_to_class.values():
        return value
    return id_to_class.get(str(value))


def build_train_samples_by_class(labels, class_map, splits):
    class_items = sort_class_items(class_map)
    id_to_class = {str(class_id): class_name for class_name, class_id in class_items}
    ordered_class_names = [class_name for class_name, _ in class_items]
    samples_by_class = {class_name: [] for class_name in ordered_class_names}

    for sample_id in splits["train"]:
        if sample_id not in labels:
            raise ValueError(f"Train sample '{sample_id}' is missing from labels.json")

        class_name = normalize_label(labels[sample_id], id_to_class)
        if class_name is None:
            raise ValueError(
                f"Could not map label '{labels[sample_id]}' for train sample '{sample_id}'"
            )

        samples_by_class[class_name].append(sample_id)

    return samples_by_class, ordered_class_names


def build_subset_split_payload(splits, samples_per_class, ordered_class_names, samples_by_class, rng):
    selected_train_samples = []
    for class_name in ordered_class_names:
        class_samples = samples_by_class[class_name]
        if len(class_samples) < samples_per_class:
            raise ValueError(
                f"class '{class_name}' has only {len(class_samples)} "
                f"train samples, but {samples_per_class} were requested"
            )
        selected_train_samples.extend(rng.sample(class_samples, samples_per_class))

    rng.shuffle(selected_train_samples)
    return {
        "train": selected_train_samples,
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }


def process_dataset(dataset_dir, output_dirname):
    required_paths = {name: dataset_dir / name for name in REQUIRED_FILES}
    missing_files = [name for name, path in required_paths.items() if not path.exists()]
    if missing_files:
        return False, f"skipped {dataset_dir.name}: missing {', '.join(missing_files)}"

    sample_stats = load_json(required_paths["sample_per_class_stats.json"])
    labels = load_json(required_paths["labels.json"])
    class_map = load_json(required_paths["class_map.json"])
    splits = load_json(required_paths["splits.json"])

    if not isinstance(sample_stats, dict):
        raise ValueError(f"{required_paths['sample_per_class_stats.json']} must contain a JSON object")
    if not isinstance(labels, dict):
        raise ValueError(f"{required_paths['labels.json']} must contain a JSON object")
    if not isinstance(class_map, dict):
        raise ValueError(f"{required_paths['class_map.json']} must contain a JSON object")
    if not isinstance(splits, dict):
        raise ValueError(f"{required_paths['splits.json']} must contain a JSON object")
    if "train" not in splits or not isinstance(splits["train"], list):
        raise ValueError(f"{required_paths['splits.json']} must contain a train split array")

    train_counts = sample_stats.get("counts_per_split", {}).get("train")
    if not isinstance(train_counts, dict):
        raise ValueError(
            f"{required_paths['sample_per_class_stats.json']} must contain counts_per_split.train"
        )

    samples_by_class, ordered_class_names = build_train_samples_by_class(labels, class_map, splits)
    output_dir = dataset_dir / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in output_dir.glob("samples_per_class_*.json"):
        existing_file.unlink()

    supported_values = []
    skipped_values = []

    for samples_per_class in SAMPLES_PER_CLASS_VALUES:
        is_supported = True
        for class_name in ordered_class_names:
            if train_counts.get(class_name, 0) < samples_per_class:
                is_supported = False
                break

        if not is_supported:
            skipped_values.append(samples_per_class)
            continue

        for trial_index in range(NUM_TRIALS):
            rng = random.Random(f"{dataset_dir.name}:{samples_per_class}:{trial_index}")
            payload = build_subset_split_payload(
                splits=splits,
                samples_per_class=samples_per_class,
                ordered_class_names=ordered_class_names,
                samples_by_class=samples_by_class,
                rng=rng,
            )
            output_path = output_dir / f"samples_per_class_{samples_per_class}_trial_{trial_index}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=False)
                handle.write("\n")
        supported_values.append(samples_per_class)

    if not supported_values:
        return False, f"skipped {dataset_dir.name}: no requested samples_per_class values are supported"

    total_written = len(supported_values) * NUM_TRIALS
    message = f"wrote {output_dir} ({total_written} subset files)"
    if skipped_values:
        message += f"; skipped unsupported values {skipped_values}"
    return True, message


def main():
    args = parse_args()
    root = args.root.resolve()

    processed = 0
    skipped = 0

    for dataset_dir in dataset_directories(root):
        success, message = process_dataset(dataset_dir, args.output_dirname)
        print(message)
        if success:
            processed += 1
        else:
            skipped += 1

    print(f"processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
