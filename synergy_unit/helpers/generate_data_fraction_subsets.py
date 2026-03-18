#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path


DEFAULT_ROOT = Path("synergy_unit/data/datasets")
DEFAULT_OUTPUT_DIRNAME = "subsets"
DEFAULT_MIN_IMAGES_PER_CLASS = 20
DATA_FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_TRIALS = 5


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate split-style subset JSON files for fixed data fractions while "
            "preserving train class balance and enforcing a minimum number of "
            "images per class."
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
    parser.add_argument(
        "--min-images-per-class",
        type=int,
        default=DEFAULT_MIN_IMAGES_PER_CLASS,
        help=(
            "Minimum number of train images to keep per class in each subset. "
            f"Defaults to {DEFAULT_MIN_IMAGES_PER_CLASS}."
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


def sort_class_names(class_names):
    try:
        return sorted(class_names, key=lambda value: int(value))
    except (TypeError, ValueError):
        return sorted(class_names, key=str)


def load_label_mapping(dataset_dir):
    labels_path = dataset_dir / "labels.json"
    class_map_path = dataset_dir / "class_map.json"
    train_labels_csv_path = dataset_dir / "trainLabels.csv"

    if labels_path.exists() and class_map_path.exists():
        labels = load_json(labels_path)
        class_map = load_json(class_map_path)
        if not isinstance(labels, dict):
            raise ValueError(f"{labels_path} must contain a JSON object")
        if not isinstance(class_map, dict):
            raise ValueError(f"{class_map_path} must contain a JSON object")

        class_items = sort_class_items(class_map)
        id_to_class = {str(class_id): class_name for class_name, class_id in class_items}
        ordered_class_names = [class_name for class_name, _ in class_items]
        return labels, ordered_class_names, id_to_class

    if train_labels_csv_path.exists():
        labels = {}
        class_names = set()
        with train_labels_csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{train_labels_csv_path} must contain a header row")

            image_column = "image" if "image" in reader.fieldnames else reader.fieldnames[0]
            label_column = "level" if "level" in reader.fieldnames else reader.fieldnames[-1]

            for row in reader:
                sample_id = row.get(image_column)
                class_name = row.get(label_column)
                if sample_id in {None, ""} or class_name in {None, ""}:
                    continue
                labels[str(sample_id)] = str(class_name)
                class_names.add(str(class_name))

        ordered_class_names = sort_class_names(class_names)
        id_to_class = {class_name: class_name for class_name in ordered_class_names}
        return labels, ordered_class_names, id_to_class

    return None, None, None


def build_train_samples_by_class(labels, ordered_class_names, id_to_class, splits):
    samples_by_class = {class_name: [] for class_name in ordered_class_names}

    for sample_id in splits["train"]:
        label_key = sample_id
        if label_key not in labels:
            label_key = Path(sample_id).stem
        if label_key not in labels:
            raise ValueError(f"Train sample '{sample_id}' is missing from label metadata")

        class_name = normalize_label(labels[label_key], id_to_class)
        if class_name is None:
            raise ValueError(
                f"Could not map label '{labels[label_key]}' for train sample '{sample_id}'"
            )

        samples_by_class[class_name].append(sample_id)

    return samples_by_class


def compute_target_class_counts(class_counts, ordered_class_names, data_fraction, min_images_per_class):
    if min_images_per_class < 1:
        raise ValueError("min_images_per_class must be at least 1")

    for class_name in ordered_class_names:
        if class_counts[class_name] < min_images_per_class:
            raise ValueError(
                f"class '{class_name}' has only {class_counts[class_name]} train samples, "
                f"which is below the required minimum of {min_images_per_class}"
            )

    total_train_samples = sum(class_counts.values())
    minimum_total = min_images_per_class * len(ordered_class_names)
    target_total = max(round(total_train_samples * data_fraction), minimum_total)
    target_total = min(target_total, total_train_samples)

    target_counts = {class_name: min_images_per_class for class_name in ordered_class_names}
    extra_budget = target_total - minimum_total
    if extra_budget <= 0:
        return target_counts

    extra_targets = {}
    remainders = {}
    for class_name in ordered_class_names:
        raw_target = class_counts[class_name] * data_fraction
        extra_target = max(raw_target - min_images_per_class, 0.0)
        max_extra_capacity = class_counts[class_name] - min_images_per_class
        floored_extra = min(int(extra_target), max_extra_capacity)
        target_counts[class_name] += floored_extra
        extra_targets[class_name] = extra_target
        remainders[class_name] = extra_target - floored_extra

    remaining_budget = target_total - sum(target_counts.values())
    if remaining_budget <= 0:
        return target_counts

    ranked_classes = sorted(
        ordered_class_names,
        key=lambda class_name: (remainders[class_name], extra_targets[class_name], class_name),
        reverse=True,
    )
    for class_name in ranked_classes:
        if remaining_budget == 0:
            break
        if target_counts[class_name] >= class_counts[class_name]:
            continue
        target_counts[class_name] += 1
        remaining_budget -= 1

    if remaining_budget != 0:
        raise ValueError("Could not distribute the full target train sample budget across classes")

    return target_counts


def build_subset_split_payload(splits, target_counts, ordered_class_names, samples_by_class, rng):
    selected_train_samples = []
    for class_name in ordered_class_names:
        selected_train_samples.extend(
            rng.sample(samples_by_class[class_name], target_counts[class_name])
        )

    rng.shuffle(selected_train_samples)
    return {
        "train": selected_train_samples,
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }


def fraction_filename_label(data_fraction):
    return f"{data_fraction:.1f}".replace(".", "_")


def process_dataset(dataset_dir, output_dirname, min_images_per_class):
    splits_path = dataset_dir / "splits.json"
    if not splits_path.exists():
        return False, f"skipped {dataset_dir.name}: missing splits.json"

    splits = load_json(splits_path)
    if not isinstance(splits, dict):
        raise ValueError(f"{splits_path} must contain a JSON object")
    for split_name in ("train", "val", "test"):
        if split_name not in splits or not isinstance(splits[split_name], list):
            raise ValueError(f"{splits_path} must contain a {split_name} split array")

    labels, ordered_class_names, id_to_class = load_label_mapping(dataset_dir)
    if labels is None:
        return False, f"skipped {dataset_dir.name}: missing labels.json/class_map.json or trainLabels.csv"

    samples_by_class = build_train_samples_by_class(labels, ordered_class_names, id_to_class, splits)
    class_counts = {
        class_name: len(samples_by_class[class_name]) for class_name in ordered_class_names
    }

    output_dir = dataset_dir / output_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_file in output_dir.glob("data_fraction_*.json"):
        existing_file.unlink()

    supported_values = []
    skipped_values = []

    for data_fraction in DATA_FRACTIONS:
        try:
            target_counts = compute_target_class_counts(
                class_counts=class_counts,
                ordered_class_names=ordered_class_names,
                data_fraction=data_fraction,
                min_images_per_class=min_images_per_class,
            )
        except ValueError:
            skipped_values.append(data_fraction)
            continue

        for trial_index in range(NUM_TRIALS):
            rng = random.Random(f"{dataset_dir.name}:{data_fraction}:{trial_index}:{min_images_per_class}")
            payload = build_subset_split_payload(
                splits=splits,
                target_counts=target_counts,
                ordered_class_names=ordered_class_names,
                samples_by_class=samples_by_class,
                rng=rng,
            )
            fraction_label = fraction_filename_label(data_fraction)
            output_path = output_dir / f"data_fraction_{fraction_label}_trial_{trial_index}.json"
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=False)
                handle.write("\n")
        supported_values.append(data_fraction)

    if not supported_values:
        return False, f"skipped {dataset_dir.name}: no requested data fractions are supported"

    total_written = len(supported_values) * NUM_TRIALS
    message = f"wrote {output_dir} ({total_written} data-fraction subset files)"
    if skipped_values:
        skipped_labels = [f"{value:.1f}" for value in skipped_values]
        message += f"; skipped unsupported values {skipped_labels}"
    return True, message


def main():
    args = parse_args()
    root = args.root.resolve()

    processed = 0
    skipped = 0

    for dataset_dir in dataset_directories(root):
        success, message = process_dataset(
            dataset_dir=dataset_dir,
            output_dirname=args.output_dirname,
            min_images_per_class=args.min_images_per_class,
        )
        print(message)
        if success:
            processed += 1
        else:
            skipped += 1

    print(f"processed={processed} skipped={skipped}")


if __name__ == "__main__":
    main()
