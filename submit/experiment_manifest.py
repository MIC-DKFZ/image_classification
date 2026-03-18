#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from human_readable_id import HridError, generate_hrid


GRID_VERSION = "v1"
HRID_SETTINGS = {
    "words": 2,
    "numbers": 6,
    "separator": "_",
    "use_hash_suffix": True,
}
ESTIMATION_ASSUMPTIONS = [
    "Dataset runtime is a per-epoch estimate in seconds at data_fraction=1.0.",
    "Runtime scales linearly with max_epochs.",
    "Each experiment uses either data_fraction or samples_per_class, never both.",
    "The train portion of an epoch scales linearly with data_fraction.",
    "The validation portion of an epoch stays fixed across data fractions.",
    "For samples_per_class sweeps, train runtime is estimated from train_epoch_seconds_per_image multiplied by samples_per_class and the number of classes.",
    "Runtime is treated as independent of model, PEFT method, learning rate, and PEFT hyperparameters.",
]

MAX_EPOCHS = [20]  # [5, 10, 20, 40, 100]
DATA_FRACTIONS = []  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
SAMPLES_PER_CLASS = [50]  # [10, 20, 30, 40, 50]
TRIALS = [0]
LEARNING_RATES = [1e-5, 2e-5, 5e-5]

MODELS = [
    # "supervised",
    # "mae_timm",
    "dinov3_reference",
]

DATASETS = [
    "aid",
    # "zooscannet",
    # "chestxray14",
    # "neudet",
    # "rxrx1",
    # "flowers102",
    # "resisc45",
    # "pcam",
    # "diabetic_retina",
    # "fgvc_aircraft",
]

PEFTS = {
    # "adapt_former": {
    #     "bottleneck": [16, 64, 256],
    #     "dropout": [0.0, 0.05, 0.1],
    # },
    # "full_finetuning": {},
    # "gps": {
    #     "gps_percent": [1, 4, 16],
    #     "gps_calib_batches": [1, 2, 4],
    # },
    "linear_probing": {},
    # "lora": {
    #     "lora_rank": [4, 8, 16],
    #     "lora_alpha": [8, 16, 32],
    # },
    # "vera": {
    #     "vera_rank": [4, 8, 16],
    #     "vera_dropout": [0.0, 0.01, 0.05],
    # },
    # "visual_prompt_tuning": {
    #     "num_tokens": [8, 20, 40],
    #     "dropout": [0.0, 0.05, 0.1],
    # },
}

DATASET_EPOCH_SPLIT_TIMINGS_PATH = (
    Path(__file__).resolve().parents[1]
    / "synergy_unit/data/dataset_mean_epoch_split_times.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def count_dataset_classes(dataset_dir: Path) -> int:
    class_map_path = dataset_dir / "class_map.json"
    if class_map_path.exists():
        class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
        if not isinstance(class_map, dict):
            raise ValueError(f"{class_map_path} must contain a JSON object")
        return len(class_map)

    train_labels_path = dataset_dir / "trainLabels.csv"
    if train_labels_path.exists():
        with train_labels_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{train_labels_path} must contain a header row")
            label_column = "level" if "level" in reader.fieldnames else reader.fieldnames[-1]
            labels = {
                row[label_column]
                for row in reader
                if row.get(label_column) not in {None, ""}
            }
        return len(labels)

    raise FileNotFoundError(
        f"Could not determine class count for {dataset_dir}: "
        "expected class_map.json or trainLabels.csv"
    )


def load_dataset_runtime_metadata(
    path: Path = DATASET_EPOCH_SPLIT_TIMINGS_PATH,
) -> dict[str, dict[str, float | int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    timings = payload["mean_epoch_seconds_per_dataset"]
    metadata = {}
    for dataset, values in timings.items():
        dataset_json_path = resolve_repo_path(values["dataset_json"])
        metadata[dataset] = {
            "train_epoch_seconds": float(values["train_epoch_seconds"]),
            "val_epoch_seconds": float(values["val_epoch_seconds"]),
            "train_epoch_seconds_per_image": float(values["train_epoch_seconds_per_image"]),
            "num_classes": count_dataset_classes(dataset_json_path.parent),
        }
    return metadata


DATASET_RUNTIME_METADATA = load_dataset_runtime_metadata()


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    dataset: str
    peft: str
    trial: int
    max_epochs: int
    data_fraction: float | None
    samples_per_class: int | None
    lr: float
    peft_params: dict[str, object]


def iter_peft_variants() -> Iterable[tuple[str, dict[str, object]]]:
    for peft_name, hparams in PEFTS.items():
        if not hparams:
            yield peft_name, {}
            continue

        param_names = list(hparams.keys())
        value_product = itertools.product(*(hparams[name] for name in param_names))
        for values in value_product:
            yield peft_name, dict(zip(param_names, values, strict=True))


def iter_experiments() -> Iterable[ExperimentSpec]:
    peft_variants = list(iter_peft_variants())
    sweep_variants = [
        {"data_fraction": data_fraction, "samples_per_class": None}
        for data_fraction in DATA_FRACTIONS
    ] + [
        {"data_fraction": None, "samples_per_class": samples_per_class}
        for samples_per_class in SAMPLES_PER_CLASS
    ]
    for (
        trial,
        max_epochs,
        sweep_variant,
        lr,
        model,
        dataset,
        (peft, peft_params),
    ) in itertools.product(
        TRIALS,
        MAX_EPOCHS,
        sweep_variants,
        LEARNING_RATES,
        MODELS,
        DATASETS,
        peft_variants,
    ):
        yield ExperimentSpec(
            model=model,
            dataset=dataset,
            peft=peft,
            trial=trial,
            max_epochs=max_epochs,
            data_fraction=sweep_variant["data_fraction"],
            samples_per_class=sweep_variant["samples_per_class"],
            lr=lr,
            peft_params=peft_params,
        )


def experiment_payload(spec: ExperimentSpec) -> dict[str, Any]:
    return asdict(spec)


def canonical_payload_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def generate_experiment_id(payload: dict[str, Any]) -> str:
    try:
        return generate_hrid(seed=canonical_payload_json(payload), **HRID_SETTINGS)
    except HridError as exc:
        raise ValueError(f"Failed to generate HRID for payload {payload}") from exc


def estimate_epoch_seconds(
    dataset: str,
    data_fraction: float | None = None,
    samples_per_class: int | None = None,
) -> float:
    if (data_fraction is None) == (samples_per_class is None):
        raise ValueError(
            "Exactly one of data_fraction or samples_per_class must be provided"
        )

    runtime = DATASET_RUNTIME_METADATA[dataset]
    if data_fraction is not None:
        train_epoch_seconds = runtime["train_epoch_seconds"] * data_fraction
    else:
        train_image_count = runtime["num_classes"] * samples_per_class
        train_epoch_seconds = runtime["train_epoch_seconds_per_image"] * train_image_count

    return train_epoch_seconds + runtime["val_epoch_seconds"]


def estimate_gpu_hours(experiments: Iterable[ExperimentSpec]) -> float:
    total_seconds = 0.0
    for experiment in experiments:
        epoch_seconds = estimate_epoch_seconds(
            dataset=experiment.dataset,
            data_fraction=experiment.data_fraction,
            samples_per_class=experiment.samples_per_class,
        )
        total_seconds += epoch_seconds * experiment.max_epochs
    return total_seconds / 3600.0


def build_manifest() -> dict[str, Any]:
    experiments = list(iter_experiments())
    estimated_gpu_hours = estimate_gpu_hours(experiments)

    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    for experiment in experiments:
        payload = experiment_payload(experiment)
        experiment_id = generate_experiment_id(payload)
        if experiment_id in seen_ids:
            raise ValueError(
                f"Duplicate experiment ID '{experiment_id}' detected. "
                "Adjust the HRID settings or the sweep definition."
            )
        seen_ids.add(experiment_id)
        records.append({"id": experiment_id, **payload})

    return {
        "grid_version": GRID_VERSION,
        "experiment_count": len(records),
        "estimated_gpu_hours": round(estimated_gpu_hours, 2),
        "estimation_assumptions": ESTIMATION_ASSUMPTIONS,
        "hrid_settings": HRID_SETTINGS,
        "experiments": records,
    }


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
