#!/usr/bin/env python3
from __future__ import annotations

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
    "Dataset mean runtime is a per-epoch estimate at data_fraction=1.0.",
    "Runtime scales linearly with max_epochs.",
    "Runtime scales linearly with data_fraction.",
    "Runtime is treated as independent of model, PEFT method, learning rate, and PEFT hyperparameters.",
]

MAX_EPOCHS = [5, 10, 20, 40, 100]
DATA_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
LEARNING_RATES = [1e-5, 2e-5, 5e-5]

MODELS = [
    "supervised",
    "mae_timm",
    "dinov3_reference",
]

DATASETS = [
    "aid",
    "zooscannet",
    "chestxray14",
    "neudet",
    "rxrx1",
    "flowers102",
    "resisc45",
    "pcam",
    "diabetic_retina",
    "fgvc_aircraft",
]

PEFTS = {
    "adapt_former": {
        "bottleneck": [16, 64, 256],
        "dropout": [0.0, 0.05, 0.1],
    },
    "full_finetuning": {},
    "gps": {
        "gps_percent": [1, 4, 16],
        "gps_calib_batches": [1, 2, 4],
    },
    "linear_probing": {},
    "lora": {
        "lora_rank": [4, 8, 16],
        "lora_alpha": [8, 16, 32],
    },
    "vera": {
        "vera_rank": [4, 8, 16],
        "vera_dropout": [0.0, 0.01, 0.05],
    },
    "visual_prompt_tuning": {
        "num_tokens": [8, 20, 40],
        "dropout": [0.0, 0.05, 0.1],
    },
}

DATASET_EPOCH_MEANS = {
    "aid": 0.43,
    "zooscannet": 25.09,
    "chestxray14": 3.67,
    "neudet": 0.13,
    "rxrx1": 3.11,
    "flowers102": 0.36,
    "resisc45": 0.99,
    "pcam": 7.74,
    "diabetic_retina": 4.96,
    "fgvc_aircraft": 0.37,
}


@dataclass(frozen=True)
class ExperimentSpec:
    model: str
    dataset: str
    peft: str
    max_epochs: int
    data_fraction: float
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
    for (
        max_epochs,
        data_fraction,
        lr,
        model,
        dataset,
        (peft, peft_params),
    ) in itertools.product(
        MAX_EPOCHS,
        DATA_FRACTIONS,
        LEARNING_RATES,
        MODELS,
        DATASETS,
        peft_variants,
    ):
        yield ExperimentSpec(
            model=model,
            dataset=dataset,
            peft=peft,
            max_epochs=max_epochs,
            data_fraction=data_fraction,
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


def estimate_gpu_hours(experiments: Iterable[ExperimentSpec]) -> float:
    total_minutes = 0.0
    for experiment in experiments:
        epoch_minutes = DATASET_EPOCH_MEANS[experiment.dataset]
        total_minutes += (
            epoch_minutes * experiment.max_epochs * experiment.data_fraction
        )
    return total_minutes / 60.0


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
