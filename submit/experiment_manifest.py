#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
from human_readable_id import HridError, generate_hrid
from omegaconf import OmegaConf


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

GLOBALS = {
    "trial": [0],
    "data_fraction": [],  # [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    "samples_per_class": [50],  # [10, 20, 30, 40, 50]
    "max_epochs": [20],  # [5, 10, 20, 40, 100]
    "lr": [0.1, 0.05, 0.01, 0.005, 0.001],
    "classification_head_dropout": [0.3],
    "warmstart": [0.1],
    "gradient_clip_val": [None],
    "layer_wise_lr_decay": [None],
    "undecay_norm": [False],
    "drop_path_rate": [None],
    "token_aggregation_method": ["cls_token"],
    "probing_location": [[-1]],
    "optimizer": ["AdamW"],
    "scheduler": ["CosineAnneal"],
    "weight_decay": [0.05],
    "compile": [True],
    "label_smoothing": [0.0],
}

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
    # ── No tunable hparams ────────────────────────────────────────────────
    "linear_probing": {},
    "bitfit": {},
    "ln_tuning": {},
    "ssf": {},
    "difffit": {},
    "ia3": {},

    # ── Global-override only ───────────────────────────────────────────────
    "full_finetuning": {
        "warmstart": [10],               # 0, 5, 10, 20
        "gradient_clip_val": [1.0],      # None, 0.5, 1.0, 5.0
        "num_first_frozen_layers": [0],
    },

    # ── Partial finetuning ────────────────────────────────────────────────
    "partial_finetuning": {
        "num_frozen_layers": [6],        # 0, 3, 6, 9, 11
    },

    # ── LoRA family ───────────────────────────────────────────────────────
    "lora": {
        "lora_rank": [8],                # 2, 4, 8, 16, 32
        "lora_alpha": [16],              # 4, 8, 16, 32, 64  (typically 2× rank)
        "lora_dropout": [0.0],           # 0.0, 0.05, 0.1, 0.2
        "use_rslora": [False],           # False, True
        "init_lora_weights": [True],     # True, "pissa", "loftq"
        "lora_bias": ["none"],           # "none", "all", "lora_only"
    },
    "dora": {
        "lora_rank": [8],                # 2, 4, 8, 16, 32
        "lora_alpha": [16],              # 4, 8, 16, 32, 64  (typically 2× rank)
        "lora_dropout": [0.0],           # 0.0, 0.05, 0.1, 0.2
        "use_rslora": [False],           # False, True
        "init_lora_weights": [True],     # True, "pissa", "loftq"
        "lora_bias": ["none"],           # "none", "all", "lora_only"
    },
    "adalora": {
        "adalora_rank": [4],             # 2, 4, 8, 12, 16
        "adalora_init_rank": [12],       # 8, 12, 16, 24  (must be ≥ adalora_rank)
        "adalora_alpha": [32],           # 8, 16, 32, 64
        "adalora_dropout": [0.0],        # 0.0, 0.05, 0.1, 0.2
        "use_rslora": [False],           # False, True
        "init_lora_weights": [True],     # True, "pissa", "loftq"
        "lora_bias": ["none"],           # "none", "all", "lora_only"
        "adalora_orth_reg_weight": [0.5],# 0.0, 0.1, 0.5, 1.0
        "adalora_beta1": [0.85],         # 0.85, 0.9, 0.95
        "adalora_beta2": [0.85],         # 0.85, 0.9, 0.95
        "adalora_tinit": [0],            # 0, 50, 100, 200
        "adalora_deltaT": [1],           # 1, 5, 10
    },
    "vera": {
        "vera_rank": [256],              # 64, 256, 1024
        "vera_dropout": [0.0],           # 0.0, 0.05, 0.1, 0.2
        "vera_projection_prng_key": [0], # 0, 1, 42
    },

    # ── Orthogonal methods ────────────────────────────────────────────────
    "boft": {
        "boft_block_size": [8],          # 4, 8, 16, 32
        "boft_n_butterfly_factor": [1],  # 1, 2, 3
        "boft_dropout": [0.0],           # 0.0, 0.05, 0.1, 0.2
        "boft_bias": ["none"],           # "none", "all", "boft_only"
    },
    "oft": {
        "oft_r": [8],                    # 4, 8, 16, 32
        "oft_dropout": [0.0],            # 0.0, 0.05, 0.1, 0.2
        "oft_coft": [False],             # False, True
        "oft_eps": [6e-5],               # 1e-5, 6e-5, 1e-4  (only active when oft_coft=True)
        "oft_bias": ["none"],            # "none", "all", "oft_only"
        "oft_use_cayley_neumann": [True],# True, False
    },

    # ── LyCORIS variants ──────────────────────────────────────────────────
    "loha": {
        "loha_rank": [4],                # 2, 4, 8, 16
        "loha_alpha": [1.0],             # 1.0, 2.0, 4.0, 8.0
        "loha_dropout": [0.0],           # 0.0, 0.05, 0.1, 0.2
        "loha_rank_dropout": [0.0],      # 0.0, 0.1, 0.2
    },
    "lokr": {
        "lokr_rank": [4],                # 2, 4, 8, 16
        "lokr_alpha": [1.0],             # 1.0, 2.0, 4.0, 8.0
        "lokr_dropout": [0.0],           # 0.0, 0.05, 0.1, 0.2
        "lokr_rank_dropout": [0.0],      # 0.0, 0.1, 0.2
        "lokr_decompose_factor": [-1],   # -1, 2, 4, 8
    },

    # ── Fourier / spectral ────────────────────────────────────────────────
    "fourierft": {
        "fourierft_n_frequency": [1000], # 100, 500, 1000, 2000, 5000
        "fourierft_scaling": [150.0],    # 50.0, 100.0, 150.0, 300.0
    },

    # ── Shared-factorisation ──────────────────────────────────────────────
    "fact": {
        "fact_r": [4],                   # 2, 4, 8, 16, 32
    },

    # ── Adapter methods ───────────────────────────────────────────────────
    "adapt_former": {
        "bottleneck": [64],              # 8, 16, 32, 64, 128, 256
        "dropout": [0.0],                # 0.0, 0.05, 0.1, 0.2
        "adapter_scalar": [1.0],         # 0.1, 0.5, 1.0, "learnable_scalar"
    },
    "convpass": {
        "bottleneck": [64],              # 8, 16, 32, 64, 128, 256
        "dropout": [0.0],                # 0.0, 0.05, 0.1, 0.2
    },
    "rep_adapter": {
        "repadapter_bottleneck": [8],    # 4, 8, 16, 32, 64
        "repadapter_groups": [2],        # 1, 2, 4, 8
        "repadapter_scale_init": [0.001],# 1e-4, 1e-3, 1e-2, 0.1
    },

    # ── Sparse / selection ────────────────────────────────────────────────
    "gps": {
        "gps_percent": [4],              # 1, 2, 4, 8, 16
        "gps_calib_batches": [4],        # 1, 2, 4, 8
    },

    # ── Visual prompt tuning ──────────────────────────────────────────────
    "visual_prompt_tuning": {            # deep=True (set in yaml)
        "num_tokens": [20],              # 5, 10, 20, 50, 100
        "dropout": [0.0],                # 0.0, 0.05, 0.1, 0.2
    },
    "visual_prompt_tuning_shallow": {    # deep=False (set in yaml)
        "num_tokens": [20],              # 5, 10, 20, 50, 100
        "dropout": [0.0],                # 0.0, 0.05, 0.1, 0.2
    },
}

DATASET_EPOCH_SPLIT_TIMINGS_PATH = (
    Path(__file__).resolve().parents[1]
    / "synergy_unit/data/dataset_mean_epoch_split_times.json"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_CONFIG_PATH = REPO_ROOT / "cli_configs" / "train.yaml"
PEFT_CONFIG_DIR = REPO_ROOT / "cli_configs" / "peft"
RESERVED_SWEEP_KEYS = {
    *GLOBALS.keys(),
}
SPLIT_AXIS_KEYS = {"trial", "data_fraction", "samples_per_class"}
SUBMISSION_IGNORED_KEYS = {"drop_path_rate", "token_aggregation_method", "probing_location"}
CONFIG_OVERRIDE_PATHS = {
    "max_epochs": "trainer.max_epochs",
    "lr": "model.lr",
    "classification_head_dropout": "model.classification_head_dropout",
    "warmstart": "model.warmstart",
    "gradient_clip_val": "trainer.gradient_clip_val",
    "layer_wise_lr_decay": "model.layer_wise_lr_decay",
    "undecay_norm": "model.undecay_norm",
    "optimizer": "model.optimizer",
    "scheduler": "model.scheduler",
    "weight_decay": "model.weight_decay",
    "compile": "model.compile",
    "label_smoothing": "model.label_smoothing",
}


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else REPO_ROOT / path


def load_raw_yaml(path: Path) -> dict[str, Any]:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=False)


TRAIN_CONFIG_DEFAULTS = load_raw_yaml(TRAIN_CONFIG_PATH)
PEFT_CONFIG_DEFAULTS = {
    path.stem: load_raw_yaml(path)
    for path in sorted(PEFT_CONFIG_DIR.glob("*.yaml"))
}


def get_nested_value(payload: dict[str, Any], dotted_key: str) -> Any:
    current: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Missing config default for '{dotted_key}'")
        current = current[part]
    return current


def get_sweep_default_values(key: str) -> list[Any]:
    if key in GLOBALS:
        return list(GLOBALS[key])
    raise KeyError(f"Unsupported sweep override key '{key}'")


def normalize_sweep_values(key: str, configured_values: Any) -> list[Any]:
    if configured_values is None:
        return get_sweep_default_values(key)

    values = configured_values if isinstance(configured_values, list) else [configured_values]
    normalized: list[Any] = []
    for value in values:
        resolved_values = get_sweep_default_values(key) if value is None else [value]
        for resolved_value in resolved_values:
            if resolved_value not in normalized:
                normalized.append(resolved_value)
    return normalized


def resolve_peft_default_value(peft_name: str, key: str) -> Any:
    if key in RESERVED_SWEEP_KEYS:
        if key in SPLIT_AXIS_KEYS:
            defaults = get_sweep_default_values(key)
            if len(defaults) != 1:
                raise ValueError(
                    f"Split axis key '{key}' cannot be resolved to a single default value"
                )
            return defaults[0]
        config_key = CONFIG_OVERRIDE_PATHS.get(key)
        if config_key is None and key not in SUBMISSION_IGNORED_KEYS:
            raise KeyError(f"Missing config override path mapping for '{key}'")
        if config_key is None:
            defaults = get_sweep_default_values(key)
            if len(defaults) != 1:
                raise ValueError(
                    f"Manifest-only key '{key}' cannot be resolved to a single default value"
                )
            return defaults[0]
        return get_nested_value(TRAIN_CONFIG_DEFAULTS, config_key)

    if key in CONFIG_OVERRIDE_PATHS:
        return get_nested_value(TRAIN_CONFIG_DEFAULTS, CONFIG_OVERRIDE_PATHS[key])

    if peft_name not in PEFT_CONFIG_DEFAULTS:
        raise KeyError(f"Missing PEFT config defaults for '{peft_name}'")
    defaults = PEFT_CONFIG_DEFAULTS[peft_name]
    if key not in defaults:
        raise KeyError(f"Missing PEFT default for '{peft_name}.{key}'")
    return defaults[key]


def split_peft_definition(
    definition: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sweep_overrides: dict[str, Any] = {}
    peft_params: dict[str, Any] = {}
    for key, value in definition.items():
        if key in RESERVED_SWEEP_KEYS:
            sweep_overrides[key] = value
        else:
            peft_params[key] = value
    return sweep_overrides, peft_params


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
    classification_head_dropout: float
    warmstart: int
    gradient_clip_val: float | None
    layer_wise_lr_decay: float | None
    undecay_norm: bool
    drop_path_rate: float | None
    token_aggregation_method: str
    probing_location: list[int]
    optimizer: str
    scheduler: str
    weight_decay: float
    compile: bool
    label_smoothing: float
    peft_params: dict[str, object]


def iter_peft_variants() -> Iterable[tuple[str, dict[str, object], dict[str, Any]]]:
    for peft_name, definition in PEFTS.items():
        sweep_overrides, hparams = split_peft_definition(definition)
        if not hparams:
            yield peft_name, {}, sweep_overrides
            continue

        param_names = list(hparams.keys())
        value_product = itertools.product(*(hparams[name] for name in param_names))
        for values in value_product:
            resolved_params = {
                name: (
                    resolve_peft_default_value(peft_name, name)
                    if value is None
                    else value
                )
                for name, value in zip(param_names, values, strict=True)
            }
            yield peft_name, resolved_params, sweep_overrides


def resolve_warmstart_epochs(max_epochs: int, warmstart_fraction: float) -> int:
    if warmstart_fraction <= 0:
        return 0
    return max(1, math.ceil(max_epochs * warmstart_fraction))


def iter_experiments() -> Iterable[ExperimentSpec]:
    for peft, peft_params, sweep_overrides in iter_peft_variants():
        trial_values = normalize_sweep_values("trial", sweep_overrides.get("trial"))
        data_fraction_values = normalize_sweep_values(
            "data_fraction", sweep_overrides.get("data_fraction")
        )
        samples_per_class_values = normalize_sweep_values(
            "samples_per_class", sweep_overrides.get("samples_per_class")
        )
        config_global_keys = [
            key for key in GLOBALS if key not in SPLIT_AXIS_KEYS
        ]
        global_values = {
            key: normalize_sweep_values(key, sweep_overrides.get(key))
            for key in config_global_keys
        }
        sweep_variants = [
            {"data_fraction": data_fraction, "samples_per_class": None}
            for data_fraction in data_fraction_values
        ] + [
            {"data_fraction": None, "samples_per_class": samples_per_class}
            for samples_per_class in samples_per_class_values
        ]
        if not sweep_variants:
            raise ValueError(f"PEFT '{peft}' produced no sweep variants")

        global_product = itertools.product(
            *(global_values[key] for key in config_global_keys)
        )

        for (
            trial,
            global_combo,
            sweep_variant,
            model,
            dataset,
        ) in itertools.product(
            trial_values,
            global_product,
            sweep_variants,
            MODELS,
            DATASETS,
        ):
            global_payload = dict(zip(config_global_keys, global_combo, strict=True))
            max_epochs = int(global_payload["max_epochs"])
            yield ExperimentSpec(
                model=model,
                dataset=dataset,
                peft=peft,
                trial=int(trial),
                max_epochs=max_epochs,
                data_fraction=sweep_variant["data_fraction"],
                samples_per_class=sweep_variant["samples_per_class"],
                lr=float(global_payload["lr"]),
                classification_head_dropout=float(global_payload["classification_head_dropout"]),
                warmstart=resolve_warmstart_epochs(
                    max_epochs=max_epochs,
                    warmstart_fraction=float(global_payload["warmstart"]),
                ),
                gradient_clip_val=(
                    None
                    if global_payload["gradient_clip_val"] is None
                    else float(global_payload["gradient_clip_val"])
                ),
                layer_wise_lr_decay=(
                    None
                    if global_payload["layer_wise_lr_decay"] is None
                    else float(global_payload["layer_wise_lr_decay"])
                ),
                undecay_norm=bool(global_payload["undecay_norm"]),
                drop_path_rate=(
                    None
                    if global_payload["drop_path_rate"] is None
                    else float(global_payload["drop_path_rate"])
                ),
                token_aggregation_method=str(global_payload["token_aggregation_method"]),
                probing_location=[int(value) for value in global_payload["probing_location"]],
                optimizer=str(global_payload["optimizer"]),
                scheduler=str(global_payload["scheduler"]),
                weight_decay=float(global_payload["weight_decay"]),
                compile=bool(global_payload["compile"]),
                label_smoothing=float(global_payload["label_smoothing"]),
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
