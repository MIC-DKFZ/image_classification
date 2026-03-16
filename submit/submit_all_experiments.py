#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm


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
class Experiment:
    model: str
    dataset: str
    peft: str
    peft_overrides: dict[str, object]
    max_epochs: int
    data_fraction: float
    lr: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit the full experiment grid via bsub2."
    )
    parser.add_argument(
        "--data-dir",
        help="Value passed to Hydra as data_dir=...",
    )
    parser.add_argument(
        "--exp-dir",
        help="Value passed to Hydra as exp_dir=...",
    )
    parser.add_argument(
        "--env",
        default="cluster",
        help="Hydra env config to use. Defaults to 'cluster'.",
    )
    parser.add_argument(
        "--python-executable",
        default="python",
        help="Python executable used inside the submitted command.",
    )
    parser.add_argument(
        "--wandb-entity",
        default="mic_acvl",
        help="Weights & Biases entity override.",
    )
    parser.add_argument(
        "--wandb-project",
        default="synergy_unit",
        help="Weights & Biases project override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands instead of submitting them.",
    )
    parser.add_argument(
        "--estimate-gpu-hours",
        action="store_true",
        help=(
            "Print the estimated GPU-hours for the full sweep and exit. "
            "Uses dataset-level mean per-epoch runtimes, scaled by epoch count and data fraction."
        ),
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override to append. Can be passed multiple times.",
    )
    return parser.parse_args()


def iter_peft_variants() -> Iterable[tuple[str, dict[str, object]]]:
    for peft_name, hparams in PEFTS.items():
        if not hparams:
            yield peft_name, {}
            continue

        param_names = list(hparams.keys())
        value_product = itertools.product(*(hparams[name] for name in param_names))
        for values in value_product:
            yield peft_name, dict(zip(param_names, values, strict=True))


def iter_experiments() -> Iterable[Experiment]:
    peft_variants = list(iter_peft_variants())
    for max_epochs, data_fraction, lr, model, dataset, (peft, peft_overrides) in itertools.product(
        MAX_EPOCHS,
        DATA_FRACTIONS,
        LEARNING_RATES,
        MODELS,
        DATASETS,
        peft_variants,
    ):
        yield Experiment(
            model=model,
            dataset=dataset,
            peft=peft,
            peft_overrides=peft_overrides,
            max_epochs=max_epochs,
            data_fraction=data_fraction,
            lr=lr,
        )


def build_python_command(args: argparse.Namespace, experiment: Experiment) -> str:
    parts = [
        args.python_executable,
        "main.py",
        f"env={args.env}",
        f"wandb.entity={args.wandb_entity}",
        f"wandb.project={args.wandb_project}",
        "+wandb.name=${ID}",
        f"model={experiment.model}",
        f"data={experiment.dataset}",
        f"peft={experiment.peft}",
        f"trainer.max_epochs={experiment.max_epochs}",
        f"data.module.data_fraction={experiment.data_fraction}",
        f"model.lr={experiment.lr}",
        f"data_dir={args.data_dir}",
        f"exp_dir={args.exp_dir}",
    ]
    parts.extend(
        f"peft.{name}={value}" for name, value in experiment.peft_overrides.items()
    )
    parts.extend(args.extra_override)
    return " ".join(parts)


def build_submit_command(python_command: str) -> list[str]:
    return [
        "bsub2",
        "-n",
        "synergy",
        "-e",
        "synergy",
        "-q",
        "gpu-pro",
        "-m",
        "35",
        "-c",
        python_command,
    ]


def format_submit_command(submit_command: list[str]) -> str:
    python_command = submit_command[-1].replace('"', '\\"')
    return " ".join(submit_command[:-1] + [f'"{python_command}"'])


def estimate_gpu_hours(experiments: Iterable[Experiment]) -> float:
    total_minutes = 0.0
    for experiment in experiments:
        epoch_minutes = DATASET_EPOCH_MEANS[experiment.dataset]
        total_minutes += epoch_minutes * experiment.max_epochs * experiment.data_fraction
    return total_minutes / 60.0


def main() -> int:
    args = parse_args()
    experiments = list(iter_experiments())

    if args.estimate_gpu_hours:
        total_gpu_hours = estimate_gpu_hours(experiments)
        print(f"Estimated GPU-hours for all {len(experiments)} experiments: {total_gpu_hours:.2f}")
        return 0

    if not args.data_dir or not args.exp_dir:
        raise SystemExit("--data-dir and --exp-dir are required unless --estimate-gpu-hours is set.")

    progress_label = "Printing commands" if args.dry_run else "Submitting jobs"
    failures: list[tuple[Experiment, subprocess.CompletedProcess[str]]] = []

    for experiment in tqdm(experiments, desc=progress_label):
        python_command = build_python_command(args, experiment)
        submit_command = build_submit_command(python_command)

        if args.dry_run:
            print(format_submit_command(submit_command))
            continue

        completed = subprocess.run(
            submit_command,
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parents[1],
        )
        if completed.returncode != 0:
            failures.append((experiment, completed))

    if args.dry_run:
        print(f"Dry run complete: {len(experiments)} commands generated.")
        return 0

    if failures:
        print(f"{len(failures)} submissions failed.")
        for experiment, completed in failures[:10]:
            print(
                "FAILED:",
                experiment,
                f"returncode={completed.returncode}",
                completed.stderr.strip() or completed.stdout.strip(),
            )
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more failures.")
        return 1

    print(f"Submitted {len(experiments)} jobs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
