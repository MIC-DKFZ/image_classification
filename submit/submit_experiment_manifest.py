#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submit.experiment_manifest import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit experiments from a generated manifest via bsub2."
    )
    parser.add_argument(
        "manifest",
        help="Path to a generated experiment manifest JSON.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Value passed to Hydra as data_dir=...",
    )
    parser.add_argument(
        "--exp-dir",
        required=True,
        help="Value passed to Hydra as exp_dir=...",
    )
    parser.add_argument(
        "--conda-path",
        default="/dkfz/cluster/gpu/data/mic_data_common/synergy_unit/scripts/submit.sh",
        help="Path to the cluster submission wrapper script.",
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
        "--subset",
        help=(
            "Optional 0-based subset slice in the form start:end. "
            "Start is inclusive, end is exclusive."
        ),
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override to append. Can be passed multiple times.",
    )
    return parser.parse_args()


def build_python_command(args: argparse.Namespace, experiment: dict[str, object]) -> str:
    parts = [
        "python",
        "main.py",
        "env=cluster",
        f"wandb.entity={args.wandb_entity}",
        f"wandb.project={args.wandb_project}",
        f"+wandb.name={experiment['id']}",
        f"model={experiment['model']}",
        f"data={experiment['dataset']}",
        f"peft={experiment['peft']}",
        f"trainer.max_epochs={experiment['max_epochs']}",
        f"data.module.data_fraction={experiment['data_fraction']}",
        f"model.lr={experiment['lr']}",
        f"data_dir={args.data_dir}",
        f"exp_dir={args.exp_dir}",
    ]

    peft_params = experiment.get("peft_params", {})
    for name, value in peft_params.items():
        parts.append(f"peft.{name}={value}")

    parts.extend(args.extra_override)
    return " ".join(parts)


def build_submit_command(
    args: argparse.Namespace, experiment: dict[str, object], python_command: str
) -> list[str]:
    return [
        args.conda_path,
        "-i",
        str(experiment["id"]),
        "-n",
        "synergy",
        "-e",
        "synergy",
        "-c",
        python_command,
    ]


def format_submit_command(submit_command: list[str]) -> str:
    python_command = submit_command[-1].replace('"', '\\"')
    return " ".join(submit_command[:-1] + [f'"{python_command}"'])


def parse_subset_arg(subset: str, total_experiments: int) -> tuple[int, int]:
    parts = subset.split(":", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"Invalid --subset value '{subset}'. Expected format start:end."
        )

    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Invalid --subset value '{subset}'. Start and end must be integers."
        ) from exc

    if start < 0 or end < 0:
        raise ValueError("--subset does not support negative indices.")
    if start > end:
        raise ValueError("--subset start must be <= end.")
    if end > total_experiments:
        raise ValueError(
            f"--subset end {end} exceeds the manifest size {total_experiments}."
        )

    return start, end


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    experiments = manifest["experiments"]

    if args.subset:
        start, end = parse_subset_arg(args.subset, len(experiments))
        experiments = experiments[start:end]

    progress_label = "Printing commands" if args.dry_run else "Submitting jobs"
    failures: list[tuple[str, subprocess.CompletedProcess[str]]] = []

    for experiment in tqdm(experiments, desc=progress_label):
        python_command = build_python_command(args, experiment)
        submit_command = build_submit_command(args, experiment, python_command)

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
            failures.append((str(experiment["id"]), completed))

    if args.dry_run:
        print(f"Dry run complete: {len(experiments)} commands generated.")
        return 0

    if failures:
        print(f"{len(failures)} submissions failed.")
        for experiment_id, completed in failures[:10]:
            print(
                "FAILED:",
                experiment_id,
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
