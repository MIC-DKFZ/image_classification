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
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override to append. Can be passed multiple times.",
    )
    return parser.parse_args()


def build_python_command(args: argparse.Namespace, experiment: dict[str, object]) -> str:
    parts = [
        args.python_executable,
        "main.py",
        f"env={args.env}",
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


def main() -> int:
    args = parse_args()
    manifest = read_manifest(args.manifest)
    experiments = manifest["experiments"]
    progress_label = "Printing commands" if args.dry_run else "Submitting jobs"
    failures: list[tuple[str, subprocess.CompletedProcess[str]]] = []

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
