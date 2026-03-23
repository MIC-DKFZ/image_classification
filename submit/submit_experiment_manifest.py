#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
import sys
from typing import Iterable

from tqdm import tqdm
import wandb

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
    parser.add_argument(
        "--trial",
        type=int,
        help=(
            "Deprecated override for subset trial index. If omitted, the value "
            "from the manifest record is used."
        ),
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help=(
            "Use reference_batch_sizes_small.json and add '-q gpu -m 1' to the "
            "submission command."
        ),
    )
    parser.add_argument(
        "--disable-checkpointing",
        action="store_true",
        help="Append trainer.enable_checkpointing=false to the Python command.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help=(
            "Disable the default behavior of skipping experiments whose W&B run "
            "name already exists in state 'finished'."
        ),
    )
    return parser.parse_args()


def fraction_label(value: float) -> str:
    return f"{value:.1f}".replace(".", "_")


SKIPPABLE_WANDB_STATES = {"finished", "running"}
WANDB_RUN_QUERY_BATCH_SIZE = 1000


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def get_skippable_run_names(entity: str, project: str, run_names: Iterable[str]) -> set[str]:
    target_run_names = sorted(set(run_names))
    if not target_run_names:
        return set()

    print(
        f"Fetching W&B runs for {entity}/{project} to identify completed or running experiments...",
        flush=True,
    )
    api = wandb.Api()

    print(
        f"W&B lookup will query {len(target_run_names)} run names in "
        f"{(len(target_run_names) + WANDB_RUN_QUERY_BATCH_SIZE - 1) // WANDB_RUN_QUERY_BATCH_SIZE} batches.",
        flush=True,
    )

    found_run_names: set[str] = set()
    for batch_index, batch in enumerate(chunked(target_run_names, WANDB_RUN_QUERY_BATCH_SIZE), start=1):
        print(
            f"W&B lookup batch {batch_index}: querying {len(batch)} run names...",
            flush=True,
        )
        try:
            runs = api.runs(
                f"{entity}/{project}",
                filters={
                    "$and": [
                        {"state": {"$in": sorted(SKIPPABLE_WANDB_STATES)}},
                        {"display_name": {"$in": batch}},
                    ]
                },
                include_sweeps=False,
                per_page=len(batch),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch runs: {e}")

        batch_found = {
            run.name
            for run in runs
            if run.name and run.state in SKIPPABLE_WANDB_STATES
        }
        found_run_names.update(batch_found)
        print(
            f"W&B lookup batch {batch_index} complete: matched {len(batch_found)} runs.",
            flush=True,
        )

    print(
        f"W&B lookup complete: found {len(found_run_names)} completed/running run names to skip.",
        flush=True,
    )
    return found_run_names


def load_reference_batch_sizes(data_dir: str, *, small: bool = False) -> dict[tuple[str, str], int]:
    filename = "reference_batch_sizes_small.json" if small else "reference_batch_sizes.json"
    reference_path = Path(data_dir) / filename
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference batch size file: {reference_path}")

    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{reference_path} must contain a JSON list")

    mapping: dict[tuple[str, str], int] = {}
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{reference_path} entries must be JSON objects")
        model = item.get("model")
        peft = item.get("peft")
        max_batch_size = item.get("max_batch_size")
        if not isinstance(model, str) or not isinstance(peft, str):
            raise ValueError(f"{reference_path} entries must contain string model and peft fields")
        if max_batch_size is None:
            continue
        if not isinstance(max_batch_size, int):
            raise ValueError(f"{reference_path} max_batch_size must be an integer or null")

        key = (model, peft)
        existing = mapping.get(key)
        if existing is not None and existing != max_batch_size:
            raise ValueError(
                f"Inconsistent max_batch_size for model={model}, peft={peft}: "
                f"{existing} vs {max_batch_size}"
            )
        mapping[key] = max_batch_size

    return mapping


def build_split_file_override(args: argparse.Namespace, experiment: dict[str, object]) -> str:
    data_fraction = experiment.get("data_fraction")
    samples_per_class = experiment.get("samples_per_class")
    trial = args.trial if args.trial is not None else experiment.get("trial", 0)

    if (data_fraction is None) == (samples_per_class is None):
        raise ValueError(
            f"Experiment {experiment['id']} must define exactly one of "
            "data_fraction or samples_per_class."
        )

    if data_fraction is not None:
        return (
            f"subsets/data_fraction_{fraction_label(float(data_fraction))}"
            f"_trial_{trial}.json"
        )

    return f"subsets/samples_per_class_{samples_per_class}_trial_{trial}.json"


def build_python_command(
    args: argparse.Namespace,
    experiment: dict[str, object],
    reference_batch_sizes: dict[tuple[str, str], int],
) -> str:
    split_file = build_split_file_override(args, experiment)
    data_fraction = experiment.get("data_fraction")
    samples_per_class = experiment.get("samples_per_class")
    batch_size_key = (str(experiment["model"]), str(experiment["peft"]))
    if batch_size_key not in reference_batch_sizes:
        raise KeyError(
            f"Missing reference max batch size for model={batch_size_key[0]}, peft={batch_size_key[1]}"
        )
    max_batch_size = reference_batch_sizes[batch_size_key]

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
        f"data.module.batch_size={max_batch_size}",
        "data.module.data_fraction=null",
        f"+data.module.split_file={split_file}",
        f"+dataset={experiment['dataset']}",
        f"+data_fraction={data_fraction if data_fraction is not None else 'null'}",
        f"+samples_per_class={samples_per_class if samples_per_class is not None else 'null'}",
        f"model.lr={experiment['lr']}",
        f"data_dir={args.data_dir}",
        f"exp_dir={args.exp_dir}",
    ]

    peft_params = experiment.get("peft_params", {})
    for name, value in peft_params.items():
        if "." in name:
            parts.append(f"{name}={value}")
        else:
            parts.append(f"++peft.{name}={value}")

    if args.disable_checkpointing:
        parts.append("trainer.enable_checkpointing=false")

    parts.extend(args.extra_override)
    return " ".join(parts)


def build_submit_command(
    args: argparse.Namespace, experiment: dict[str, object], python_command: str
) -> list[str]:
    command = [
        args.conda_path,
        "-i",
        str(experiment["id"]),
        "-n",
        "synergy",
        "-e",
        "synergy",
    ]
    if args.small:
        command.extend(["-q", "gpu", "-m", "1"])
    command.extend(["-c", python_command])
    return command


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
    reference_batch_sizes = load_reference_batch_sizes(args.data_dir, small=args.small)

    if args.subset:
        start, end = parse_subset_arg(args.subset, len(experiments))
        experiments = experiments[start:end]

    progress_label = "Printing commands" if args.dry_run else "Submitting jobs"
    failures: list[tuple[str, subprocess.CompletedProcess[str]]] = []
    skipped_completed = 0
    submitted_count = 0
    skip_completed = not args.no_skip_completed
    target_run_names = [str(experiment["id"]) for experiment in experiments]
    finished_run_names = (
        get_skippable_run_names(args.wandb_entity, args.wandb_project, target_run_names)
        if skip_completed
        else None
    )
    if skip_completed:
        print("Skipping experiments already marked finished or currently running in W&B.", flush=True)
    else:
        print("W&B skip check disabled; all manifest experiments will be considered for submission.", flush=True)

    for experiment in tqdm(experiments, desc=progress_label):
        if skip_completed and str(experiment["id"]) in finished_run_names:
            skipped_completed += 1
            continue

        python_command = build_python_command(args, experiment, reference_batch_sizes)
        submit_command = build_submit_command(args, experiment, python_command)

        if args.dry_run:
            print(format_submit_command(submit_command))
            submitted_count += 1
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
        else:
            submitted_count += 1

    if args.dry_run:
        print(
            f"Dry run complete: submitted={submitted_count} skipped={skipped_completed}"
        )
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
        print(f"Summary: submitted={submitted_count} skipped={skipped_completed}")
        return 1

    print(f"Submitted {submitted_count} jobs. Skipped {skipped_completed} completed experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
