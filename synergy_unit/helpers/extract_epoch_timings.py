#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


DEFAULT_MANIFEST_PATH = Path("submit/experiment_manifest.py")
DEFAULT_TIMINGS_PATH = Path("synergy_unit/data/epoch_timings.txt")
DEFAULT_OUTPUT_PATH = Path("synergy_unit/data/epoch_timings.json")

HEADER_RE = re.compile(r"^\[(\d+)/(\d+)\] ([^ ]+) \(data_frac=([0-9.]+)\)$")
SUCCESS_RE = re.compile(r"^\s*✓ SUCCESS \((\d+)s\)$")
CONFIG_RE = re.compile(r"^Configuration: max_epochs=(\d+), data_fractions=(.+)$")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract 5-epoch timings from the timing log, convert them to "
            "per-epoch timings, and write a structured JSON summary."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=f"Path to the experiment manifest definition. Defaults to {DEFAULT_MANIFEST_PATH}.",
    )
    parser.add_argument(
        "--timings-path",
        type=Path,
        default=DEFAULT_TIMINGS_PATH,
        help=f"Path to the timing log. Defaults to {DEFAULT_TIMINGS_PATH}.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path to the output JSON file. Defaults to {DEFAULT_OUTPUT_PATH}.",
    )
    return parser.parse_args()


def load_manifest_grid(manifest_path: Path) -> tuple[list[str], list[str], list[str]]:
    module = ast.parse(manifest_path.read_text(encoding="utf-8"), filename=str(manifest_path))
    values: dict[str, object] = {}

    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in {"MODELS", "DATASETS", "PEFTS"}:
                values[target.id] = ast.literal_eval(node.value)

    missing = {"MODELS", "DATASETS", "PEFTS"} - values.keys()
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Missing manifest assignments: {missing_names}")

    models = list(values["MODELS"])
    datasets = list(values["DATASETS"])
    pefts = list(values["PEFTS"].keys())
    return models, datasets, pefts


def parse_timing_name(name: str, models: list[str], datasets: list[str], pefts: list[str]) -> tuple[str, str, str]:
    base_name, _, _ = name.rpartition("_frac")
    if not base_name:
        raise ValueError(f"Unable to strip data fraction suffix from '{name}'")

    model = next(
        (candidate for candidate in sorted(models, key=len, reverse=True) if base_name.startswith(f"{candidate}_")),
        None,
    )
    if model is None:
        raise ValueError(f"Unable to match model in '{name}'")

    remainder = base_name[len(model) + 1 :]
    peft = next(
        (candidate for candidate in sorted(pefts, key=len, reverse=True) if remainder.endswith(f"_{candidate}")),
        None,
    )
    if peft is None:
        raise ValueError(f"Unable to match PEFT method in '{name}'")

    dataset = remainder[: -(len(peft) + 1)]
    if dataset not in datasets:
        raise ValueError(f"Unable to match dataset '{dataset}' in '{name}'")

    return model, dataset, peft


def parse_timings(
    timings_path: Path,
    models: list[str],
    datasets: list[str],
    pefts: list[str],
) -> tuple[int, list[str], list[dict[str, object]], list[dict[str, object]]]:
    max_epochs = None
    data_fractions: list[str] = []
    completed_runs: list[dict[str, object]] = []
    incomplete_runs: list[dict[str, object]] = []
    pending_run: dict[str, object] | None = None

    for raw_line in timings_path.read_text(encoding="utf-8").splitlines():
        config_match = CONFIG_RE.match(raw_line)
        if config_match:
            max_epochs = int(config_match.group(1))
            data_fractions = config_match.group(2).split()
            continue

        header_match = HEADER_RE.match(raw_line)
        if header_match:
            if pending_run is not None:
                incomplete_runs.append(pending_run)

            run_name = header_match.group(3)
            data_fraction = header_match.group(4)
            model, dataset, peft = parse_timing_name(run_name, models, datasets, pefts)
            pending_run = {
                "index": int(header_match.group(1)),
                "declared_total": int(header_match.group(2)),
                "name": run_name,
                "model": model,
                "dataset": dataset,
                "peft": peft,
                "data_fraction": data_fraction,
            }
            continue

        success_match = SUCCESS_RE.match(raw_line)
        if success_match and pending_run is not None:
            total_seconds = int(success_match.group(1))
            completed_runs.append(
                {
                    **pending_run,
                    "total_seconds_for_5_epochs": total_seconds,
                }
            )
            pending_run = None

    if pending_run is not None:
        incomplete_runs.append(pending_run)

    if max_epochs is None:
        raise ValueError(f"Could not parse max_epochs from {timings_path}")
    if not data_fractions:
        raise ValueError(f"Could not parse data_fractions from {timings_path}")

    return max_epochs, data_fractions, completed_runs, incomplete_runs


def build_nested_timings(
    models: list[str],
    datasets: list[str],
    pefts: list[str],
    data_fractions: list[str],
    max_epochs: int,
    completed_runs: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, dict[str, object | None]]]]:
    timings = {
        model: {
            dataset: {
                peft: {
                    data_fraction: None for data_fraction in data_fractions
                }
                for peft in pefts
            }
            for dataset in datasets
        }
        for model in models
    }

    for run in completed_runs:
        total_seconds = run["total_seconds_for_5_epochs"]
        timings[run["model"]][run["dataset"]][run["peft"]][run["data_fraction"]] = {
            "total_seconds": total_seconds,
            "epoch_seconds": round(total_seconds / max_epochs, 4),
        }

    return timings


def find_missing_runs(
    models: list[str],
    datasets: list[str],
    pefts: list[str],
    data_fractions: list[str],
    completed_runs: list[dict[str, object]],
    incomplete_runs: list[dict[str, object]],
) -> list[dict[str, str]]:
    completed_keys = {
        (run["model"], run["dataset"], run["peft"], run["data_fraction"])
        for run in completed_runs
    }
    incomplete_keys = {
        (run["model"], run["dataset"], run["peft"], run["data_fraction"])
        for run in incomplete_runs
    }

    missing_runs = []
    for model in models:
        for dataset in datasets:
            for peft in pefts:
                for data_fraction in data_fractions:
                    key = (model, dataset, peft, data_fraction)
                    if key in completed_keys:
                        continue
                    missing_runs.append(
                        {
                            "model": model,
                            "dataset": dataset,
                            "peft": peft,
                            "data_fraction": data_fraction,
                            "status": "incomplete" if key in incomplete_keys else "missing",
                        }
                    )
    return missing_runs


def build_output(
    manifest_path: Path,
    timings_path: Path,
    output_path: Path,
    models: list[str],
    datasets: list[str],
    pefts: list[str],
    max_epochs: int,
    data_fractions: list[str],
    completed_runs: list[dict[str, object]],
    incomplete_runs: list[dict[str, object]],
) -> dict[str, object]:
    timings = build_nested_timings(
        models=models,
        datasets=datasets,
        pefts=pefts,
        data_fractions=data_fractions,
        max_epochs=max_epochs,
        completed_runs=completed_runs,
    )
    missing_runs = find_missing_runs(
        models=models,
        datasets=datasets,
        pefts=pefts,
        data_fractions=data_fractions,
        completed_runs=completed_runs,
        incomplete_runs=incomplete_runs,
    )

    return {
        "source_files": {
            "manifest": str(manifest_path),
            "timings": str(timings_path),
        },
        "output_file": str(output_path),
        "max_epochs_in_log": max_epochs,
        "data_fractions_in_log": data_fractions,
        "models": models,
        "datasets": datasets,
        "pefts": pefts,
        "expected_run_count": len(models) * len(datasets) * len(pefts) * len(data_fractions),
        "completed_run_count": len(completed_runs),
        "incomplete_run_count": len(incomplete_runs),
        "missing_run_count": len(missing_runs),
        "incomplete_runs": incomplete_runs,
        "missing_runs": missing_runs,
        "timings": timings,
    }


def main():
    args = parse_args()
    models, datasets, pefts = load_manifest_grid(args.manifest_path)
    max_epochs, data_fractions, completed_runs, incomplete_runs = parse_timings(
        timings_path=args.timings_path,
        models=models,
        datasets=datasets,
        pefts=pefts,
    )

    output = build_output(
        manifest_path=args.manifest_path,
        timings_path=args.timings_path,
        output_path=args.output_path,
        models=models,
        datasets=datasets,
        pefts=pefts,
        max_epochs=max_epochs,
        data_fractions=data_fractions,
        completed_runs=completed_runs,
        incomplete_runs=incomplete_runs,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(
        f"wrote {args.output_path} "
        f"(completed={len(completed_runs)} incomplete={len(incomplete_runs)} missing={len(output['missing_runs'])})"
    )


if __name__ == "__main__":
    main()
