#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_CONDA_PATH = "/dkfz/cluster/gpu/data/mic_data_common/synergy_unit/scripts/submit.sh"
DEFAULT_OUTPUT_JSON = "synergy_unit/data/max_batch_sizes.json"
DEFAULT_JOB_ID = "max_batch_size_probe"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit a single cluster job that probes maximum usable batch sizes."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Value passed to the probe worker as --data-dir.",
    )
    parser.add_argument(
        "--output-json",
        default=DEFAULT_OUTPUT_JSON,
        help=f"Where the probe worker writes results. Defaults to {DEFAULT_OUTPUT_JSON}.",
    )
    parser.add_argument(
        "--conda-path",
        default=DEFAULT_CONDA_PATH,
        help="Path to the cluster submission wrapper script.",
    )
    parser.add_argument(
        "--job-id",
        default=DEFAULT_JOB_ID,
        help=f"Cluster job id used for submission. Defaults to {DEFAULT_JOB_ID}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the submission command instead of executing it.",
    )
    parser.add_argument(
        "--worker-arg",
        action="append",
        default=[],
        help=(
            "Additional argument to forward to submit/find_max_batch_sizes.py. "
            "Can be passed multiple times."
        ),
    )
    return parser.parse_args()


def shell_quote(value: str) -> str:
    return subprocess.list2cmdline([value])


def build_python_command(args: argparse.Namespace) -> str:
    parts = [
        "python",
        "submit/find_max_batch_sizes.py",
        f"--data-dir={args.data_dir}",
        f"--output-json={args.output_json}",
    ]
    parts.extend(args.worker_arg)
    return " ".join(shell_quote(part) for part in parts)


def build_submit_command(args: argparse.Namespace, python_command: str) -> list[str]:
    return [
        args.conda_path,
        "-i",
        args.job_id,
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


def main() -> int:
    args = parse_args()
    python_command = build_python_command(args)
    submit_command = build_submit_command(args, python_command)

    if args.dry_run:
        print(format_submit_command(submit_command))
        return 0

    completed = subprocess.run(
        submit_command,
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).resolve().parents[1],
    )
    if completed.returncode != 0:
        print(
            completed.stderr.strip() or completed.stdout.strip() or "Submission failed without output."
        )
        return completed.returncode

    print(completed.stdout.strip() or f"Submitted job {args.job_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
