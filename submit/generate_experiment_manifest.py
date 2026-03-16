#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from submit.experiment_manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic experiment manifest JSON."
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output path for the generated manifest JSON.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the manifest summary without writing a JSON file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.summary_only and not args.output:
        raise SystemExit("An output path is required unless --summary-only is set.")

    manifest = build_manifest()
    print(f"Grid version: {manifest['grid_version']}")
    print(f"Experiments: {manifest['experiment_count']}")
    print(f"Estimated GPU-hours: {manifest['estimated_gpu_hours']:.2f}")

    if args.summary_only:
        return 0

    output_path = Path(args.output)
    write_manifest(output_path, manifest)
    print(f"Wrote manifest to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
