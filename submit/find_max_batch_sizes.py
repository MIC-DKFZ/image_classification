#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from tqdm.auto import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synergy_unit.helpers.generate_samples_per_class_subsets import (
    build_train_samples_by_class,
    load_label_mapping,
)


DEFAULT_OUTPUT_JSON = Path("synergy_unit/data/max_batch_sizes.json")
DEFAULT_SAMPLES_PER_CLASS = 50
DEFAULT_TRIAL = 0
DEFAULT_MAX_EPOCHS = 1
DEFAULT_MAX_BATCH_SIZE_CAP = 4096
DEFAULT_TIMEOUT_SECONDS = 1800
DEFAULT_LIMIT_TRAIN_BATCHES = 1
DEFAULT_LIMIT_VAL_BATCHES = 1
DEFAULT_JOB_LABEL = "batch_size_probe"
BATCH_SIZE_GRANULARITY = 16
VERBOSE = False
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
MODELS = [
    "supervised",
    "mae_timm",
    "dinov3_reference",
]
PEFTS = [
    "adapt_former",
    "full_finetuning",
    "gps",
    "linear_probing",
    "lora",
    "vera",
    "visual_prompt_tuning",
]
OOM_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "cudnn_status_alloc_failed",
    "cublas_status_alloc_failed",
)


@dataclass(frozen=True)
class ProbeCombination:
    dataset: str
    model: str
    peft: str

    @property
    def key(self) -> str:
        return f"{self.dataset}__{self.model}__{self.peft}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Determine the maximum usable batch size for each dataset/model/peft "
            "combination by launching short training subprocesses."
        )
    )
    parser.add_argument("--data-dir", required=True, help="Value passed to Hydra as data_dir=...")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_JSON,
        help=f"Where to write the incremental results JSON. Defaults to {DEFAULT_OUTPUT_JSON}.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=DEFAULT_SAMPLES_PER_CLASS,
        help=f"Subset size to probe against. Defaults to {DEFAULT_SAMPLES_PER_CLASS}.",
    )
    parser.add_argument(
        "--trial",
        type=int,
        default=DEFAULT_TRIAL,
        help=f"Subset trial index to probe against. Defaults to {DEFAULT_TRIAL}.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help=f"Epochs per probe run. Defaults to {DEFAULT_MAX_EPOCHS}.",
    )
    parser.add_argument(
        "--max-batch-size-cap",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE_CAP,
        help=f"Upper batch-size cap during search. Defaults to {DEFAULT_MAX_BATCH_SIZE_CAP}.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Timeout for a single probe subprocess. Defaults to {DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--limit-train-batches",
        type=int,
        default=DEFAULT_LIMIT_TRAIN_BATCHES,
        help=f"Limit train batches per epoch during probing. Defaults to {DEFAULT_LIMIT_TRAIN_BATCHES}.",
    )
    parser.add_argument(
        "--limit-val-batches",
        type=int,
        default=DEFAULT_LIMIT_VAL_BATCHES,
        help=f"Limit val batches per epoch during probing. Defaults to {DEFAULT_LIMIT_VAL_BATCHES}.",
    )
    parser.add_argument(
        "--temp-root",
        type=Path,
        help="Optional parent directory for temporary probe run directories.",
    )
    parser.add_argument(
        "--initial-batch-size",
        type=int,
        help="Optional override for the initial batch size used for all combinations.",
    )
    parser.add_argument(
        "--job-label",
        default=DEFAULT_JOB_LABEL,
        help=f"Run-name prefix for probe subprocesses. Defaults to {DEFAULT_JOB_LABEL}.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSON by skipping combinations already marked successful.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed per-probe logging. Defaults to off.",
    )
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str, *, always: bool = False) -> None:
    if always or VERBOSE:
        tqdm.write(f"[batch-probe] {message}")


def load_data_batch_size(dataset: str) -> int:
    config_path = Path(__file__).resolve().parents[1] / "cli_configs" / "data" / f"{dataset}.yaml"
    cfg = OmegaConf.load(config_path)
    batch_size = cfg["data"]["module"]["batch_size"]
    return int(batch_size)


def load_dataset_root_name(dataset: str) -> str:
    config_path = Path(__file__).resolve().parents[1] / "cli_configs" / "data" / f"{dataset}.yaml"
    cfg = OmegaConf.load(config_path)
    raw_cfg = OmegaConf.to_container(cfg, resolve=False)
    data_root_dir = str(raw_cfg["data"]["module"]["data_root_dir"])
    return Path(data_root_dir).name


def resolve_mic_data_common_root(data_dir: str) -> Path | None:
    data_path = Path(data_dir).resolve()
    for candidate in [data_path] + list(data_path.parents):
        if candidate.name == "mic_data_common":
            return candidate
    return None


def build_model_overrides(args: argparse.Namespace, combo: ProbeCombination) -> list[str]:
    overrides: list[str] = []
    if combo.model == "dinov3_reference":
        mirrored_root = resolve_mic_data_common_root(args.data_dir)
        if mirrored_root is not None:
            mirrored_weight_dir = mirrored_root / "dinov3" / "weights"
            if mirrored_weight_dir.exists():
                overrides.append(f"model.weight_dir={mirrored_weight_dir}")
    return overrides


def build_combinations() -> list[ProbeCombination]:
    combinations = [
        ProbeCombination(dataset=dataset, model=model, peft=peft)
        for dataset in DATASETS
        for model in MODELS
        for peft in PEFTS
    ]
    return sorted(combinations, key=lambda combo: (combo.dataset, combo.model, combo.peft))


def load_results(args: argparse.Namespace, combinations: list[ProbeCombination]) -> dict[str, Any]:
    if args.resume and args.output_json.exists():
        raw_results = json.loads(args.output_json.read_text(encoding="utf-8"))
        payload = {"combinations": [asdict(combo) for combo in combinations], "results": {}}
        for item in raw_results:
            key = f"{item['dataset']}__{item['model']}__{item['peft']}"
            max_batch_size = item.get("max_batch_size")
            if max_batch_size is not None:
                payload["results"][key] = {"status": "success", "max_batch_size": max_batch_size}
        return payload

    return {
        "combinations": [asdict(combo) for combo in combinations],
        "results": {},
    }


def build_summary_results(payload: dict[str, Any], combinations: list[ProbeCombination]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for combo in combinations:
        entry = payload["results"].get(combo.key)
        if entry is None:
            continue
        if entry.get("status") not in {"success", "error", "missing_dataset_dir"}:
            continue
        summary.append(
            {
                "dataset": combo.dataset,
                "model": combo.model,
                "peft": combo.peft,
                "max_batch_size": entry.get("max_batch_size"),
            }
        )
    return summary


def write_results(path: Path, payload: dict[str, Any], combinations: list[ProbeCombination]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(build_summary_results(payload, combinations), indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def is_oom_failure(returncode: int, combined_output: str) -> bool:
    lowered = combined_output.lower()
    if any(marker in lowered for marker in OOM_MARKERS):
        return True
    return returncode in {137, 139}


def granularity_floor(value: int) -> int:
    if value <= BATCH_SIZE_GRANULARITY:
        return value
    remainder = value % BATCH_SIZE_GRANULARITY
    return value if remainder == 0 else value - remainder


def choose_probe_batch_size(initial_batch_size: int) -> int:
    return max(BATCH_SIZE_GRANULARITY, granularity_floor(initial_batch_size))


def build_probe_command(
    args: argparse.Namespace,
    combo: ProbeCombination,
    batch_size: int,
    exp_dir: Path,
    split_file: str,
) -> list[str]:
    run_name = f"{args.job_label}__{combo.dataset}__{combo.model}__{combo.peft}__bs{batch_size}"
    return [
        sys.executable,
        "main.py",
        "env=cluster",
        f"+wandb.name={run_name}",
        f"model={combo.model}",
        f"data={combo.dataset}",
        f"peft={combo.peft}",
        f"trainer.max_epochs={args.max_epochs}",
        f"data.module.batch_size={batch_size}",
        "data.module.data_fraction=null",
        f"+data.module.split_file={split_file}",
        f"+dataset={combo.dataset}",
        "+data_fraction=null",
        f"+samples_per_class={args.samples_per_class}",
        f"+trial={args.trial}",
        "trainer.enable_checkpointing=false",
        "trainer.enable_progress_bar=false",
        f"+trainer.limit_train_batches={args.limit_train_batches}",
        f"+trainer.limit_val_batches={args.limit_val_batches}",
        "data.module.num_workers=0",
        f"data_dir={args.data_dir}",
        f"exp_dir={exp_dir}",
        *build_model_overrides(args=args, combo=combo),
    ]


def tail_text(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def run_probe_once(
    args: argparse.Namespace,
    combo: ProbeCombination,
    batch_size: int,
) -> dict[str, Any]:
    tmp_dir = tempfile.mkdtemp(
        prefix=f"batch_probe_{combo.dataset}_{combo.model}_{combo.peft}_{batch_size}_",
        dir=str(args.temp_root) if args.temp_root else None,
    )
    tmp_path = Path(tmp_dir)
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    exp_dir = tmp_path / "exp"
    split_file = resolve_probe_split_file(args=args, combo=combo, temp_dir=tmp_path)
    command = build_probe_command(
        args=args,
        combo=combo,
        batch_size=batch_size,
        exp_dir=exp_dir,
        split_file=split_file,
    )
    log(
        "starting probe "
        f"dataset={combo.dataset} model={combo.model} peft={combo.peft} "
        f"batch_size={batch_size}"
    )
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["WANDB_SILENT"] = "true"
    env["PYTHONUNBUFFERED"] = "1"

    start = time.monotonic()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=False,
                timeout=args.timeout_seconds,
            )
        stdout_text = stdout_path.read_text(encoding="utf-8")
        stderr_text = stderr_path.read_text(encoding="utf-8")
        combined_output = stdout_text + "\n" + stderr_text
        oom = is_oom_failure(completed.returncode, combined_output)
        status = "success" if completed.returncode == 0 else ("oom" if oom else "error")
        log(
            "finished probe "
            f"dataset={combo.dataset} model={combo.model} peft={combo.peft} "
            f"batch_size={batch_size} status={status} "
            f"returncode={completed.returncode} duration_seconds={round(time.monotonic() - start, 3)}"
        )
        return {
            "batch_size": batch_size,
            "status": status,
            "returncode": completed.returncode,
            "duration_seconds": round(time.monotonic() - start, 3),
            "stdout_tail": tail_text(stdout_text),
            "stderr_tail": tail_text(stderr_text),
            "command": command,
            "split_file": split_file,
        }
    except subprocess.TimeoutExpired as exc:
        stdout_text = exc.stdout or ""
        stderr_text = exc.stderr or ""
        log(
            "finished probe "
            f"dataset={combo.dataset} model={combo.model} peft={combo.peft} "
            f"batch_size={batch_size} status=timeout "
            f"duration_seconds={round(time.monotonic() - start, 3)}"
        )
        return {
            "batch_size": batch_size,
            "status": "timeout",
            "returncode": None,
            "duration_seconds": round(time.monotonic() - start, 3),
            "stdout_tail": tail_text(stdout_text),
            "stderr_tail": tail_text(stderr_text),
            "command": command,
            "split_file": split_file,
        }
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def build_temporary_probe_split(
    args: argparse.Namespace,
    combo: ProbeCombination,
    dataset_dir: Path,
    output_path: Path,
) -> None:
    splits = json.loads((dataset_dir / "splits.json").read_text(encoding="utf-8"))
    labels, ordered_class_names, id_to_class = load_label_mapping(dataset_dir)
    if labels is None:
        raise FileNotFoundError(
            f"Could not derive labels for {dataset_dir}: expected labels.json/class_map.json or trainLabels.csv"
        )

    samples_by_class, ordered_class_names = build_train_samples_by_class(
        labels, ordered_class_names, id_to_class, splits
    )
    rng = random.Random(
        f"{combo.dataset}:{combo.model}:{combo.peft}:{args.samples_per_class}:{args.trial}"
    )
    selected_train_samples = []
    for class_name in ordered_class_names:
        class_samples = samples_by_class[class_name]
        if not class_samples:
            raise ValueError(f"{dataset_dir.name}: class '{class_name}' has no train samples")
        if len(class_samples) >= args.samples_per_class:
            selected_train_samples.extend(rng.sample(class_samples, args.samples_per_class))
        else:
            selected_train_samples.extend(rng.choices(class_samples, k=args.samples_per_class))

    rng.shuffle(selected_train_samples)
    payload = {
        "train": selected_train_samples,
        "val": list(splits["val"]),
        "test": list(splits["test"]),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def resolve_probe_split_file(args: argparse.Namespace, combo: ProbeCombination, temp_dir: Path) -> str:
    dataset_dir = Path(args.data_dir) / load_dataset_root_name(combo.dataset)
    subset_path = (
        dataset_dir / "subsets" / f"samples_per_class_{args.samples_per_class}_trial_{args.trial}.json"
    )
    if subset_path.exists():
        return str(subset_path)

    temp_split_path = temp_dir / f"samples_per_class_{args.samples_per_class}_trial_{args.trial}.json"
    build_temporary_probe_split(
        args=args,
        combo=combo,
        dataset_dir=dataset_dir,
        output_path=temp_split_path,
    )
    return str(temp_split_path)


def append_attempt(
    payload: dict[str, Any],
    combo: ProbeCombination,
    entry: dict[str, Any],
    attempt: dict[str, Any],
) -> None:
    entry.setdefault("attempts", []).append(attempt)
    if attempt["status"] == "success":
        current = entry.get("current_max_batch_size")
        if current is None or attempt["batch_size"] > current:
            entry["current_max_batch_size"] = attempt["batch_size"]
    entry["last_attempt_at"] = now_iso()
    payload["results"][combo.key] = entry


def find_max_batch_size(
    args: argparse.Namespace,
    combo: ProbeCombination,
    payload: dict[str, Any],
    entry: dict[str, Any],
    combinations: list[ProbeCombination],
) -> tuple[int | None, str | None]:
    initial_batch_size = choose_probe_batch_size(entry["initial_batch_size"])
    current = initial_batch_size
    low_success: int | None = None
    high_failure: int | None = None

    while True:
        attempt = run_probe_once(args=args, combo=combo, batch_size=current)
        append_attempt(payload, combo, entry, attempt)

        if attempt["status"] == "success":
            low_success = current
            break
        if attempt["status"] == "oom":
            high_failure = current
            if current <= BATCH_SIZE_GRANULARITY:
                return None, f"oom_at_batch_size_{BATCH_SIZE_GRANULARITY}"

            next_current = max(BATCH_SIZE_GRANULARITY, granularity_floor(current // 2))
            if next_current == current:
                next_current = max(BATCH_SIZE_GRANULARITY, current - BATCH_SIZE_GRANULARITY)
            current = next_current
            continue
        return None, f"probe_{attempt['status']}_at_batch_size_{current}"

    while low_success is not None and low_success < args.max_batch_size_cap:
        candidate = low_success * 2
        if candidate > args.max_batch_size_cap:
            candidate = args.max_batch_size_cap
        candidate = choose_probe_batch_size(candidate)
        if candidate <= low_success:
            break

        attempt = run_probe_once(args=args, combo=combo, batch_size=candidate)
        append_attempt(payload, combo, entry, attempt)
        if attempt["status"] == "success":
            low_success = candidate
            continue
        if attempt["status"] == "oom":
            high_failure = candidate
            break
        return low_success, f"probe_{attempt['status']}_at_batch_size_{candidate}"

    if high_failure is None:
        return low_success, None

    while low_success is not None and high_failure - low_success > BATCH_SIZE_GRANULARITY:
        candidate = (low_success + high_failure) // 2
        if low_success >= BATCH_SIZE_GRANULARITY:
            candidate = granularity_floor(candidate)
            if candidate <= low_success:
                candidate = low_success + BATCH_SIZE_GRANULARITY
        if candidate >= high_failure:
            break

        attempt = run_probe_once(args=args, combo=combo, batch_size=candidate)
        append_attempt(payload, combo, entry, attempt)
        if attempt["status"] == "success":
            low_success = candidate
        elif attempt["status"] == "oom":
            high_failure = candidate
        else:
            return low_success, f"probe_{attempt['status']}_at_batch_size_{candidate}"

    return low_success, None


def main() -> int:
    global VERBOSE
    args = parse_args()
    VERBOSE = args.verbose
    combinations = build_combinations()
    payload = load_results(args, combinations)
    log(
        f"loaded {len(combinations)} combinations "
        f"samples_per_class={args.samples_per_class} trial={args.trial} "
        f"output_json={args.output_json}",
        always=args.verbose,
    )

    with tqdm(total=len(combinations), desc="Batch-size combos", unit="combo") as progress_bar:
        for index, combo in enumerate(combinations, start=1):
            progress_bar.set_postfix_str(f"{combo.dataset}/{combo.model}/{combo.peft}")
            existing = payload["results"].get(combo.key)
            if args.resume and existing and existing.get("status") == "success":
                log(
                    f"skipping completed combination {index}/{len(combinations)} "
                    f"dataset={combo.dataset} model={combo.model} peft={combo.peft}"
                )
                progress_bar.update(1)
                continue

            dataset_dir = Path(args.data_dir) / load_dataset_root_name(combo.dataset)
            subset_path = (
                dataset_dir
                / "subsets"
                / f"samples_per_class_{args.samples_per_class}_trial_{args.trial}.json"
            )

            initial_batch_size = (
                int(args.initial_batch_size)
                if args.initial_batch_size is not None
                else load_data_batch_size(combo.dataset)
            )
            entry = {
                "dataset": combo.dataset,
                "model": combo.model,
                "peft": combo.peft,
                "split_file": str(subset_path),
                "initial_batch_size": initial_batch_size,
                "status": "running",
                "attempts": existing.get("attempts", []) if existing else [],
                "current_max_batch_size": existing.get("current_max_batch_size") if existing else None,
                "started_at": existing.get("started_at", now_iso()) if existing else now_iso(),
            }
            log(
                f"processing combination {index}/{len(combinations)} "
                f"dataset={combo.dataset} model={combo.model} peft={combo.peft} "
                f"initial_batch_size={initial_batch_size}"
            )
            payload["results"][combo.key] = entry

            if not dataset_dir.exists():
                log(f"missing dataset directory for {combo.key}: {dataset_dir}")
                entry["status"] = "missing_dataset_dir"
                entry["error"] = f"Missing dataset directory: {dataset_dir}"
                payload["results"][combo.key] = entry
                log(
                    f"failed combo dataset={combo.dataset} model={combo.model} "
                    f"peft={combo.peft} error=missing_dataset_dir",
                    always=True,
                )
                write_results(args.output_json, payload, combinations)
                progress_bar.update(1)
                continue

            max_batch_size, error = find_max_batch_size(
                args=args,
                combo=combo,
                payload=payload,
                entry=entry,
                combinations=combinations,
            )
            entry["finished_at"] = now_iso()
            if error is None:
                entry["status"] = "success"
                entry["max_batch_size"] = max_batch_size
                log(
                    f"completed combination {index}/{len(combinations)} "
                    f"dataset={combo.dataset} model={combo.model} peft={combo.peft} "
                    f"max_batch_size={max_batch_size}"
                )
                log(
                    f"found max batch size dataset={combo.dataset} model={combo.model} "
                    f"peft={combo.peft} max_batch_size={max_batch_size}",
                    always=True,
                )
            else:
                entry["status"] = "error"
                entry["max_batch_size"] = None
                entry["error"] = error
                log(
                    f"combination failed {index}/{len(combinations)} "
                    f"dataset={combo.dataset} model={combo.model} peft={combo.peft} "
                    f"max_batch_size={max_batch_size} error={error}"
                )
                log(
                    f"failed combo dataset={combo.dataset} model={combo.model} "
                    f"peft={combo.peft} error={error}",
                    always=True,
                )
            payload["results"][combo.key] = entry
            write_results(args.output_json, payload, combinations)
            progress_bar.update(1)

    log("all combinations processed", always=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
