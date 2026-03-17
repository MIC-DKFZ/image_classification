#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_INPUT = "global_train_class_threshold_counts.json"
DEFAULT_OUTPUT = "global_train_class_threshold_counts_grouped_bar.png"
DEFAULT_Y_MAX = 100
DEFAULT_X_MAX_STEP = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped bars for global train threshold counts. "
            "Each x-axis group is a threshold step and each bar is a dataset."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help=f"Input JSON path. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"Output image path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--max-x-labels",
        type=int,
        default=25,
        help="Maximum number of x-axis labels to show. Defaults to 25.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=DEFAULT_Y_MAX,
        help=f"Upper limit for the y-axis. Defaults to {DEFAULT_Y_MAX}.",
    )
    parser.add_argument(
        "--x-max-step",
        type=int,
        default=DEFAULT_X_MAX_STEP,
        help=f"Maximum threshold step to include on the x-axis. Defaults to {DEFAULT_X_MAX_STEP}.",
    )
    return parser.parse_args()


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(path, cwd):
    return path if path.is_absolute() else cwd / path


def build_positions(steps, datasets):
    x = list(range(len(steps)))
    group_width = 0.8
    bar_width = group_width / max(len(datasets), 1)
    midpoint = (len(datasets) - 1) / 2.0

    dataset_positions = {}
    for dataset_index, dataset in enumerate(datasets):
        offset = (dataset_index - midpoint) * bar_width
        dataset_positions[dataset] = [value + offset for value in x]

    return x, bar_width, dataset_positions


def choose_tick_positions(steps, max_x_labels):
    if not steps:
        return []
    if len(steps) <= max_x_labels:
        return list(range(len(steps)))

    stride = max(1, len(steps) // (max_x_labels - 1))
    positions = list(range(0, len(steps), stride))
    if positions[-1] != len(steps) - 1:
        positions.append(len(steps) - 1)
    return positions


def plot_grouped_bars(data, output_path, max_x_labels, y_max, x_max_step):
    steps = sorted(int(step) for step in data.keys() if int(step) <= x_max_step)
    if not steps:
        raise ValueError(f"No threshold steps at or below {x_max_step} were found in the input data")
    first_step_key = str(steps[0])
    datasets = list(data[first_step_key].keys())

    x, bar_width, dataset_positions = build_positions(steps, datasets)

    fig_width = min(max(24.0, len(steps) * 0.015), 160.0)
    fig, ax = plt.subplots(figsize=(fig_width, 10.0), constrained_layout=True)

    for dataset in datasets:
        heights = [data[str(step)][dataset] for step in steps]
        ax.bar(
            dataset_positions[dataset],
            heights,
            width=bar_width,
            label=dataset,
            linewidth=0,
        )

    tick_positions = choose_tick_positions(steps, max_x_labels)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(steps[index]) for index in tick_positions], rotation=45, ha="right")

    ax.set_xlabel("Threshold Step")
    ax.set_ylabel("Remaining Classes (%)")
    ax.set_title("Train Class Retention per Dataset Above Threshold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", ncols=3, fontsize=9)
    ax.set_xlim(-0.5, len(steps) - 0.5)
    ax.set_ylim(0, y_max)
    ax.margins(x=0.001)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    cwd = Path.cwd()
    input_path = resolve_path(args.input, cwd)
    output_path = resolve_path(args.output, cwd)

    data = load_json(input_path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{input_path} must contain a non-empty JSON object")

    plot_grouped_bars(data, output_path, args.max_x_labels, args.y_max, args.x_max_step)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
