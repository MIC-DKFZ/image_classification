#!/usr/bin/env python3
"""
Create a minimal splits.json:
{
  "train": [...image_ids...],
  "val":   [...image_ids...],
  "test":  [...image_ids...]
}

EyePACS / Kaggle DR:
- Patient-level stratified splitting (MANDATORY)
- No option for non-stratified splits
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def extract_patient_id(image_id: str) -> str:
    s = os.path.splitext(str(image_id))[0]
    return s.split("_", 1)[0]  # "10_left" -> "10"


def build_patient_table(df: pd.DataFrame, image_col: str, label_col: str, agg: str) -> pd.DataFrame:
    df = df.copy()
    df["patient_id"] = df[image_col].map(extract_patient_id)

    g = df.groupby("patient_id")[label_col]
    if agg == "max":
        patient_label = g.max()
    elif agg == "mean_round":
        patient_label = g.mean().round().astype(int)
    else:
        raise ValueError("agg must be one of: max, mean_round")

    return patient_label.reset_index().rename(columns={label_col: "patient_label"})


def stratified_patient_split(patient_df: pd.DataFrame, train_frac: float, val_frac: float, test_frac: float, seed: int):
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    y = patient_df["patient_label"].to_numpy()

    # Split out test
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros_like(y), y))
    trainval = patient_df.iloc[trainval_idx].reset_index(drop=True)
    test = patient_df.iloc[test_idx].reset_index(drop=True)

    # Split train vs val from remaining
    val_rel = val_frac / (train_frac + val_frac)
    y_tv = trainval["patient_label"].to_numpy()
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed + 1)
    train_idx, val_idx = next(sss2.split(np.zeros_like(y_tv), y_tv))

    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    return (
        train["patient_id"].tolist(),
        val["patient_id"].tolist(),
        test["patient_id"].tolist(),
    )


def assert_no_overlap(a, b, name_a, name_b):
    inter = set(a) & set(b)
    if inter:
        raise RuntimeError(f"Patient leakage between {name_a} and {name_b}: e.g. {next(iter(inter))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--image_col", default="image")
    ap.add_argument("--label_col", default="level")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patient_label_agg", choices=["max", "mean_round"], default="max")
    args = ap.parse_args()

    df = pd.read_csv(args.labels_csv)
    df = df[[args.image_col, args.label_col]].dropna().copy()
    df[args.image_col] = df[args.image_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(int)
    df["patient_id"] = df[args.image_col].map(extract_patient_id)

    patient_df = build_patient_table(df, args.image_col, args.label_col, args.patient_label_agg)

    train_p, val_p, test_p = stratified_patient_split(
        patient_df,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    # Hard leakage checks
    assert_no_overlap(train_p, val_p, "train", "val")
    assert_no_overlap(train_p, test_p, "train", "test")
    assert_no_overlap(val_p, test_p, "val", "test")

    train_ids = df[df["patient_id"].isin(train_p)][args.image_col].tolist()
    val_ids = df[df["patient_id"].isin(val_p)][args.image_col].tolist()
    test_ids = df[df["patient_id"].isin(test_p)][args.image_col].tolist()

    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids).isdisjoint(test_ids)
    assert set(val_ids).isdisjoint(test_ids)

    out = {"train": train_ids, "val": val_ids, "test": test_ids}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {args.out_json}")
    print(f"train: patients={len(set(train_p))}, images={len(train_ids)}")
    print(f"val:   patients={len(set(val_p))}, images={len(val_ids)}")
    print(f"test:  patients={len(set(test_p))}, images={len(test_ids)}")


if __name__ == "__main__":
    main()
