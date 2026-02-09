#!/usr/bin/env python3
"""
Create splits.json for ChestXray14 dataset.

ChestXray14 has multi-label annotations but we'll use the primary/dominant finding
for single-label classification. We'll apply patient-level splitting to avoid leakage.
"""

import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def extract_patient_id(image_name: str) -> str:
    """Extract patient ID from image name (e.g., '00000001_000.png' -> '00000001')"""
    return image_name.split("_")[0]


def parse_finding_labels(finding_str: str) -> str:
    """
    Parse finding labels and return primary finding.
    Multi-label format: "Cardiomegaly|Emphysema"
    Single finding: "Cardiomegaly"
    No finding: "No Finding"
    """
    findings = [f.strip() for f in finding_str.split("|")]
    # Return first finding (or "No Finding")
    return findings[0] if findings else "No Finding"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root")
    ap.add_argument("--csv_file", default="Data_Entry_2017_v2020.csv")
    ap.add_argument("--out_json", default="splits.json", help="Output splits file")
    ap.add_argument("--out_labels", default="labels.json", help="Output labels file")
    ap.add_argument("--train_frac", type=float, default=0.6)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    root = Path(args.root)
    csv_path = root / args.csv_file

    # Load data
    df = pd.read_csv(csv_path)
    df = df[["Image Index", "Finding Labels", "Patient ID"]].copy()
    df = df.dropna()

    # Parse primary finding
    df["Primary Finding"] = df["Finding Labels"].apply(parse_finding_labels)

    # Create class mapping
    unique_findings = sorted(df["Primary Finding"].unique())
    class_to_idx = {finding: idx for idx, finding in enumerate(unique_findings)}
    df["label"] = df["Primary Finding"].map(class_to_idx)

    # Patient-level aggregation (use most common finding per patient)
    patient_df = df.groupby("Patient ID").agg({
        "label": lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }).reset_index()
    patient_df.columns = ["patient_id", "patient_label"]

    # Stratified patient split
    y = patient_df["patient_label"].to_numpy()

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed)
    trainval_idx, test_idx = next(sss1.split(np.zeros_like(y), y))

    trainval_patients = patient_df.iloc[trainval_idx]
    test_patients = patient_df.iloc[test_idx]

    # Split train vs val
    val_rel = args.val_frac / (args.train_frac + args.val_frac)
    y_tv = trainval_patients["patient_label"].to_numpy()
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=args.seed + 1)
    train_idx, val_idx = next(sss2.split(np.zeros_like(y_tv), y_tv))

    train_patients = trainval_patients.iloc[train_idx]
    val_patients = trainval_patients.iloc[val_idx]

    # Get image IDs for each split
    train_patient_ids = set(train_patients["patient_id"])
    val_patient_ids = set(val_patients["patient_id"])
    test_patient_ids = set(test_patients["patient_id"])

    # Verify no patient overlap
    assert train_patient_ids.isdisjoint(val_patient_ids)
    assert train_patient_ids.isdisjoint(test_patient_ids)
    assert val_patient_ids.isdisjoint(test_patient_ids)

    train_images = df[df["Patient ID"].isin(train_patient_ids)]["Image Index"].tolist()
    val_images = df[df["Patient ID"].isin(val_patient_ids)]["Image Index"].tolist()
    test_images = df[df["Patient ID"].isin(test_patient_ids)]["Image Index"].tolist()

    # Create labels dict
    labels_dict = dict(zip(df["Image Index"], df["label"]))

    # Save splits
    splits_out = {"train": train_images, "val": val_images, "test": test_images}
    splits_path = root / args.out_json
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_out, f, indent=2)

    # Save labels
    labels_path = root / args.out_labels
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(labels_dict, f, indent=2)

    # Save class mapping
    class_map_path = root / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)

    print(f"Wrote {splits_path}")
    print(f"Wrote {labels_path}")
    print(f"Wrote {class_map_path}")
    print(f"train: patients={len(train_patient_ids)}, images={len(train_images)}")
    print(f"val:   patients={len(val_patient_ids)}, images={len(val_images)}")
    print(f"test:  patients={len(test_patient_ids)}, images={len(test_images)}")
    print(f"Number of classes: {len(class_to_idx)}")


if __name__ == "__main__":
    main()
