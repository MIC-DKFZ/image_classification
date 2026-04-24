from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from glovita.datasets.generic_image_dataset import GenericImageDataset


def _write_rgb_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[..., 0] = 255
    Image.fromarray(image).save(path)


def test_generic_image_dataset_with_splits_json_and_json_labels(tmp_path: Path):
    images_dir = tmp_path / "images"
    _write_rgb_image(images_dir / "cats" / "cat_001.jpg")
    _write_rgb_image(images_dir / "dogs" / "dog_001.jpg")

    (tmp_path / "splits.json").write_text(
        json.dumps({"train": ["cats/cat_001.jpg"], "val": ["dogs/dog_001.jpg"]}),
        encoding="utf-8",
    )
    (tmp_path / "labels.json").write_text(
        json.dumps({"cats/cat_001.jpg": 0, "dogs/dog_001.jpg": 1}),
        encoding="utf-8",
    )

    train_ds = GenericImageDataset(
        tmp_path,
        split="train",
        split_source="splits_json",
        label_source="json",
    )
    val_ds = GenericImageDataset(
        tmp_path,
        split="val",
        split_source="splits_json",
        label_source="json",
    )

    train_x, train_y = train_ds[0]
    val_x, val_y = val_ds[0]
    assert isinstance(train_x, torch.Tensor)
    assert isinstance(val_x, torch.Tensor)
    assert train_x.shape == (3, 8, 8)
    assert int(train_y) == 0
    assert int(val_y) == 1


def test_generic_image_dataset_with_subdirs_and_folder_labels(tmp_path: Path):
    _write_rgb_image(tmp_path / "images" / "train" / "airplane" / "img_001.png")
    _write_rgb_image(tmp_path / "images" / "val" / "ship" / "img_002.png")

    dataset = GenericImageDataset(
        tmp_path,
        split="train",
        split_source="subdirs",
        label_source="folder",
        class_names=["airplane", "ship"],
    )
    image, label = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 8, 8)
    assert int(label) == 0
