from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class GenericImageDataset(Dataset):
    """Reusable image dataset for common folder/CSV/JSON layouts.

    Supported split sources:
    - ``splits_json``: sample ids come from a split file such as:
      ``{"train": [...], "val": [...], "test": [...]}``
    - ``subdirs``: samples are discovered by scanning
      ``root/images_dir/<split_name>/...``

    Supported label sources:
    - ``folder``: infer the class name from the first label directory
    - ``json``: read a path->label mapping from JSON
    - ``csv``: read a path/label table from CSV
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform=None,
        *,
        images_dir: str = "images",
        split_source: Literal["splits_json", "subdirs"] = "splits_json",
        label_source: Literal["folder", "json", "csv"] = "json",
        split_file: str = "splits.json",
        labels_file: str = "labels.json",
        path_column: str = "path",
        label_column: str = "label",
        train_split_name: str = "train",
        val_split_name: str = "val",
        test_split_name: str = "test",
        allowed_exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"),
        class_names: list[str] | None = None,
        task: Literal["Classification", "Regression"] = "Classification",
        strict: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.images_root = self.root / images_dir
        self.split_source = split_source
        self.label_source = label_source
        self.path_column = path_column
        self.label_column = label_column
        self.allowed_exts = tuple(ext.lower() for ext in allowed_exts)
        self.task = task
        self.strict = strict
        self.split_names = {
            "train": train_split_name,
            "val": val_split_name,
            "test": test_split_name,
        }

        split_name = self._resolve_split_name(split)
        self._split_file = self.root / split_file
        self._labels_file = self.root / labels_file
        self._external_label_map = self._load_external_label_map()
        self.class_to_idx = self._build_class_to_idx(class_names)
        samples = self._build_samples(split_name)

        self.img_paths = [path for path, _ in samples]
        if self.task == "Regression":
            self.labels = np.asarray([float(label) for _, label in samples], dtype=np.float32)
        else:
            self.labels = np.asarray([int(label) for _, label in samples], dtype=np.int64)

    def _resolve_split_name(self, split: str) -> str:
        if split not in self.split_names:
            raise ValueError(f"Unknown split {split!r}. Expected one of {sorted(self.split_names)}.")
        return self.split_names[split]

    def _build_samples(self, split_name: str) -> list[tuple[Path, int | float]]:
        if self.split_source == "subdirs":
            return self._samples_from_subdirs(split_name)
        return self._samples_from_split_file(split_name)

    def _samples_from_subdirs(self, split_name: str) -> list[tuple[Path, int | float]]:
        split_dir = self.images_root / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples: list[tuple[Path, int | float]] = []
        for path in sorted(split_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in self.allowed_exts:
                continue
            label = self._resolve_label(path, split_name=split_name, split_root=split_dir)
            if label is None:
                continue
            samples.append((path, label))

        if not samples:
            raise ValueError(f"No images found for split {split_name!r} under {split_dir}.")
        return samples

    def _samples_from_split_file(self, split_name: str) -> list[tuple[Path, int | float]]:
        if not self._split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self._split_file}")
        with self._split_file.open("r", encoding="utf-8") as f:
            split_payload = json.load(f)
        if split_name not in split_payload:
            raise ValueError(
                f"Split {split_name!r} not present in {self._split_file}. "
                f"Available keys: {sorted(split_payload)}."
            )

        samples: list[tuple[Path, int | float]] = []
        for sample_id in split_payload[split_name]:
            path = self._resolve_path_from_id(str(sample_id), split_name=split_name)
            if path is None:
                if self.strict:
                    raise FileNotFoundError(
                        f"Could not resolve sample {sample_id!r} under {self.root} / {self.images_root}."
                    )
                continue
            label = self._resolve_label(path, split_name=split_name, sample_id=str(sample_id))
            if label is None:
                continue
            samples.append((path, label))

        if not samples:
            raise ValueError(f"No valid samples resolved for split {split_name!r} from {self._split_file}.")
        return samples

    def _load_external_label_map(self) -> dict[str, int | float] | None:
        if self.label_source == "folder":
            return None
        if not self._labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self._labels_file}")

        if self.label_source == "json":
            with self._labels_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                raise TypeError(f"{self._labels_file} must contain a JSON object for label_source='json'.")
            return {str(k): v for k, v in payload.items()}

        with self._labels_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if self.path_column not in reader.fieldnames or self.label_column not in reader.fieldnames:
                raise KeyError(
                    f"CSV labels file {self._labels_file} must contain columns "
                    f"{self.path_column!r} and {self.label_column!r}."
                )
            mapping: dict[str, int | float] = {}
            for row in reader:
                mapping[str(row[self.path_column])] = row[self.label_column]
            return mapping

    def _build_class_to_idx(self, class_names: list[str] | None) -> dict[str, int]:
        if self.task != "Classification" or self.label_source != "folder":
            return {}
        if class_names is not None:
            return {name: idx for idx, name in enumerate(class_names)}

        if self.split_source == "subdirs":
            roots = [self.images_root / split_name for split_name in self.split_names.values()]
            discovered = {
                path.name
                for split_root in roots
                if split_root.exists()
                for path in split_root.iterdir()
                if path.is_dir()
            }
        else:
            if not self._split_file.exists():
                discovered = set()
            else:
                with self._split_file.open("r", encoding="utf-8") as f:
                    split_payload = json.load(f)
                discovered = set()
                for values in split_payload.values():
                    for sample_id in values:
                        class_name = self._infer_class_name_from_path_id(str(sample_id))
                        if class_name is not None:
                            discovered.add(class_name)
        if not discovered:
            raise ValueError(
                "Could not infer folder-based class names automatically. "
                "Pass class_names explicitly or use label_source='json'/'csv'."
            )
        return {name: idx for idx, name in enumerate(sorted(discovered))}

    def _resolve_path_from_id(self, sample_id: str, *, split_name: str) -> Path | None:
        sample_path = Path(sample_id)
        candidates: list[Path] = []
        if sample_path.is_absolute():
            candidates.append(sample_path)
        else:
            candidates.extend(
                [
                    self.root / sample_path,
                    self.images_root / sample_path,
                    self.images_root / split_name / sample_path,
                ]
            )

        resolved: list[Path] = []
        for base in candidates:
            if base.suffix:
                resolved.append(base)
            else:
                resolved.extend(base.with_suffix(ext) for ext in self.allowed_exts)

        for candidate in resolved:
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _resolve_label(
        self,
        path: Path,
        *,
        split_name: str,
        split_root: Path | None = None,
        sample_id: str | None = None,
    ) -> int | float | None:
        if self.label_source == "folder":
            class_name = self._infer_class_name_from_path(path, split_root=split_root)
            if class_name is None:
                raise ValueError(f"Could not infer class label from path {path}.")
            if class_name not in self.class_to_idx:
                raise KeyError(f"Unknown folder class {class_name!r}; known classes: {sorted(self.class_to_idx)}")
            return self.class_to_idx[class_name]

        assert self._external_label_map is not None
        lookup_keys = self._candidate_label_keys(path, split_name=split_name, sample_id=sample_id)
        for key in lookup_keys:
            if key in self._external_label_map:
                raw_label = self._external_label_map[key]
                return float(raw_label) if self.task == "Regression" else int(raw_label)

        if self.strict:
            raise KeyError(
                f"Could not find a label for {path} in {self._labels_file}. "
                f"Tried keys: {lookup_keys}"
            )
        return None

    def _candidate_label_keys(self, path: Path, *, split_name: str, sample_id: str | None) -> list[str]:
        candidates: list[str] = []
        if sample_id is not None:
            sample_path = Path(sample_id)
            candidates.extend([str(sample_path), sample_path.as_posix()])
            if sample_path.suffix:
                candidates.append(sample_path.stem)

        rel_candidates = []
        for base in (self.root, self.images_root, self.images_root / split_name):
            try:
                rel_candidates.append(path.relative_to(base).as_posix())
            except ValueError:
                pass

        for rel in rel_candidates:
            candidates.append(rel)
            rel_path = Path(rel)
            if rel_path.suffix:
                candidates.append(rel_path.stem)
            else:
                candidates.extend(Path(rel).with_suffix(ext).as_posix() for ext in self.allowed_exts)

        candidates.append(path.name)
        candidates.append(path.stem)

        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            if candidate not in seen:
                seen.add(candidate)
                ordered.append(candidate)
        return ordered

    def _infer_class_name_from_path_id(self, sample_id: str) -> str | None:
        parts = Path(sample_id).parts
        split_names = set(self.split_names.values())
        for idx, part in enumerate(parts):
            if part in split_names and idx + 1 < len(parts):
                return parts[idx + 1]
        if parts:
            return parts[0]
        return None

    def _infer_class_name_from_path(self, path: Path, *, split_root: Path | None) -> str | None:
        if split_root is not None:
            rel = path.relative_to(split_root)
            if rel.parts:
                return rel.parts[0]

        rel_to_images = None
        try:
            rel_to_images = path.relative_to(self.images_root)
        except ValueError:
            pass

        if rel_to_images is not None and rel_to_images.parts:
            first = rel_to_images.parts[0]
            if first in set(self.split_names.values()) and len(rel_to_images.parts) > 1:
                return rel_to_images.parts[1]
            return first
        return None

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image_np = np.ascontiguousarray(np.array(image), dtype=np.uint8)
            image = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()

        label = self.labels[idx]
        if self.task == "Regression":
            target = torch.tensor(float(label), dtype=torch.float32)
        else:
            target = int(label)
        return image, target
