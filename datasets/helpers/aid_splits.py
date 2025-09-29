# build_splits_aid.py
import json, sys, random
from pathlib import Path
from collections import defaultdict

def main(root, train_ratio=0.5, folds=5, seed0=0):
    root = Path(root)
    labels_path = root / "labels.json"
    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        # group by class index
        by_cls = defaultdict(list)
        for k, v in labels.items():
            by_cls[v].append(k)
    else:
        # Fall back: infer classes from folders (airport/..., bareland/..., etc.)
        img_dir = root / "images"
        class_names = sorted([d.name for d in img_dir.iterdir() if d.is_dir()])
        name2idx = {c: i for i, c in enumerate(class_names)}
        by_cls = defaultdict(list)
        for c in class_names:
            for p in (img_dir / c).glob("*.*"):
                if p.is_file():
                    by_cls[name2idx[c]].append(f"{c}/{p.stem}")

        # also write class_map.json for reproducibility
        (root / "class_map.json").write_text(json.dumps(name2idx, indent=2))

    out = []
    for s in range(seed0, seed0 + folds):
        random.seed(s)
        train, val = [], []
        for _, ids in by_cls.items():
            ids = ids[:]  # copy
            random.shuffle(ids)
            k = int(len(ids) * train_ratio)
            train += ids[:k]
            val   += ids[k:]
        out.append({"train": train, "val": val, "test": []})

    (root / "splits_final.json").write_text(json.dumps(out, indent=2))
    print(f"Wrote {folds} folds @ train_ratio={train_ratio}")

if __name__ == "__main__":
    root = sys.argv[1]
    train_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    folds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    main(root, train_ratio, folds)
