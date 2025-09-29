import json
from pathlib import Path
import argparse
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from PIL import Image
import numpy as np
import os
import os.path as osp


def load_dataset(fname=None):
    if fname is None:
        fname = os.path.join(os.path.dirname(__file__), 'data', 'labels.csv')

    # NOTE: dtypes match your current code; adjust widths if your CSV has longer values.
    data = np.genfromtxt(
        fname, dtype=['|S19', '<f8', '|S4'], names=['path', 'probability', 'type']
    )
    image_fnames = np.char.decode(data['path'])
    probs = data['probability']
    types = np.char.decode(data['type'])

    def load_cell_image(fname):
        with Image.open(fname) as image:
            return np.asarray(image)

    dir = os.path.dirname(fname)
    images = np.array([load_cell_image(os.path.join(dir, fn)) for fn in image_fnames])

    return images, probs, types


def build_splits_and_labels(
    test_ratio=0.2,
    folds=5,
    seed=0,
    csv_path=None,
    save_id2path=False,
    out_path="splits_final.json",
    labels_types_path="labels_types.json",
    labels_probs_path="labels_probs.json",
    class_map_path="class_map_types.json",
):
    # Load dataset once
    images, probs, types = load_dataset(csv_path)  # (N, H, W), (N,), (N,)
    N = images.shape[0]
    idx_all = np.arange(N, dtype=int)

    # --- Write label JSONs (IDs are indices matching splits) ---
    # types as strings, probs as floats
    labels_types = {str(i): str(types[i]) for i in range(N)}
    labels_probs = {str(i): float(probs[i]) for i in range(N)}
    Path(labels_types_path).write_text(json.dumps(labels_types, indent=2))
    Path(labels_probs_path).write_text(json.dumps(labels_probs, indent=2))

    # Optional class map (string class -> integer index for reference)
    classes = sorted(set(map(str, types.tolist())))
    class_map = {c: i for i, c in enumerate(classes)}
    Path(class_map_path).write_text(json.dumps(class_map, indent=2))

    print(f"Wrote {labels_types_path}, {labels_probs_path}, and {class_map_path}")

    # --- Build splits (stratify by type) ---
    y = np.array(types, dtype=str)

    # 1) Fixed test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss.split(idx_all, y))
    test_idx = test_idx.astype(int)

    # 2) K-fold CV on remaining
    y_trainval = y[trainval_idx]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    out = []
    for tr_sub, val_sub in skf.split(trainval_idx, y_trainval):
        train_ids = trainval_idx[tr_sub].astype(int).tolist()
        val_ids   = trainval_idx[val_sub].astype(int).tolist()
        out.append({"train": train_ids, "val": val_ids, "test": test_idx.tolist()})

    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"Wrote {out_path} with {folds} folds (test_ratio={test_ratio}).")

    # Optional: also save index -> relative path (from CSV)
    if save_id2path:
        if csv_path is None:
            csv_path = osp.join(osp.dirname(__file__), "data", "labels.csv")
        data = np.genfromtxt(
            csv_path, dtype=['|S19', '<f8', '|S4'], names=['path', 'probability', 'type']
        )
        rel_paths = np.char.decode(data['path']).tolist()
        id2path = {int(i): rel_paths[i] for i in range(len(rel_paths))}
        Path("id2path.json").write_text(json.dumps(id2path, indent=2))
        print("Wrote id2path.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None,
                    help="Path to labels.csv; if omitted, load_dataset default is used.")
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=69)
    ap.add_argument("--save_id2path", action="store_true")
    ap.add_argument("--out", type=str, default="splits_final.json")
    ap.add_argument("--labels_types", type=str, default="labels_types.json")
    ap.add_argument("--labels_probs", type=str, default="labels_probs.json")
    ap.add_argument("--class_map", type=str, default="class_map_types.json")
    args = ap.parse_args()

    build_splits_and_labels(
        test_ratio=args.test_ratio,
        folds=args.folds,
        seed=args.seed,
        csv_path=args.csv,
        save_id2path=args.save_id2path,
        out_path=args.out,
        labels_types_path=args.labels_types,
        labels_probs_path=args.labels_probs,
        class_map_path=args.class_map,
    )
