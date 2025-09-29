# build_labels_aid.py
import json, sys, re
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def main(root):
    root = Path(root)
    img_dir = root / "images"
    assert img_dir.is_dir(), f"Missing {img_dir}"

    # Deterministic class order
    class_names = sorted([d.name for d in img_dir.iterdir() if d.is_dir()])
    cls2idx = {c: i for i, c in enumerate(class_names)}

    labels = {}
    for c in class_names:
        for p in (img_dir / c).rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                stem = re.sub(r"\s+", "_", p.stem)  # sanitize spaces
                labels[f"{c}/{stem}"] = cls2idx[c]

    (root / "class_map.json").write_text(json.dumps(cls2idx, indent=2))
    (root / "labels.json").write_text(json.dumps(labels, indent=2))
    print(f"Wrote {len(class_names)} classes, {len(labels)} labeled samples.")

if __name__ == "__main__":
    main(sys.argv[1])
