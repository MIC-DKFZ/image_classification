#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ImageNet normalization (standard for pretrained models)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(img_path: Path, out_path: Path, size: int = 512):
    """
    Load image -> resize -> normalize -> save tensor
    Output tensor: float32, shape (3, size, size)
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((size, size), resample=Image.BILINEAR)

    img_np = np.array(img, dtype=np.float32) / 255.0   # HWC [0,1]
    img_t = torch.from_numpy(img_np).permute(2, 0, 1)  # CHW

    img_t = (img_t - IMAGENET_MEAN) / IMAGENET_STD
    img_t = img_t.contiguous()  # safety

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(img_t, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root")
    ap.add_argument("--in_dir", default="train", help="Input image folder")
    ap.add_argument("--out_dir", default="train_preprocess", help="Output folder")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--exts", nargs="+", default=[".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    args = ap.parse_args()

    root = Path(args.root)
    in_dir = root / args.in_dir
    out_dir = root / args.out_dir

    exts = {e.lower() for e in args.exts}

    img_paths = [
        p for p in in_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ]

    print(f"Found {len(img_paths)} images in {in_dir}")
    print(f"Writing preprocessed tensors to {out_dir}")

    for p in tqdm(img_paths):
        out_path = out_dir / (p.stem + ".pt")
        preprocess_image(p, out_path, size=args.size)

    print("Done.")


if __name__ == "__main__":
    main()


