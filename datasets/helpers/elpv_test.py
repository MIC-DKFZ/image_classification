import os
from PIL import Image
from pathlib import Path
from collections import Counter

data_root = os.environ.get("DATA_ROOT", "./data")
root = Path(f"{data_root}/elpv/data/images")
counts = Counter()

for p in root.rglob("*.png"):
    with Image.open(p) as im:
        counts[im.mode] += 1

print(counts)