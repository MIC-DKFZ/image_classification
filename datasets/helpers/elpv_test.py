from PIL import Image
from pathlib import Path
from collections import Counter

root = Path("/home/d246a/Documents/data/elpv/data/images")
counts = Counter()

for p in root.rglob("*.png"):
    with Image.open(p) as im:
        counts[im.mode] += 1

print(counts)