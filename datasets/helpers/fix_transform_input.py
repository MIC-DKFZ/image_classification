#!/usr/bin/env python3
"""
Fix __getitem__ methods to pass PIL Images to transforms, not tensors.
Transforms (especially ToTensor) expect PIL Images or numpy arrays.
"""

from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent

DATASET_FILES = [
    "aid.py",
    "zooscannet.py",
    "chestxray14.py",
    "neudet.py",
    "rxrx1.py",
    "flowers102.py",
    "resisc45.py",
    "pcam.py",
    "diabetic_retina.py",
    "fgvc_aircraft.py",
]


def fix_getitem(file_path):
    """Fix __getitem__ to apply transforms to PIL Image, not tensor."""

    with open(file_path, 'r') as f:
        content = f.read()

    # The problematic pattern (converting to tensor before transform)
    old_pattern = '''        img = Image.open(img_path).convert("RGB")
        img_np = np.ascontiguousarray(np.array(img), dtype=np.uint8)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        if self.transform:
            img_t = self.transform(**{"image": img_t})["image"]

        img_t = img_t.contiguous().clone()
        return img_t, y'''

    # The correct pattern (apply transform to PIL Image)
    new_pattern = '''        img = Image.open(img_path).convert("RGB")

        if self.transform:
            # Transform expects PIL Image or numpy array
            img = self.transform(img)
        else:
            # No transform - convert to tensor manually
            img_np = np.ascontiguousarray(np.array(img), dtype=np.uint8)
            img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float()

        return img, y'''

    if old_pattern in content:
        content = content.replace(old_pattern, new_pattern)

        with open(file_path, 'w') as f:
            f.write(content)

        return True

    return False


def main():
    print("Fixing transform input in dataset __getitem__ methods...\n")

    for filename in DATASET_FILES:
        file_path = DATASETS_DIR / filename

        if not file_path.exists():
            print(f"  ⚠️  {filename}: Not found")
            continue

        print(f"Processing {filename}...")
        fixed = fix_getitem(file_path)

        if fixed:
            print(f"  ✓ Fixed __getitem__ method\n")
        else:
            print(f"  - Pattern not found or already fixed\n")

    print("="*80)
    print("✓ All dataset files fixed!")
    print("="*80)
    print("\nKey changes:")
    print("  - Transforms now receive PIL Image (not tensor)")
    print("  - ToTensor() in transforms will handle conversion")
    print("  - If no transform, manually convert to tensor")


if __name__ == "__main__":
    main()
