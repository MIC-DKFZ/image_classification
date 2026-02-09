#!/usr/bin/env python3
"""
Fix __main__ blocks in dataset files to use actual transform classes.
"""

from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent

# Mapping of dataset files to their actual transform classes
DATASET_TRANSFORMS = {
    "aid.py": {
        "policy": "aid",
        "train_class": "FlipRotateTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "zooscannet.py": {
        "policy": "zooscannet",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "chestxray14.py": {
        "policy": "chestxray14",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "neudet.py": {
        "policy": "neudet",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "rxrx1.py": {
        "policy": "rxrx1",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "flowers102.py": {
        "policy": "flowers102",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "resisc45.py": {
        "policy": "resisc45",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "pcam.py": {
        "policy": "pcam",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "diabetic_retina.py": {
        "policy": "diabetic_retina",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
    "fgvc_aircraft.py": {
        "policy": "fgvc_aircraft",
        "train_class": "TrainTransform",
        "test_class": "TestTransform",
    },
}


def create_new_main_block(dataset_class, dataset_name, policy, train_class, test_class):
    """Generate new __main__ block using actual transform classes."""

    return f'''if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from augmentation.policies.{policy} import {train_class}, {test_class}

    print("="*80)
    print("Testing {dataset_name} Dataset")
    print("="*80)

    # Get augmentation transforms (instantiate the classes)
    train_aug = {train_class}()()
    val_aug = {test_class}()()

    # Test train set with augmentations
    print("\\n[Train Set with Augmentations]")
    train_ds = {dataset_class}(root="$DATA_ROOT/{dataset_name}", split="train", transform=train_aug)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    print(f"Total train samples: {{len(train_ds)}}")
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        if batch_idx >= 2:  # Test 2 batches
            break
        print(f"\\nBatch {{batch_idx + 1}}:")
        print(f"  Images: {{imgs.shape}}, dtype={{imgs.dtype}}, min={{imgs.min().item():.3f}}, max={{imgs.max().item():.3f}}")
        print(f"  Labels: {{labels.shape}}, dtype={{labels.dtype}}")
        print(f"  Unique labels: {{torch.unique(labels).tolist()}}")

    # Test val set with augmentations
    print("\\n[Val Set with Augmentations]")
    val_ds = {dataset_class}(root="$DATA_ROOT/{dataset_name}", split="val", transform=val_aug)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f"Total val samples: {{len(val_ds)}}")
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        if batch_idx >= 2:  # Test 2 batches
            break
        print(f"\\nBatch {{batch_idx + 1}}:")
        print(f"  Images: {{imgs.shape}}, dtype={{imgs.dtype}}, min={{imgs.min().item():.3f}}, max={{imgs.max().item():.3f}}")
        print(f"  Labels: {{labels.shape}}, dtype={{labels.dtype}}")
        print(f"  Unique labels: {{torch.unique(labels).tolist()}}")

    print("\\n" + "="*80)
    print("✓ {dataset_name} Dataset test completed successfully!")
    print("="*80)
'''


def fix_main_block(file_path, dataset_class, dataset_name, config):
    """Fix the __main__ block in a dataset file."""

    with open(file_path, 'r') as f:
        content = f.read()

    # Find where __main__ block starts
    main_start = content.find("if __name__ == '__main__':")
    if main_start == -1:
        return False

    # Generate new main block
    new_main = create_new_main_block(
        dataset_class,
        dataset_name,
        config["policy"],
        config["train_class"],
        config["test_class"]
    )

    # Replace from main_start to end
    new_content = content[:main_start] + new_main

    with open(file_path, 'w') as f:
        f.write(new_content)

    return True


# Map dataset files to their classes and names
DATASET_INFO = {
    "aid.py": ("AIDData", "AID"),
    "zooscannet.py": ("ZooScanNetData", "ZooScanNet"),
    "chestxray14.py": ("ChestXray14Data", "ChestXray14"),
    "neudet.py": ("NEUDETData", "NEUDET"),
    "rxrx1.py": ("RxRx1Data", "RxRx1"),
    "flowers102.py": ("Flowers102Data", "Flowers102"),
    "resisc45.py": ("RESISC45Data", "RESISC45"),
    "pcam.py": ("PCamData", "PCam"),
    "diabetic_retina.py": ("EyePACSData", "DiabeticRetinopathy"),
    "fgvc_aircraft.py": ("FGVCAircraftData", "FGVCAircraft"),
}


def main():
    print("Fixing __main__ blocks in dataset files...\n")

    for filename, (dataset_class, dataset_name) in DATASET_INFO.items():
        file_path = DATASETS_DIR / filename

        if not file_path.exists():
            print(f"  ⚠️  {filename}: Not found")
            continue

        if filename not in DATASET_TRANSFORMS:
            print(f"  ⚠️  {filename}: No transform config")
            continue

        print(f"Processing {filename}...")
        fixed = fix_main_block(
            file_path,
            dataset_class,
            dataset_name,
            DATASET_TRANSFORMS[filename]
        )

        if fixed:
            print(f"  ✓ Fixed __main__ block\n")
        else:
            print(f"  - No __main__ block found\n")

    print("="*80)
    print("✓ All dataset files fixed!")
    print("="*80)


if __name__ == "__main__":
    main()
