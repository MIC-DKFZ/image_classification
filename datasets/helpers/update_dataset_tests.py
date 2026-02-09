#!/usr/bin/env python3
"""
Update all dataset .py files to include comprehensive batch testing with augmentations.
"""

from pathlib import Path

DATASETS_DIR = Path(__file__).parent.parent

# Mapping of dataset file to augmentation policy module
DATASET_CONFIGS = {
    "chestxray14.py": {
        "class_name": "ChestXray14Data",
        "dataset_name": "ChestXray14",
        "policy": "chestxray14",
    },
    "neudet.py": {
        "class_name": "NEUDETData",
        "dataset_name": "NEUDET",
        "policy": "neudet",
    },
    "rxrx1.py": {
        "class_name": "RxRx1Data",
        "dataset_name": "RxRx1",
        "policy": "rxrx1",
    },
    "flowers102.py": {
        "class_name": "Flowers102Data",
        "dataset_name": "Flowers102",
        "policy": "flowers102",
    },
    "resisc45.py": {
        "class_name": "RESISC45Data",
        "dataset_name": "RESISC45",
        "policy": "resisc45",
    },
    "pcam.py": {
        "class_name": "PCamData",
        "dataset_name": "PCam",
        "policy": "pcam",
    },
    "diabetic_retina.py": {
        "class_name": "EyePACSData",
        "dataset_name": "DiabeticRetinopathy",
        "policy": "diabetic_retina",
    },
    "fgvc_aircraft.py": {
        "class_name": "FGVCAircraftData",
        "dataset_name": "FGVCAircraft",
        "policy": "fgvc_aircraft",
    },
}


def create_test_code(class_name, dataset_name, policy, show_limited_labels=False):
    """Generate the test code for a dataset."""

    unique_labels_line = (
        f'print(f"  Unique labels: {{torch.unique(labels).tolist()[:10]}}... ({{len(torch.unique(labels))}} unique)")'
        if show_limited_labels
        else 'print(f"  Unique labels: {torch.unique(labels).tolist()}")'
    )

    return f'''if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from augmentation.policies.{policy} import get_train_transforms, get_val_transforms

    print("="*80)
    print("Testing {dataset_name} Dataset")
    print("="*80)

    # Get augmentation transforms
    train_aug = get_train_transforms()
    val_aug = get_val_transforms()

    # Test train set with augmentations
    print("\\n[Train Set with Augmentations]")
    train_ds = {class_name}(root="$DATA_ROOT/{dataset_name}", split="train", transform=train_aug)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)

    print(f"Total train samples: {{len(train_ds)}}")
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        if batch_idx >= 2:  # Test 2 batches
            break
        print(f"\\nBatch {{batch_idx + 1}}:")
        print(f"  Images: {{imgs.shape}}, dtype={{imgs.dtype}}, min={{imgs.min().item():.3f}}, max={{imgs.max().item():.3f}}")
        print(f"  Labels: {{labels.shape}}, dtype={{labels.dtype}}")
        {unique_labels_line}

    # Test val set with augmentations
    print("\\n[Val Set with Augmentations]")
    val_ds = {class_name}(root="$DATA_ROOT/{dataset_name}", split="val", transform=val_aug)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    print(f"Total val samples: {{len(val_ds)}}")
    for batch_idx, (imgs, labels) in enumerate(val_loader):
        if batch_idx >= 2:  # Test 2 batches
            break
        print(f"\\nBatch {{batch_idx + 1}}:")
        print(f"  Images: {{imgs.shape}}, dtype={{imgs.dtype}}, min={{imgs.min().item():.3f}}, max={{imgs.max().item():.3f}}")
        print(f"  Labels: {{labels.shape}}, dtype={{labels.dtype}}")
        {unique_labels_line}

    print("\\n" + "="*80)
    print("✓ {dataset_name} Dataset test completed successfully!")
    print("="*80)
'''


def update_dataset_file(file_path, class_name, dataset_name, policy, show_limited_labels=False):
    """Update a dataset file with new test code."""

    with open(file_path, 'r') as f:
        content = f.read()

    # Find the main block
    main_start = content.find("if __name__ == '__main__':")
    if main_start == -1:
        print(f"  ⚠️  No main block found, skipping")
        return False

    # Generate new test code
    new_test = create_test_code(class_name, dataset_name, policy, show_limited_labels)

    # Replace everything from main block to end
    new_content = content[:main_start] + new_test

    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)

    return True


def main():
    print("Updating dataset test blocks...\n")

    for filename, config in DATASET_CONFIGS.items():
        file_path = DATASETS_DIR / filename
        if not file_path.exists():
            print(f"⚠️  {filename}: Not found")
            continue

        print(f"Updating {filename}...")

        # Datasets with many classes should show limited labels
        show_limited = config["dataset_name"] in ["RxRx1", "Flowers102"]

        success = update_dataset_file(
            file_path,
            config["class_name"],
            config["dataset_name"],
            config["policy"],
            show_limited_labels=show_limited
        )

        if success:
            print(f"  ✓ Updated\n")
        else:
            print()

    print("="*80)
    print("✓ All dataset files updated!")
    print("="*80)


if __name__ == "__main__":
    main()
