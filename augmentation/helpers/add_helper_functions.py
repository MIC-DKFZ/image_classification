#!/usr/bin/env python3
"""
Add get_train_transforms() and get_val_transforms() helper functions
to all augmentation policy files for consistent interface.
"""

from pathlib import Path

POLICIES_DIR = Path(__file__).parent.parent / "policies"

# Mapping of policy files to their train and test transform classes
POLICY_CONFIGS = {
    "aid.py": {
        "train_class": "FlipRotateTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "zooscannet.py": {
        "train_class": "FlipRotateTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "chestxray14.py": {
        "train_class": "FlipTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "neudet.py": {
        "train_class": "FlipColorJitterTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "rxrx1.py": {
        "train_class": "FlipRotateTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "flowers102.py": {
        "train_class": "RandomResizedCropTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "resisc45.py": {
        "train_class": "FlipRotateTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "pcam.py": {
        "train_class": "FlipRotateTransformImgNetNorm96",
        "test_class": "TestTransformImgNetNorm96",
    },
    "diabetic_retina.py": {
        "train_class": "FlipRotateTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
    "fgvc_aircraft.py": {
        "train_class": "RandomResizedCropTransformImgNetNorm",
        "test_class": "TestTransformImgNetNorm",
    },
}


def create_helper_functions(train_class, test_class):
    """Generate helper function code."""
    return f'''

# Helper functions for consistent interface
def get_train_transforms():
    """Get training transforms with augmentations."""
    return {train_class}()()


def get_val_transforms():
    """Get validation/test transforms (minimal augmentations)."""
    return {test_class}()()
'''


def add_helpers_to_file(file_path, train_class, test_class):
    """Add helper functions to a policy file if they don't exist."""

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if helpers already exist
    if "def get_train_transforms" in content:
        print(f"  ✓ {file_path.name}: Helpers already exist")
        return False

    # Add helpers at the end
    helpers = create_helper_functions(train_class, test_class)
    new_content = content + helpers

    with open(file_path, 'w') as f:
        f.write(new_content)

    return True


def main():
    print("Adding helper functions to augmentation policies...\n")

    for filename, config in POLICY_CONFIGS.items():
        file_path = POLICIES_DIR / filename

        if not file_path.exists():
            print(f"  ⚠️  {filename}: Not found")
            continue

        print(f"Processing {filename}...")
        added = add_helpers_to_file(
            file_path,
            config["train_class"],
            config["test_class"]
        )

        if added:
            print(f"  ✓ Added helpers\n")
        else:
            print()

    print("="*80)
    print("✓ All policy files updated!")
    print("="*80)


if __name__ == "__main__":
    main()
