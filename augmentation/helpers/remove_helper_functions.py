#!/usr/bin/env python3
"""
Remove the broken helper functions from all augmentation policy files.
"""

from pathlib import Path

POLICIES_DIR = Path(__file__).parent.parent / "policies"

POLICY_FILES = [
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


def remove_helper_functions(file_path):
    """Remove helper functions from a policy file."""

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find where helper functions start
    helper_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("# Helper functions for consistent interface"):
            helper_start = i
            break

    if helper_start is None:
        return False

    # Remove everything from helper_start onwards
    new_content = ''.join(lines[:helper_start])

    with open(file_path, 'w') as f:
        f.write(new_content)

    return True


def main():
    print("Removing broken helper functions from augmentation policies...\n")

    for filename in POLICY_FILES:
        file_path = POLICIES_DIR / filename

        if not file_path.exists():
            print(f"  ⚠️  {filename}: Not found")
            continue

        print(f"Processing {filename}...")
        removed = remove_helper_functions(file_path)

        if removed:
            print(f"  ✓ Removed helper functions\n")
        else:
            print(f"  - No helper functions found\n")

    print("="*80)
    print("✓ Cleanup complete!")
    print("="*80)


if __name__ == "__main__":
    main()
