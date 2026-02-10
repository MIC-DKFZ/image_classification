#!/usr/bin/env python3
"""
Fix __main__ blocks to properly use DATA_ROOT environment variable.
Replace "$DATA_ROOT" literal strings with os.environ.get().
"""

import re
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


def fix_data_root(file_path):
    """Fix DATA_ROOT environment variable usage in __main__ block."""

    with open(file_path, 'r') as f:
        content = f.read()

    # Check if already fixed
    if 'os.environ.get("DATA_ROOT"' in content:
        return False

    # Find the __main__ block
    if "if __name__ == '__main__':" not in content:
        return False

    # Add os import if not present in __main__ block
    # Replace the import section
    old_import = "if __name__ == '__main__':\n    from torch.utils.data import DataLoader"
    new_import = "if __name__ == '__main__':\n    import os\n    from torch.utils.data import DataLoader"

    content = content.replace(old_import, new_import)

    # Add DATA_ROOT line after imports
    # Find the print("="*80) line and insert before it
    dataset_name = None
    for line in content.split('\n'):
        if 'print("Testing' in line and 'Dataset")' in line:
            # Extract dataset name from print statement
            match = re.search(r'Testing (\w+) Dataset', line)
            if match:
                dataset_name = match.group(1)
                break

    if not dataset_name:
        return False

    # Insert DATA_ROOT setup after imports, before first print
    old_section = f'''    print("="*80)
    print("Testing {dataset_name} Dataset")
    print("="*80)'''

    new_section = f'''    # Get DATA_ROOT from environment or use default
    data_root = os.environ.get("DATA_ROOT", "./data")

    print("="*80)
    print("Testing {dataset_name} Dataset")
    print(f"Using DATA_ROOT: {{data_root}}")
    print("="*80)'''

    content = content.replace(old_section, new_section)

    # Replace all "$DATA_ROOT/DatasetName" with f"{data_root}/DatasetName"
    content = re.sub(
        r'root="\$DATA_ROOT/(\w+)"',
        r'root=f"{data_root}/\1"',
        content
    )

    with open(file_path, 'w') as f:
        f.write(content)

    return True


def main():
    print("Fixing DATA_ROOT environment variable usage...\n")

    for filename in DATASET_FILES:
        file_path = DATASETS_DIR / filename

        if not file_path.exists():
            print(f"  ⚠️  {filename}: Not found")
            continue

        print(f"Processing {filename}...")
        fixed = fix_data_root(file_path)

        if fixed:
            print(f"  ✓ Fixed DATA_ROOT usage\n")
        else:
            print(f"  - Already fixed or no __main__ block\n")

    print("="*80)
    print("✓ All dataset files fixed!")
    print("="*80)
    print("\nNow you can run:")
    print("  export DATA_ROOT=./data")
    print("  python -m datasets.chestxray14")


if __name__ == "__main__":
    main()
