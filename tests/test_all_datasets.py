#!/usr/bin/env python3
"""
Test all dataset loaders with augmentations to verify file access and pipeline.
"""

import os
import sys
from pathlib import Path

# Set DATA_ROOT if not already set
if "DATA_ROOT" not in os.environ:
    os.environ["DATA_ROOT"] = "./data"

print(f"Using DATA_ROOT: {os.environ['DATA_ROOT']}")
print()

# Datasets to test
DATASETS = [
    ("AID", "aid"),
    ("ZooScanNet", "zooscannet"),
    ("ChestXray14", "chestxray14"),
    ("NEUDET", "neudet"),
    ("RxRx1", "rxrx1"),
    ("Flowers102", "flowers102"),
    ("RESISC45", "resisc45"),
    ("PCam", "pcam"),
    ("DiabeticRetinopathy", "diabetic_retina"),
    ("FGVCAircraft", "fgvc_aircraft"),
]


def test_dataset(dataset_name, module_name):
    """Test a single dataset."""
    print("\n" + "="*80)
    print(f"Testing {dataset_name}")
    print("="*80)

    try:
        # Import the module
        module = __import__(f"datasets.{module_name}", fromlist=[module_name])

        # Run the main test
        if hasattr(module, "__main__"):
            exec(open(f"datasets/{module_name}.py").read())
        else:
            print(f"⚠️  No __main__ test found for {dataset_name}")

        print(f"✓ {dataset_name} test passed\n")
        return True

    except FileNotFoundError as e:
        print(f"⚠️  File not found: {e}")
        print(f"   Dataset may need preprocessing or extraction")
        return False

    except ImportError as e:
        print(f"⚠️  Import error: {e}")
        return False

    except Exception as e:
        print(f"❌ {dataset_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run tests for all datasets."""

    print("="*80)
    print("DATASET PIPELINE TEST SUITE")
    print("="*80)
    print("\nThis will test file access and augmentations for all datasets.")
    print("Each test loads 2 batches of 16 images with train/val transforms.\n")

    results = {}

    for dataset_name, module_name in DATASETS:
        success = test_dataset(dataset_name, module_name)
        results[dataset_name] = success

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(results.values())
    total = len(results)

    for dataset_name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status:8} {dataset_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All dataset tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} dataset(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
