# Documentation

This directory contains all documentation for the classification_downstream project.

## 📚 Quick Links

### Datasets
- [Dataset Structure](datasets/DATASET_STRUCTURE.md) - Directory organization and file formats
- [Image Size Analysis](datasets/IMAGE_SIZE_ANALYSIS.md) - Native resolutions across datasets
- [Preprocessing Guide](datasets/PREPROCESSING_GUIDE.md) - Data preparation steps
- [Augmentation Strategy](datasets/AUGMENTATION_STRATEGY.md) - Transform pipelines and rationale

### Augmentation
- [Augmentation Policies](augmentation/policies.md) - Dataset-specific transform configurations

### Testing
- [Testing Quick Start](testing/TESTING_QUICK_START.md) - Get started testing datasets
- [Testing Summary](testing/TESTING_SUMMARY.md) - Test results and validation
- [Testing Guide](testing/TESTING.md) - Comprehensive testing documentation

## 📊 Available Datasets

The project supports 10 classification datasets:

1. **AID** - Aerial Image Dataset (600x600)
2. **ZooScanNet** - Plankton classification (24-4911px variable)
3. **ChestXray14** - Chest X-ray pathology (224x224)
4. **NEUDET** - Steel surface defects (200x200)
5. **RxRx1** - Cell microscopy (256x256)
6. **Flowers102** - Flower species (variable sizes)
7. **RESISC45** - Remote sensing scenes (256x256)
8. **PCam** - Histopathology patches (96x96)
9. **DiabeticRetinopathy** - Retinal images (variable sizes)
10. **FGVCAircraft** - Aircraft variants (variable sizes)

## 🚀 Getting Started

1. Set your data root:
   ```bash
   export DATA_ROOT=/path/to/SynergyUnitDatasets
   ```

2. Test a dataset:
   ```bash
   python -m datasets.chestxray14
   ```

3. Run full test suite:
   ```bash
   pytest tests/ -v
   ```

See [Testing Quick Start](testing/TESTING_QUICK_START.md) for more details.
