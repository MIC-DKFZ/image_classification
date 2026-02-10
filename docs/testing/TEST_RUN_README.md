# Test Run Scripts for Model/Dataset/PEFT Combinations

## Overview

Test all combinations of:
- **3 Models**: supervised, mae_timm, dinov3
- **10 Datasets**: aid, zooscannet, chestxray14, neudet, rxrx1, flowers102, resisc45, pcam, diabetic_retina, fgvc_aircraft
- **7 PEFT Methods**: adapt_former, full_finetuning, gps, linear_probing, lora, vera, visual_prompt_tuning
- **2 Data Fractions**: 0.1 (10% of data), 1.0 (full data)

**Total**: 3 × 10 × 7 × 2 = **420 experiments**

## Test Configuration

- `max_epochs=5` (reliable timing measurement)
- `data_fraction=0.1, 1.0` (10% and 100% of data)
- Purpose: Verify all combinations run without errors and measure training time

## Scripts

### 1. Dry Run (Recommended First Step)
```bash
./test_scripts/test_dryrun.sh
```
Shows all 420 commands without executing. Review to ensure commands look correct.

### 2. Single Test (Recommended Second Step)
```bash
./test_scripts/test_single.sh
```
Runs ONE experiment (supervised + aid + lora, 5 epochs, 10% data) to verify:
- Python environment works
- All dependencies are installed
- Config files are correct
- Data paths are set up

**⚠️ Run this FIRST before the full test suite!**

### 3. Full Test Suite
```bash
./test_scripts/test_all_combinations.sh
```
Runs all 420 experiments sequentially. This will:
- Create a timestamped log directory
- Run each experiment with 5 epochs and both 0.1 and 1.0 data fractions
- Log each experiment to separate file with timing information
- Generate summary of successes/failures with durations

**Estimated time**:
- 0.1 data fraction: ~1-3 min per experiment = 7-21 hours for 210 experiments
- 1.0 data fraction: ~5-30 min per experiment = 17.5-105 hours for 210 experiments
- **Total: ~1-5 days** (depending on hardware and dataset sizes)

## Output Structure

```
test_runs_YYYYMMDD_HHMMSS/
├── summary.txt                                   # Overall results with timing
├── supervised_aid_lora_frac0_1.log             # 10% data experiments
├── supervised_aid_lora_frac1_0.log             # 100% data experiments
├── supervised_aid_vera_frac0_1.log
├── mae_timm_flowers102_gps_frac1_0.log
└── ...
```

## Monitoring Progress

While running, you can monitor in another terminal:
```bash
# Watch summary file
tail -f test_runs_*/summary.txt

# Count completed experiments
ls test_runs_*/*.log | wc -l

# Check for failures
grep "FAILED" test_runs_*/summary.txt
```

## After Testing

Once all tests pass, you can run full experiments by:
1. Removing `data_fraction` parameter
2. Setting `trainer.max_epochs=100` (or desired value)
3. Using a job scheduler for cluster runs

## Troubleshooting

If experiments fail:
1. Check individual log files in the test_runs directory
2. Look for common errors (missing configs, data path issues, CUDA errors)
3. Fix issues and re-run test_all_combinations.sh
4. The script will overwrite previous logs

## Example: Run Specific Combination Manually

```bash
# Run with 10% data for 5 epochs
python main.py \
    model=supervised \
    data=aid \
    peft=lora \
    trainer.max_epochs=5 \
    data.module.data_fraction=0.1

# Run with full data for 5 epochs
python main.py \
    model=supervised \
    data=aid \
    peft=lora \
    trainer.max_epochs=5 \
    data.module.data_fraction=1.0
```
