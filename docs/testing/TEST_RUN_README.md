# Test Run Scripts for Model/Dataset/PEFT Combinations

## Overview

Test all combinations of:
- **3 Models**: supervised, mae_timm, dinov3
- **10 Datasets**: aid, zooscannet, chestxray14, neudet, rxrx1, flowers102, resisc45, pcam, diabetic_retina, fgvc_aircraft
- **7 PEFT Methods**: adapt_former, full_finetuning, gps, linear_probing, lora, vera, visual_prompt_tuning

**Total**: 3 × 10 × 7 = **210 experiments**

## Test Configuration

- `max_epochs=1` (fast iteration)
- `data_fraction=0.01` (1% of data)
- Purpose: Verify all combinations run without errors

## Scripts

### 1. Dry Run (Recommended First Step)
```bash
./test_scripts/test_dryrun.sh
```
Shows all 210 commands without executing. Review to ensure commands look correct.

### 2. Single Test (Recommended Second Step)
```bash
./test_scripts/test_single.sh
```
Runs ONE experiment (supervised + aid + lora) to verify:
- Python environment works
- All dependencies are installed
- Config files are correct
- Data paths are set up

**⚠️ Run this FIRST before the full test suite!**

### 3. Full Test Suite
```bash
./test_scripts/test_all_combinations.sh
```
Runs all 210 experiments sequentially. This will:
- Create a timestamped log directory
- Run each experiment with 1 epoch, 0.01 data fraction
- Log each experiment to separate file
- Generate summary of successes/failures

**Estimated time**: ~30-60 seconds per experiment = 1.75-3.5 hours total (depending on hardware)

## Output Structure

```
test_runs_YYYYMMDD_HHMMSS/
├── summary.txt                          # Overall results
├── supervised_aid_lora.log             # Individual experiment logs
├── supervised_aid_vera.log
├── mae_timm_flowers102_gps.log
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
python main.py \
    model=supervised \
    data=aid \
    peft=lora \
    trainer.max_epochs=1 \
    data.module.data_fraction=0.01
```
