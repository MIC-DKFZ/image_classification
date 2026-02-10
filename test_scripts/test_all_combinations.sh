#!/bin/bash

# Test script for all model/dataset/peft combinations
# 3 models × 10 datasets × 7 adaptations × 2 data fractions = 420 experiments

# Configuration
MAX_EPOCHS=5
DATA_FRACTIONS=(0.1 1.0)

# Model configs
MODELS=(
    "supervised"
    "mae_timm"
    "dinov3_reference"
)

# Dataset configs
DATASETS=(
    "aid"
    "zooscannet"
    "chestxray14"
    "neudet"
    "rxrx1"
    "flowers102"
    "resisc45"
    "pcam"
    "diabetic_retina"
    "fgvc_aircraft"
)

# PEFT configs
PEFTS=(
    "adapt_former"
    "full_finetuning"
    "gps"
    "linear_probing"
    "lora"
    "vera"
    "visual_prompt_tuning"
)

# Create log directory
LOG_DIR="test_runs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Summary file
SUMMARY_FILE="$LOG_DIR/summary.txt"
echo "Test Run Summary - $(date)" > "$SUMMARY_FILE"
echo "Configuration: max_epochs=$MAX_EPOCHS, data_fractions=${DATA_FRACTIONS[*]}" >> "$SUMMARY_FILE"
echo "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]} * ${#PEFTS[@]} * ${#DATA_FRACTIONS[@]}))" >> "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

# Counters
TOTAL=0
SUCCESS=0
FAILED=0

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#PEFTS[@]} * ${#DATA_FRACTIONS[@]}))

# Run all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for peft in "${PEFTS[@]}"; do
            for data_frac in "${DATA_FRACTIONS[@]}"; do
                TOTAL=$((TOTAL + 1))

                # Create experiment name (replace . with p for file names)
                FRAC_STR=$(echo "$data_frac" | sed 's/\./_/g')
                EXP_NAME="${model}_${dataset}_${peft}_frac${FRAC_STR}"
                LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

                echo "[$TOTAL/$TOTAL_EXPERIMENTS] Running: $EXP_NAME (data_frac=$data_frac)"
                echo "----------------------------------------" | tee -a "$SUMMARY_FILE"
                echo "[$TOTAL/$TOTAL_EXPERIMENTS] $EXP_NAME (data_frac=$data_frac)" | tee -a "$SUMMARY_FILE"

                # Start timer
                START_TIME=$(date +%s)

                # Run experiment
                python main.py \
                    model="$model" \
                    data="$dataset" \
                    peft="$peft" \
                    trainer.max_epochs="$MAX_EPOCHS" \
                    data.module.data_fraction="$data_frac" \
                    > "$LOG_FILE" 2>&1

                # End timer
                END_TIME=$(date +%s)
                DURATION=$((END_TIME - START_TIME))

                # Check result
                if [ $? -eq 0 ]; then
                    SUCCESS=$((SUCCESS + 1))
                    echo "  ✓ SUCCESS (${DURATION}s)" | tee -a "$SUMMARY_FILE"
                else
                    FAILED=$((FAILED + 1))
                    echo "  ✗ FAILED (${DURATION}s) - Check $LOG_FILE" | tee -a "$SUMMARY_FILE"
                fi
            done
        done
    done
done

# Final summary
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Test Run Complete!" | tee -a "$SUMMARY_FILE"
echo "Total: $TOTAL" | tee -a "$SUMMARY_FILE"
echo "Success: $SUCCESS" | tee -a "$SUMMARY_FILE"
echo "Failed: $FAILED" | tee -a "$SUMMARY_FILE"
echo "Success Rate: $(awk "BEGIN {printf \"%.1f%%\", ($SUCCESS/$TOTAL)*100}")" | tee -a "$SUMMARY_FILE"
echo "========================================" | tee -a "$SUMMARY_FILE"
echo "Results saved to: $LOG_DIR"
