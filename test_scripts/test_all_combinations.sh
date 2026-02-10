#!/bin/bash

# Test script for all model/dataset/peft combinations
# 3 models × 10 datasets × 7 adaptations = 210 experiments

# Configuration
MAX_EPOCHS=1
DATA_FRACTION=0.01

# Model configs
MODELS=(
    "supervised"
    "mae_timm"
    "dinov3"
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
echo "Configuration: max_epochs=$MAX_EPOCHS, data_fraction=$DATA_FRACTION" >> "$SUMMARY_FILE"
echo "Total experiments: $((${#MODELS[@]} * ${#DATASETS[@]} * ${#PEFTS[@]}))" >> "$SUMMARY_FILE"
echo "----------------------------------------" >> "$SUMMARY_FILE"

# Counters
TOTAL=0
SUCCESS=0
FAILED=0

# Run all combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for peft in "${PEFTS[@]}"; do
            TOTAL=$((TOTAL + 1))
            
            # Create experiment name
            EXP_NAME="${model}_${dataset}_${peft}"
            LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
            
            echo "[$TOTAL/210] Running: $EXP_NAME"
            echo "----------------------------------------" | tee -a "$SUMMARY_FILE"
            echo "[$TOTAL/210] $EXP_NAME" | tee -a "$SUMMARY_FILE"
            
            # Run experiment
            python main.py \
                model="$model" \
                data="$dataset" \
                peft="$peft" \
                trainer.max_epochs="$MAX_EPOCHS" \
                data.module.data_fraction="$DATA_FRACTION" \
                trainer.fast_dev_run=false \
                > "$LOG_FILE" 2>&1
            
            # Check result
            if [ $? -eq 0 ]; then
                SUCCESS=$((SUCCESS + 1))
                echo "  ✓ SUCCESS" | tee -a "$SUMMARY_FILE"
            else
                FAILED=$((FAILED + 1))
                echo "  ✗ FAILED - Check $LOG_FILE" | tee -a "$SUMMARY_FILE"
            fi
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
