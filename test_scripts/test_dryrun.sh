#!/bin/bash

# Dry run - just print commands without executing

MODELS=("supervised" "mae_timm" "dinov3")
DATASETS=("aid" "zooscannet" "chestxray14" "neudet" "rxrx1" "flowers102" "resisc45" "pcam" "diabetic_retina" "fgvc_aircraft")
PEFTS=("adapt_former" "full_finetuning" "gps" "linear_probing" "lora" "vera" "visual_prompt_tuning")
DATA_FRACTIONS=(0.1 1.0)

COUNT=0
TOTAL=$((${#MODELS[@]} * ${#DATASETS[@]} * ${#PEFTS[@]} * ${#DATA_FRACTIONS[@]}))

for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for peft in "${PEFTS[@]}"; do
            for data_frac in "${DATA_FRACTIONS[@]}"; do
                COUNT=$((COUNT + 1))
                echo "[$COUNT/$TOTAL] python main.py model=$model data=$dataset peft=$peft trainer.max_epochs=5 data.module.data_fraction=$data_frac"
            done
        done
    done
done

echo ""
echo "Total commands: $COUNT"
