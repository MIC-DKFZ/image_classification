#!/bin/bash

# Dry run - just print commands without executing

MODELS=("supervised" "mae_timm" "dinov3")
DATASETS=("aid" "zooscannet" "chestxray14" "neudet" "rxrx1" "flowers102" "resisc45" "pcam" "diabetic_retina" "fgvc_aircraft")
PEFTS=("adapt_former" "full_finetuning" "gps" "linear_probing" "lora" "vera" "visual_prompt_tuning")

COUNT=0
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for peft in "${PEFTS[@]}"; do
            COUNT=$((COUNT + 1))
            echo "[$COUNT/210] python main.py model=$model data=$dataset peft=$peft trainer.max_epochs=1 data.module.data_fraction=0.01"
        done
    done
done

echo ""
echo "Total commands: $COUNT"
