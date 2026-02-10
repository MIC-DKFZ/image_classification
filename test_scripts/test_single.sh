#!/bin/bash

# Test a single experiment first to verify setup

echo "Testing single experiment: supervised + aid + lora"
echo "Configuration: max_epochs=5, data_fraction=0.1"
echo "========================================"

HYDRA_FULL_ERROR=1 python main.py \
    model=supervised \
    data=aid \
    peft=lora \
    trainer.max_epochs=5 \
    data.module.data_fraction=0.1

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Single test passed! Ready to run full test suite."
    echo "  Run: ./test_all_combinations.sh"
else
    echo ""
    echo "✗ Single test failed. Check configuration before running full suite."
fi
