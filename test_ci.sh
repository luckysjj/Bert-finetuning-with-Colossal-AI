#!/bin/bash
set -x

BATCH_SIZE=64

for plugin in "torch_ddp" "torch_ddp_fp16" "gemini" "low_level_zero"; do
    echo "Running with plugin: $plugin and batch size: $BATCH_SIZE"
    torchrun --standalone --nproc_per_node=1 finetune.py --target_f1 0.86 --plugin "$plugin" --batch_size "$BATCH_SIZE"
    if [ $? -eq 0 ]; then
        echo "Success: Plugin: $plugin with Batch Size: $BATCH_SIZE"
    else
        echo "Failed: Plugin: $plugin with Batch Size: $BATCH_SIZE"
    fi
done
