#!/bin/bash
set -xe

for plugin in "torch_ddp" "torch_ddp_fp16" "gemini" "low_level_zero"; do
   torchrun --standalone --nproc_per_node 1  benchmark.py --plugin $plugin --model_type "bert"
done
