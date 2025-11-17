#!/bin/bash

# Set number of GPUs to use (change this number as needed)
NUM_GPUS=2

accelerate launch --num_processes ${NUM_GPUS} train_uv_diffusion.py \
  --config config.json \
  --root_dir /app/data/vfhq/ \
  --json_dir json_dir/ \
  --mask_path template/mask.png \
  --template-dir template/ \
  --batch-size 1 \
  --num-workers 0 \
  --mixed-precision bf16 \
  --use-tensorboard \
  --save-checkpoints \
  --compile \
  --save-every 10000 \
  --name uv_diffusion