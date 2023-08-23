#!/bin/bash

python huggingface_gpu.py \
    --seed=42 \
    --epoch=50 \
    --batch_size=64 \
    --lr=2e-4