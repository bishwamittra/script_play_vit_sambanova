#!/bin/bash

python huggingface_gpu.py \
    --seed=42 \
    --epoch=10 \
    --batch_size=512 \
    --lr=2e-4