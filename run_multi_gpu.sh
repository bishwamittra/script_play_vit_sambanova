#!/bin/bash

python huggingface_gpu.py \
    --seed=42 \
    --epoch=50 \
    --batch_size=512 \
    --lr=1e-4