#!/bin/bash

CUDA_VISIBLE_DEVICE=0 python huggingface_gpu.py \
    --seed=42 \
    --epoch=1 \
    --batch_size=64 \
    --lr=2e-4



