#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
python train_memcam.py \
    --task train \
    --dataset_path data \
    --dit_path models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \
    --learning_rate 1e-5 --max_epochs 5 --batch_size 1 \
    --use_tensorboard \
    --use_gradient_checkpointing \
    --gradient_checkpointing_ratio 0.3