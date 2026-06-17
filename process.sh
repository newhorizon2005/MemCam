#!/bin/bash
export CUDA_VISIBLE_DEVICES=1,2
python train_memcam.py \
    --task process \
    --text_encoder_path models/Wan-AI/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth \
    --vae_path models/Wan-AI/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth 