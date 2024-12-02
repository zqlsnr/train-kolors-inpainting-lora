#!/bin/bash

# Lora dreambooth training with inpainting tuned SD/SDXL model

# source ./venv/bin/activate

# pwd

accelerate launch train_kolors_inpainting_lora.py \
    --instance_data_dir="/nas/datasets/train-mae/openimage/images" \
    --output_dir="./inpainting-lora-openimages-2000" \
    --instance_prompt_dir="/nas/datasets/train-mae/openimage/texts" \
    --mixed_precision="fp16" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=5e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=200 \
    --seed="42"