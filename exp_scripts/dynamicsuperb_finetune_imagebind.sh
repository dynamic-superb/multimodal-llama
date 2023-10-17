#!/usr/bin/bash

LLAMA_PATH="$1"
PRETRAINED_PATH="$2"
DATA_PATH="$3"
ENCODER_TYPE="$4"
OUTPUT_DIR="$5"

mkdir -p "$OUTPUT_DIR"

export LOCAL_RANK=0

torchrun --master_port=1112 --nproc_per_node=1 dynamicsuperb_finetune.py\
        --batch_size 8 \
        --epochs 4 \
        --warmup_epochs 1 \
        --accum_iter 1 \
        --blr 10e-4 \
        --weight_decay 0.02 \
        --llama_type "7B" \
        --llama_path "$LLAMA_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --pretrained_path "$PRETRAINED_PATH" \
        --data_path "$DATA_PATH" \
        --encoder_type "$ENCODER_TYPE" \
        &>> "$OUTPUT_DIR"/output.log

# python dynamicsuperb_inference.py --exp_path "$OUTPUT_DIR" \
#         --model_path "checkpoint-latest.pth" \
#         --output_dir results \
#         --encoder_type imagebind \
#         --dataset_list test_dataset.txt \
#         --data_path /home/u2619111/hank/Dataset/big-superb-train-data-renamed \
#         --llama_path /home/u2619111/hank/lab/big-superb/LLaMA-Adapter/imagebind_LLM/ckpts/llama 