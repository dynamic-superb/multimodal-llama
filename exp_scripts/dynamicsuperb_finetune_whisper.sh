#!/usr/bin/bash

OUTPUT_DIR="exp/whisper_newdata/whisper1"
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
        --llama_path "/home/u2619111/hank/lab/big-superb/LLaMA-Adapter/imagebind_LLM/ckpts/llama" \
        --output_dir "$OUTPUT_DIR" \
        --pretrained_path "/home/u2619111/hank/lab/big-superb/LLaMA-Adapter/imagebind_LLM/ckpts/7B.pth" \
        --data_config "/home/u2619111/hank/lab/big-superb/LLaMA-Adapter/imagebind_LLM/exps/config.yaml" \
        --data_path "/home/u2619111/hank/Dataset/big-superb-train-data-renamed" \
        --encoder_type "whisper" \
        &>> "$OUTPUT_DIR"/output.log

python bigsuperb_inference.py --exp_path "$OUTPUT_DIR" \
        --model_path "checkpoint-latest.pth" \
        --output_dir results \
        --encoder_type whisper \
        --dataset_list test_dataset.txt \
        --data_path /home/u2619111/hank/Dataset/big-superb-train-data-renamed \
        --llama_path /home/u2619111/hank/lab/big-superb/LLaMA-Adapter/imagebind_LLM/ckpts/llama 