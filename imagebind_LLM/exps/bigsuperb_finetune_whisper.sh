#!/usr/bin/bash

# LLAMA_PATH="$1"
# PRETRAINED_PATH="$2" # path to pre-trained checkpoint
OUTPUT_DIR="exp/whisper2"
mkdir -p "$OUTPUT_DIR"

export LOCAL_RANK=0

# python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=2 \      --use_env \
#  bigsuperb_finetune.py --batch_size 4 \
#  --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
#  --llama_path "$LLAMA_PATH" \
#  --output_dir "$OUTPUT_DIR" \
#  --pretrained_path "$PRETRAINED_PATH" \
#  &>> "$OUTPUT_DIR"/output.log &

torchrun --master_port=1112 --nproc_per_node=1 bigsuperb_finetune_whisper.py\
        --batch_size 8 \
        --epochs 4 \
        --warmup_epochs 1 \
        --accum_iter 1 \
        --blr 10e-4 \
        --weight_decay 0.02 \
        --llama_type "7B" \
        --llama_path "/home/u8915687/lab/big-superb/Macaw-LLM2/weights/llama/" \
        --output_dir "$OUTPUT_DIR" \
        --pretrained_path "/home/u8915687/lab/big-superb/LLaMA-Adapter/imagebind_LLM/ckpts/7B.pth" \
        --data_config "/home/u8915687/lab/big-superb/LLaMA-Adapter/imagebind_LLM/exps/config.yaml" \
        &>> "$OUTPUT_DIR"/output.log

python bigsuperb_inference_test.py --exp_path "$OUTPUT_DIR" --model_path "checkpoint-3.pth" --decode_300 "true"