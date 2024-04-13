#!/bin/bash
model_name=llava-v1.5-13b-steer-lora-helpsteerformat-interleaf-lrv1500llava2000-likert-1219_interleaved_gpt4only3337samples
model_base=llava-v1.5-13b
export CUDA_VISIBLE_DEVICES=2
python -m llava.eval.model_vqa \
    --model-path /home/ubuntu/LLaVA/checkpoints/${model_name} \
    --model-base /home/ubuntu/LLaVA/checkpoints/${model_base} \
    --question-file /home/ubuntu/LLaVA/playground/data/eval/mm-vet/llava-mm-vet-helpfulsteer-prompted.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${model_name}all4.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${model_name}all4.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${model_name}all4.jsonl