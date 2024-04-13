#!/bin/bash

model_name=llava-v1.5-7b
export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/${model_name} \
    --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${model_name}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${model_name}.jsonl

