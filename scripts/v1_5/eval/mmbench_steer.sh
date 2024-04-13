#!/bin/bash
SPLIT="mmbench_dev_20230712"
model_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unfrozen-adapter-no-scigraph-context--added-imagetoken-26-samples-0204REMAKE-DATA
model_base=llava-v1.5-13b
export CUDA_VISIBLE_DEVICES=2
python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base} \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${model_name}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${model_name}