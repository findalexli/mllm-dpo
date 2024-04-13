#!/bin/bash
model_name=llava-13b-dpo-lora-0206_datascaling_images-25%
model_base=llava-v1.5-13b
question_file=llava-mm-vet-helpfulsteer-promptedcomplex2verbo2.jsonl
question_file=llava-mm-vet-helpfulsteer-prompted.jsonl
question_file=llava-mm-vet.jsonl
question_file_base=base=$(basename "$question_file" .jsonl)
export CUDA_VISIBLE_DEVICES=0
python -m llava.eval.model_vqa \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base} \
    --question-file ./playground/data/eval/mm-vet/${question_file} \
    --image-folder ./playground/data/eval/mm-vet/images \
    --answers-file ./playground/data/eval/mm-vet/answers/${model_name}-${question_file_base}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
    --src ./playground/data/eval/mm-vet/answers/${model_name}-${question_file_base}.jsonl \
    --dst ./playground/data/eval/mm-vet/results/${model_name}-${question_file_base}.jsonl