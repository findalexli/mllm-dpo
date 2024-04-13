#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
model_name=llava-v1.5-13b-lora-dpo-lrv--1klrv-1kllava-self-sampled-logpFrom13Bsteersft-1231
model_base=llava-v1.5-13b-steer-lora-helpsteerformat-interleaf-lrv3067llava3067-likert-1229_interleavedgpt4v_responseonly-pure-merged
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base}\
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${model_name}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${model_name}.jsonl
