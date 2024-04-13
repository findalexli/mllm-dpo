#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=llava-v1.5-13b-to_lrv10k-13kScigraph_llava5k_pureddp0105-lora
model_base=llava-v1.5-13b
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base} \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${model_name}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${model_name}.jsonl