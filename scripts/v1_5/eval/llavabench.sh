#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unfrozen-adapter-no-scigraph-context--added-imagetoken-26-samples-0204REMAKE-DATA
model_base=llava-v1.5-13b
prompted_questions=questions_helpfulsteer_prompted.jsonl
origionaL_questions=questions.jsonl
# # All 4 prompted 
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/${model_name} \
#     --model-base ./checkpoints/${model_base} \
#     --question-file ./playground/data/eval/llava-bench-in-the-wild/${prompted_questions} \
#     --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${model_name}-all4-prompted-rerun-non-loaded.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# python llava/eval/eval_gpt_review_bench.py \
#     --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
#     --rule llava/eval/table/rule.json \
#     --answer-list \
#         playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#         playground/data/eval/llava-bench-in-the-wild/answers/${model_name}-all4-prompted-rerun-non-loaded.jsonl \
#     --output \
#         playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}-all4-prompted-rerun-non-loaded.jsonl


# python llava/eval/summarize_gpt_review.py -f ./playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}-all4-prompted-rerun-non-loaded.jsonl

# # This is the complex2verbose2 prompt
# python -m llava.eval.model_vqa \
#     --model-path ./checkpoints/${model_name} \
#     --model-base ./checkpoints/${model_base} \
#     --question-file ./playground/data/eval/llava-bench-in-the-wild/questions_helpfulsteer_promptedcomplex2verbo2.jsonl \
#     --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
#     --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${model_name}complex2verbose2-rerun.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

# python llava/eval/eval_gpt_review_bench.py \
#     --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
#     --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
#     --rule llava/eval/table/rule.json \
#     --answer-list \
#         playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
#         playground/data/eval/llava-bench-in-the-wild/answers/${model_name}complex2verbose2-rerun.jsonl \
#     --output \
#         playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}complex2verbose2-rerun.jsonl

# python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}complex2verbose2-rerun.jsonl

# This is the origional prompt

python -m llava.eval.model_vqa \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base} \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/${origionaL_questions} \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${model_name}-origionalprompt-rerun.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
    --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
    --rule llava/eval/table/rule.json \
    --answer-list \
        playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
        playground/data/eval/llava-bench-in-the-wild/answers/${model_name}-origionalprompt-rerun.jsonl \
    --output \
        playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}-origionalprompt-rerun.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}-origionalprompt-rerun.jsonl