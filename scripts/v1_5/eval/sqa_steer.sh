#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
model_name=llava-v1.5-13b-to_lrv10k-13kScigraph_llava5k_pureddp0105-lora
model_base=llava-v1.5-13b
question_prompt_file=llava_test_CQM-A-helpfulsteerprompted-all4.json
origional_file=llava_test_CQM-A.json

question_file=${origional_file}
question_file_base=base=$(basename "$question_file" .jsonl)
python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base}\
    --question-file ./playground/data/eval/scienceqa/${question_file} \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/${model_name}-${question_file_base}.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/${model_name}-${question_file_base}.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${model_name}-${question_file_base}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${model_name}-${question_file_base}_result.json
