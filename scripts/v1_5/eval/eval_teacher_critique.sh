#!/bin/bash
model_name=llava-v1.5-13b
IMAGE_FOLDER=/home/ubuntu/train2017
input_data_filename=teacher-critique-0-2125
python -m llava.eval.eval_teacher_critique \
    --model-path ./checkpoints/${model_name} \
    --question-file ./playground/data/teacher-critique/${input_data_filename}.json \
    --image-folder ${IMAGE_FOLDER} \
    --answers-file ./playground/data/teacher-critique/answers/${model_name}-on-${input_data_filename}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1