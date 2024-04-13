#!/bin/bash

# Define the arguments
model_name=llava-13b-0203-all--image-preference-sci3k-lrv3k-lora-dpo-beta-0.1-lr-5e-5-avg-False-question-addedimagetoken
model_base=llava-v1.5-13b
model_path=./checkpoints/${model_name}
model_base=./checkpoints/${model_base}
# model_path=/home/ubuntu/RLHF/LLaVA-RLHF-13b-v1.5-336/sft_model_llava
# model_base=/home/ubuntu/RLHF/LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model
save_model_path=./checkpoints/${model_name}-merged

# Call the Python script
python scripts/merge_lora_weights.py --model-path $model_path --model-base $model_base --save-model-path $save_model_path