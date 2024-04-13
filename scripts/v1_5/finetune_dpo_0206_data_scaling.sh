#!/bin/bash

# Directory containing the subset JSON files
DATA_DIR="/home/ubuntu/latest_llava/LLaVA/playground/data/dpo/0105_self-sampled/data_scaling_experiments"

# Updated base name for your run
BASE_RUN_NAME="0206_datascaling_images"

# Base output directory
BASE_OUTPUT_DIR="./checkpoints"

# Updated model name (assuming it's directly under the checkpoints folder)
MODEL_NAME="./checkpoints/llava-v1.5-13b"

# Image folder (assuming it remains constant for all runs)
IMAGE_FOLDER="/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017/"

# Iterate over the subset files in the directory
for FILE_PATH in $DATA_DIR/subset_*resample*.json; do
    # Extract the percentage value from the filename
    PERCENTAGE=$(echo $FILE_PATH | grep -oP '(?<=subset_)\d+(?=%)')

    # Update the run name with the percentage postfix
    RUN_NAME="${BASE_RUN_NAME}-${PERCENTAGE}%"

    # Specify the output directory for this run
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${RUN_NAME}"

    # Execute the main command with the updated DATA_PATH and run_name
    deepspeed llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --deepspeed ./scripts/zero3.json \
        --model_name_or_path ${MODEL_NAME} \
        --version v1 \
        --task DPO --dpo_use_average False --dpo_beta 0.1 \
        --data_path ${FILE_PATH} \
        --image_folder ${IMAGE_FOLDER} \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length False \
        --bf16 True \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 500 \
        --save_total_limit 1 \
        --learning_rate 5e-5 \
        --is_multimodal True \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 3000 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name ${RUN_NAME} \

    echo "Completed run for subset ${PERCENTAGE}%"
done
