#!/bin/bash

# Base paths and settings
IMAGE_FOLDER=/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017
MODEL_BASE=llava-v1.5-13b
MODEL_PATH=./checkpoints/${MODEL_BASE}
OUTPUT_DIR_BASE=./checkpoints
DEEPSPEED_CONFIG=./scripts/zero3.json
VISION_TOWER=openai/clip-vit-large-patch14-336

# Subsets directory
SUBSETS_DIR=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/data_scaling_rejection_sampling

# Iterate over each subset (25%, 50%, and 75%)
for PERCENTAGE in 25 50 75; do
    # Update DATA_PATH for the current subset
    DATA_PATH=${SUBSETS_DIR}/cleaned_subset_${PERCENTAGE}%.json

    # Update run_name to include the percentage and other details
    RUN_NAME="LLaVa_Rejection_subset_${PERCENTAGE}%_lora_5034remake"

    # Specify the output directory for this run
    OUTPUT_DIR=${OUTPUT_DIR_BASE}/${RUN_NAME}

    # Run the experiment with the current settings
    deepspeed llava/train/train_mem.py \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --freeze_mm_mlp_adapter False \
        --deepspeed ${DEEPSPEED_CONFIG} \
        --model_name_or_path ${MODEL_PATH} \
        --version v1 \
        --data_path ${DATA_PATH} \
        --image_folder ${IMAGE_FOLDER} \
        --vision_tower ${VISION_TOWER} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length False \
        --bf16 True \
        --output_dir ${OUTPUT_DIR} \
        --num_train_epochs 3 \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 2 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 300 \
        --save_total_limit 1 \
        --learning_rate 4e-4 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb \
        --run_name ${RUN_NAME}

    echo "Completed run for subset ${PERCENTAGE}%"
done
