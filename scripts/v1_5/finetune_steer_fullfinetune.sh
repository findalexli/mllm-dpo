#!/bin/bash
DATA_PATH=/home/ubuntu/LLaVA/playground/data/steer_sft_model/training/extracted_attributes_sft_helpsteer4046-5879-likert-1209-withinbatch.json
IMAGE_FOLDER=/home/ubuntu/train2017
run_name=llava-v1.5-7b-steer-lora-helpsteerformat-4046-5879-withinbatch0-1214-from-llava-pretrained-fullfinetune-another2epoch
ouput_dir=./checkpoints/${run_name}
# Notice that I am loading the latest model checkopint 
model_name=/home/ubuntu/LLaVA/checkpoints/llava-v1.5-7b-steer-lora-helpsteerformat-4046-5879-withinbatch0-1214-from-llava-pretrained-fullfinetune

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ${model_name} \
    --version v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ${ouput_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
    --run_name ${run_name} \