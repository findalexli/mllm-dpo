#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA_PATH=./playground/data/dpo/dpo_logp_lrvtail2000_sft-self-sampled.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/dpo/1231-13b-selfsamples/3118samples_1419llava_1699lrv_interleaved_1231_logprob_generatedby13b-steersft.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/dpo/0108_dpo_7b/lrv10k-13k-scigraph3k-helpsteer6k_0107_7b_4k.json

IMAGE_FOLDER=/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017/
# Define the constant paths
MODEL_NAME=./checkpoints/llava-v1.5-7b
# Define the hyperparameters arrays
declare -a dpo_beta_values=(0.3 0.5)
declare -a learning_rates=(5e-5 5e-6)
declare -a num_train_epochs_integers=(3)
# Loop through each combination of hyperparameters
for dpo_beta in "${dpo_beta_values[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        # Configure the run name to reflect hyperparameter choices
        run_name="llava-7b-dpo-0108-4k-subset${dpo_beta}-lr-${learning_rate}-avg-False-lora"

        # Set the output directory
        ouput_dir="./checkpoints/${run_name}"

        # Training command with hyperparameters
        deepspeed llava/train/train.py \
            --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
            --deepspeed ./scripts/zero3.json \
            --model_name_or_path ${MODEL_NAME} \
            --version v1 \
            --task DPO --dpo_beta ${dpo_beta} --dpo_use_average False \
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
            --num_train_epochs 3 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 4 \
            --gradient_accumulation_steps 4 \
            --evaluation_strategy "no" \
            --save_strategy "steps" \
            --save_steps 50 \
            --save_total_limit 1 \
            --learning_rate ${learning_rate} \
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
            --run_name ${run_name} \

        # Add any other relevant flags or options for deepspeed command

        echo "Experiment ${run_name} done!"
    done
done
