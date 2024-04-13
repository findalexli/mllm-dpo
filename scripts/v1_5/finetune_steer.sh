#!/bin/bash
SFT_DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/steer_sft_model/training/gpt4_response_only/extracted_attributes_sft_helpsteer_lrv1500llava2000-likert-1219_interleaved_gpt4only3337samples-pure.json
STEER_DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/steer_sft_model/training/interleaf_helpsteer_on_llava_lrv/extracted_attributes_sft_helpsteer_lrv1500llava2000-likert-1219_interleaved.json
COH_DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/steer_sft_model/training/interleaf_helpsteer_on_llava_lrv/extracted_attributes_sft_helpsteer_lrv1500llava2000_interleaved_coh.json
full_coh_balanced_data_path=/home/ubuntu/RLHF/COH/SFT_ready/extracted_attributes_COH_helpsteer_lrv2816llava2816_interleaved_balanced.json
lima_500mc500sw_data_path=/home/ubuntu/latest_llava/LLaVA/playground/data/lima/shuffled.json
lima=/home/ubuntu/latest_llava/LLaVA/playground/data/lima/lima.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/lima/lima.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/puresft_lrv10k-13k-scigraph3000_perfectscored_json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/scigraph_lrv10kto13k_GeminiGoldenResponse.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/steer_sft_model/training/0105_self-sampledlrv10k-13k-scigraph3000-inteleaf-for-helpsteer.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/steer_sft_model/training/0105_self-sampledlrv10k-13k-scigraph3000-inteleaf-for-SteerLM_0128_added_imagetoken.json
# DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/puresft_lrv10k-13k-scigraph3000_perfectscored_json_0128_added_imagetoken.json
# DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/scigraph_lrv10kto13k_GeminiGoldenResponse_0129_no_context_no_image.json

DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/puresft_lrv10k-13k-scigraph3000_added_imagetoken_only_1_per_image_quetion_5052samples.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/puresft/puresft_lrv10k-13k-scigraph3000_added_imagetoken_only_1_per_image_quetion_5034samples_0204_remake.json
DATA_PATH=/home/ubuntu/latest_llava/LLaVA/playground/data/steerlm_lrv10k-30k_scigraph3000_0204.json
IMAGE_FOLDER=/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-inteleaf-for-perfect-score-lora-unforzen-adapter
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-GeminiGoldenResponse-lora-unforzen-adapter
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unforzen-adapter-added-imagetoken-noscigraphcontextual
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-inteleaf-for-perfect-score-lora-unforzen-adapter-added-imagetoken
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-GeminiGoldenResponse-lora-unforzen-adapter-no-scigraph-context--noimagetoken
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unfrozen-adapter-no-scigraph-context--added-imagetoken
run_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unfrozen-adapter-no-scigraph-context--added-imagetoken-26-samples-0204REMAKE-DATA
# IMAGE_FOLDER=/home/ubuntu/LLaVA/playground/data/lrv-instruction/image
ouput_dir=./checkpoints/${run_name}
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Notice that I am loading the latest model checkopint 
model_base=llava-v1.5-13b
deepspeed llava/train/train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --freeze_mm_mlp_adapter False \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/${model_base} \
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
    --run_name ${run_name}