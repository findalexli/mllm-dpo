model_name=llava-13b-dpo-lora-0206_data-noising-75%
model_base=llava-v1.5-13b
# prompted_questions=questions_helpfulsteer_prompted.jsonl
original_questions=questions.jsonl
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m llava.eval.gen_model_answer \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base} \
    --model-id ${model_name}-rerun2015 \
    --num-gpus-total 4 \
