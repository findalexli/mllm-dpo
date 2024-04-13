model_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unfrozen-adapter-no-scigraph-context--added-imagetoken-26-samples-0204REMAKE-DATA
model_base=llava-v1.5-13b
# prompted_questions=questions_helpfulsteer_prompted.jsonl
original_questions=questions.jsonl
python -m llava.eval.gen_model_answeralpaca \
    --model-path checkpoints/${model_name} \
    --model-base checkpoints/${model_base} \
    --model-id ${model_name} \
    --num-gpus-total 3 \