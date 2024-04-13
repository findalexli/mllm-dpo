model_name=llava-v1.5-13-to-lrv10k-13k-scigraph3000-SteerLM-lora-unfrozen-adapter-no-scigraph-context--added-imagetoken-26-samples-0204REMAKE-DATA
model_base=llava-v1.5-13b
origional_question_file=llava_pope_test.jsonl
prompted_question_file=llava_pope_test_helpfulsteer_prompted-all4.jsonl
question_file_string=${origional_question_file}
export CUDA_VISIBLE_DEVICES=3


question_file_base=${question_file_string%.*}
python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/${model_name} \
    --model-base ./checkpoints/${model_base} \
    --question-file ./playground/data/eval/pope/${question_file_string} \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/${model_name}-${question_file_base}.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/${model_name}-${question_file_base}.jsonl