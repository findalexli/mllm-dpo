model_name=instructblip-vicuna-13b
# prompted_questions=questions_helpfulsteer_prompted.jsonl
original_questions=questions.jsonl
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m llava.eval.gen_model_answer_instructblip \
    --model-id ${model_name}-random-coco \
    --num-gpus-total 4 \
    --use-random-cocoimage True \

