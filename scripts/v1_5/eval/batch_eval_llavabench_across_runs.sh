export CUDA_VISIBLE_DEVICES=0
# Define the hyperparameters arrays

origionaL_questions=questions.jsonl
model_base=llava-v1.5-13b-steer-lora-helpsteerformat-interleaf-lrv3067llava3067-likert-1222_interleavedgpt4v_responseonly-pure-merged
# Loop through each combination of hyperparameters
declare -a dpo_beta_values=(0.1 0.3 0.5)
declare -a learning_rates=(5e-5 5e-6)
declare -a num_train_epochs_integers=(2 3)
# Loop through each combination of hyperparameters
for dpo_beta in "${dpo_beta_values[@]}"; do
    for learning_rate in "${learning_rates[@]}"; do
        for num_train_epoch in "${num_train_epochs_integers[@]}"; do

            # Configure the run name to reflect hyperparameter choices
            model_name="llava-lora-dpo-1227lrvtail2000_sft-self-sampled-beta-${dpo_beta}-lr-${learning_rate}-avg-False-epoch-${num_train_epoch}"

            python -m llava.eval.model_vqa \
                --model-path /home/ubuntu/LLaVA/checkpoints/${model_name} \
                --model-base /home/ubuntu/LLaVA/checkpoints/${model_base} \
                --question-file /home/ubuntu/LLaVA/playground/data/eval/llava-bench-in-the-wild/${origionaL_questions} \
                --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
                --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/${model_name}-origionalprompt-rerun.jsonl \
                --temperature 0 \
                --conv-mode vicuna_v1

            mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

            python llava/eval/eval_gpt_review_bench.py \
                --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
                --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
                --rule llava/eval/table/rule.json \
                --answer-list \
                    playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
                    playground/data/eval/llava-bench-in-the-wild/answers/${model_name}-origionalprompt-rerun.jsonl \
                --output \
                    playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}-origionalprompt-rerun.jsonl

            python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/${model_name}-origionalprompt-rerun.jsonl
            # Add any other relevant flags or options for deepspeed command

            echo "Experiment ${model_name} done!"
        done
    done
done