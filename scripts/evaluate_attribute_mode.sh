mkdir ./playground/data/attribute_model/answers/
python -m llava.eval.model_vqa \
    --model-path /home/ubuntu/LLaVA/checkpoints/llava-v1.5-13b-attribute-reward-model-1poch-lora-1118newprompt-2e-5 \
    --model-base /home/ubuntu/LLaVA/checkpoints/llava-v1.5-13b \
    --question-file /home/ubuntu/LLaVA/playground/data/attribute_model/validation_data/extracted_attributes_for_attribute_modelling_30_1225-1425-newprompt1118.json \
    --image-folder /home/ubuntu/train2017 \
    --answers-file ./playground/data/attribute_model/answers/attribute_predictions_1225-1425_llava-1118-corrected.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1