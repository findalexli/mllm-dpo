#!/bin/bash
"""
This script will fun the mm-vet evaluation using complex2verbose2 prompting template with helpful steer
"""
# question_file=llava-mm-vet.json

# Define your custom model parameters
export CUDA_VISIBLE_DEVICES=3
model_name=LLaVa_Rejection_subset_50%_lora_5034remake
model_base=llava-v1.5-13b
question_file=llava-mm-vet-helpfulsteer-promptedcomplex2verbo2.jsonl
question_file=llava-mm-vet-helpfulsteer-prompted.jsonl
# question_file=llava-mm-vet.jsonl
question_file_base=base=$(basename "$question_file" .jsonl)
model_name_for_extraction=${model_name}-${question_file_base}

# Path to the original Bash script
ORIGINAL_SCRIPT="./scripts/v1_5/eval/mmvet_steer.sh"

# Create a temporary modified Bash script
TEMP_SCRIPT=/tmp/temp_script-${model_name_for_extraction}.sh

# Replace model_name and model_base in the Bash script
cat $ORIGINAL_SCRIPT | \
    sed "s/model_name=.*$/model_name=$model_name/" | \
    sed "s/model_base=.*$/model_base=$model_base/" | \
    sed "s/question_file=.*/question_file=$question_file/" > $TEMP_SCRIPT

# Make the temporary Bash script executable
chmod +x $TEMP_SCRIPT

# Path to the original Python script
PYTHON_SCRIPT=./playground/data/eval/mm-vet/mm-vet-evaluator.py

# Create a temporary modified Python script
TEMP_PYTHON_SCRIPT=/tmp/temp_python_script_${model_name_for_extraction}.py

# Replace model variable in the Python script and append the postfix
cat $PYTHON_SCRIPT | \
    sed "s/model_name_unique = .*/model_name_unique = \"$model_name_for_extraction\"/" > $TEMP_PYTHON_SCRIPT

# Run the temporary Bash script
$TEMP_SCRIPT
python $TEMP_PYTHON_SCRIPT

# Remove the temporary scripts after execution
rm $TEMP_SCRIPT
rm $TEMP_PYTHON_SCRIPT
