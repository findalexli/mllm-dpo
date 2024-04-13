model_name=llava-v1.5-13b-steer-lora-helpsteerformat-interleaf-lrv_0-3k-llava-0-3k-likert-rerun
model_base=llava-v1.5-13b
export CUDA_VISIBLE_DEVICES=2

mkdir logs/${model_name}

ORIGINAL_SCRIPT="/home/ubuntu/LLaVA/scripts/v1_5/eval/sqa_steer.sh"
# Create a temporary modified Bash script
TEMP_SCRIPT=/tmp/${model_name}temp_script.sh
# Replace model_name and model_base in the Bash script
cat $ORIGINAL_SCRIPT | \
    sed "s/model_name=.*$/model_name=$model_name/" | \
    sed "s/model_base=.*$/model_base=$model_base/" > $TEMP_SCRIPT

# Make the temporary Bash script executable

chmod +x $TEMP_SCRIPT
(cat $TEMP_SCRIPT; bash $TEMP_SCRIPT) | tee logs/${model_name}-scienceqa.log