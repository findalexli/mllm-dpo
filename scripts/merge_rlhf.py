from transformers import LlavaLlamaForCausalLM
from transformers import PeftModel
import torch
dtype = torch.bfloat16

model_path = "LLaVA-RLHF-13b-v1.5-336/sft_model"
lora_path = "LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model"

model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    device_map={"": "cuda:0"},
    torch_dtype=dtype,
)

model = PeftModel.from_pretrained(
    model,
    lora_path,
)
model = model.merge_and_unload()
model.save_pretrained("/home/ubuntu/latest_llava/LLaVA/checkpoints/llava-rlhf-13b-v1.5-336-merged")