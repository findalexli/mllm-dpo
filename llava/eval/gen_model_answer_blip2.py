"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm

from fastchat.llm_judge.common import load_questions, temperature_config
# from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
import argparse
import torch
import os
import json
import pdb
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image
import math
import random

def get_random_cocoimage():
    IMAGE_FOLDER = "/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017"

    # Get a list of all .jpg files in the folder
    jpg_files = [file for file in os.listdir(IMAGE_FOLDER) if file.endswith(".jpg")]

    # Randomly pick a file from the list
    random_file = random.choice(jpg_files)

    # Construct the full file path
    file_path = os.path.join(IMAGE_FOLDER, random_file)

    # Open the image using PIL's Image module
    random_cocoimage = Image.open(file_path)
    print("random_cocoimage: ", file_path)
    # Display the image
    return random_cocoimage

def run_eval(
    model_path,
    model_base,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    use_random_cocoimage,
):
    questions = load_questions(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_base,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                use_random_cocoimage=use_random_cocoimage,
            )
        )

    if use_ray:
        ray.get(ans_handles)


@torch.inference_mode()
def get_model_answers(model_path, model_base, model_id, questions, answer_file, max_new_token, num_choices, num_gpus_per_model, max_gpu_memory, dtype, revision, use_random_cocoimage):
    # Assuming disable_torch_init() is a function defined elsewhere that you're using
    disable_torch_init()

    # Load model and processor
    model_id = 'Salesforce/blip2-flan-t5-xxl'
    processor = AutoProcessor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    for question in tqdm(questions):
        choices = []
        
        # Determine the temperature based on question category, default to 0.7
        temperature = temperature_config.get(question["category"], 0.7)

        for i in range(num_choices):
            turns = []  # A list to hold all turns in the conversation for this choice
            prompt = "Question: {} Answer: ".format(question["turns"][0])  # Initialize the prompt with the first question
            
            for j in range(len(question["turns"])):
                if j > 0:
                    prompt += "Question: {} Answer: ".format(question["turns"][j])

                random_cocoimage = None
                if use_random_cocoimage:
                    random_cocoimage = get_random_cocoimage()  # Ensure this function is defined

                # Process the prompt with the potential image and generate the answer
                inputs = processor(random_cocoimage, text=prompt, return_tensors="pt").to("cuda", dtype=torch.float16)
                do_sample = temperature >= 1e-4
                outputs = model.generate(
                    **inputs,
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                    use_cache=True,
                )
                generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()

                # Update the prompt with the generated answer for the next iteration
                prompt = prompt.rstrip(" Answer: ") + "Answer: " + generated_text + " "
                turns.append(generated_text)  # Save the generated answer for this turn

            choices.append({"index": i, "turns": turns})  # After all turns, save the generated sequence of answers for this choice

        # Dump answers to the file
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        required=False,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-base",
        type=str,
        required=False,
        help="The path to the base LLAVA weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, required=True, help="A custom name for the model."
    )
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--conv-mode",
        type=str,
        default="vicuna_v1",
    )
    parser.add_argument(
        "--use-random-cocoimage",
        type=bool,
        default=True,
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"llava/eval/data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        answer_file = f"llava/eval/data/{args.bench_name}/model_answer/{args.model_id}.jsonl"

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_base=args.model_base,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        use_random_cocoimage=args.use_random_cocoimage,
    )

    reorg_answer_file(answer_file)
