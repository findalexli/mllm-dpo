## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## Introduction

This guide provides step-by-step instructions for fine-tuning using the alignment mehotds and evaluating the LLaVA model, specifically focusing on visual instruction tuning using SciGraphQA and LRV-instruct datasets.

## Installation

1. **Unzip the repository:**


2. **Set up the environment:**

    ```bash
    conda create -n llava python=3.10 -y
    conda activate llava
    pip install --upgrade pip
    pip install -e .
    ```

3. **Install packages for training:**

    ```bash
    pip install -e ".[train]"
    pip install flash-attn --no-build-isolation
    ```

## Data Preparation

1. **Download datasets and images:**
   - SciGraphQA: [Download Link](https://huggingface.co/datasets/alexshengzhili/SciGraphQA-295K-train)
   - LRV-Insturct: [Download Link](https://github.com/FuxiaoLiu/LRV-Instruction)

2. **Organize the images in `./playground/data`:**

    ```
    playground/
    └── data/
        ├── scigraphqa/
        │   └── images/
        └── lrv_instruct/
            └── images/
    ```

3. For DPO data, the dataset with pre-computed log likelihood are also provided in the submission
4. For non-DPO data, we also provide each of the alignment method (SteerLM, Rejection Sampling and Standard SFT) in the zipped data 
## Training

1. Use scripts/v1/finetune_dpo.sh for DPO experiments
2. Use scripts/v1/finetune_steer.sh for non-DPO experiments

## Evaluation

1. **Use the provided evaluation scripts under scripts/v1_5/eval/ to assess the performance of your fine-tuned model on various benchmarks.** Ensure that you follow the guidelines for using greedy decoding to ensure consistency with real-time outputs.

# We thank the authors of LLaVA, Vicuna for which the origional state of this repo is based on
