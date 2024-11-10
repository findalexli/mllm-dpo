This repo contains the code and the data for the following paper:

```bibtex
@inproceedings{li-etal-2024-multi,
    title = "Multi-modal Preference Alignment Remedies Degradation of Visual Instruction Tuning on Language Models",
    author = "Li, Shengzhi  and
      Lin, Rongyu  and
      Pei, Shichao",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.765",
    pages = "14188--14200",
    abstract = "Multi-modal large language models (MLLMs) are expected to support multi-turn queries of interchanging image and text modalities in production. However, the current MLLMs trained with visual-question-answering (VQA) datasets could suffer from degradation, as VQA datasets lack the diversity and complexity of the original text instruction datasets with which the underlying language model was trained. To address this degradation, we first collect a lightweight, 5k-sample VQA preference dataset where answers were annotated by Gemini for five quality metrics in a granular fashion and investigate standard Supervised Fine-tuning, rejection sampling, Direct Preference Optimization (DPO) and SteerLM algorithms. Our findings indicate that with DPO, we can surpass the instruction-following capabilities of the language model, achieving a 6.73 score on MT-Bench, compared to Vicuna{'}s 6.57 and LLaVA{'}s 5.99. This enhancement in textual instruction-following capability correlates with boosted visual instruction performance (+4.9{\%} on MM-Vet, +6{\%} on LLaVA-Bench), with minimal alignment tax on visual knowledge benchmarks compared to the previous RLHF approach. In conclusion, we propose a distillation-based multi-modal alignment model with fine-grained annotations on a small dataset that restores and boosts MLLM{'}s language capability after visual instruction tuning.",
}
```
[[Arxiv paper](https://arxiv.org/abs/2402.10884)]
[[GitHub](https://github.com/findalexli/mllm-dpo)]
[[Data](https://huggingface.co/datasets/alexshengzhili/llava-preference-dpo)]
[[Model](https://huggingface.co/alexshengzhili/llava-v1.5-13b-dpo/edit/main/README.md)]
[[Data](https://huggingface.co/datasets/alexshengzhili/mllm-dpo)]
[[Youtube](https://youtu.be/uv_ZtwA8pkY)]

Developers: Shengzhi Li (TIFIN.AI), Rongyu Lin (KAUST), Shichao Pei (University of Massachusetts Boston)
Affiliations: TIFIN, KAUST, University of Massachusetts Boston
Contact Information: alex.li@tifin.com, rongyu.lin@kaust.edu.sa, shichao.pei@umb.edu
# Youtube video

[![Youtube](https://img.youtube.com/vi/uv_ZtwA8pkY/0.jpg)](https://www.youtube.com/watch?v=uv_ZtwA8pkY) 


## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)

## Introduction

This guide provides step-by-step instructions for fine-tuning using the alignment methods and evaluating the LLaVA model, specifically focusing on visual instruction tuning using SciGraphQA and LRV-instruct datasets.

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

    The images for LRC-Instruct shall be downloaded by: 
gdown https://drive.google.com/uc?id=1k9MNV-ImEV9BYEOeLEIb4uGEUZjd3QbM

The images for SciGraphQA can be downloaded by: 
https://huggingface.co/datasets/alexshengzhili/SciGraphQA-295K-train/resolve/main/img.zip?download=true
2. **Organize the images in `./playground/data`:**

    ```
    playground/
    └── data/
        ├── scigraphqa/
        │   └── images/
        └── lrv_instruct/
            └── images/
    ```

3. For DPO, please see playground/data/dpo_inference0104.with_logpllava-v1.5-13b_2024-02-03.json
4. For non-DPO data, we also provide each of the alignment method (SteerLM, Rejection Sampling and Standard SFT) in the data folder such as
playground/data/rejection_sampling.json
playground/data/standard_sft.json
playground/data/steerlm.json
## Training

1. Use scripts/v1/finetune_dpo.sh for DPO experiments
2. Use scripts/v1/finetune_steer.sh for non-DPO experiments, 


## Evaluation

1. **Use the provided evaluation scripts under scripts/v1_5/eval/ to assess the performance of your fine-tuned model on various benchmarks.** Ensure that you follow the guidelines for using greedy decoding to ensure consistency with real-time outputs.

We thank the authors of LLaVA, Vicuna for which the origional state of this repo is based on


