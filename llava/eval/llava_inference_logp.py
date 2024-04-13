import os
import json
import tqdm
import torch
import base64
import torch.utils.data as torch_data

from typing import List
from functools import partial

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob

class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data_dir,
                 tokenizer,
                 tsv_filenames: List[str],
                 image_token_len,
                 img_processor,
                 use_im_start_end):
        if 'DPO_preference_llava' in data_dir or 'llavarlhf' in tsv_filenames[0]:
            self.data = SingleDataSourceDataset('dpo_preference_llava_7b_v1_preference_hallonly' ,data_dir, tsv_filenames)
        else:
            self.data = SingleDataSourceDataset('RLHF-V-Hall_v0' ,data_dir, tsv_filenames)

        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': True,
            'image_token_len': image_token_len,
            'use_im_start_end': use_im_start_end
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.data[index]
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(sample, self.tokenizer, self.mm_cfg)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)