import torch
import os
import json
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path
from llava.train.train import preference_collator_fn
from functools import partial
import tqdm
from llava.train.llava_trainer import get_batch_logps
from PIL import Image
import math
disable_torch_init()
model_path = '/home/ubuntu/latest_llava/LLaVA/checkpoints/llava-v1.5-13b'
model_base = None
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)

use_im_start_end= False # Default setting from llava 1.5 
intermediate_csv = '/home/ubuntu/latest_llava/LLaVA/playground/data/dpo/ultrafeedback_binarized_2k_sample.csv'
IMAGE_FOLDER="/home/ubuntu/latest_llava/llava_1dot5data/coco/train2017/"
inference_data_path = intermediate_csv
output_path = intermediate_csv.split('csv')[0] + 'with_logp.json'
from llava.train.train import encode_multimodal_preference_sample
import torch.utils.data as torch_data

class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data_path,
                 tokenizer,
                 img_processor,
                 use_im_start_end):
        self.data = pd.read_csv(data_path)
        # self.data = self.filter_df(self.data, sample_size=2000)
        for col in ['chosen', 'rejected', 'question']:
            self.data[col] = self.data[col].astype(str)
        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': False,
            # 'image_token_len': image_token_len, # TODO check if needed
            'use_im_start_end': use_im_start_end,
            'image_aspect_ratio': 'pad'
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # After encode multimodal preference sample, 
        # the image would ahve the pixel values values
        sample = self.convert_dataframe_row_to_source(self.data.iloc[index])
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(sample, self.tokenizer, self.mm_cfg)
        return rej_data_dict, win_data_dict
    
    def _get_image(self, img_filename):
        img_path = os.path.join(IMAGE_FOLDER, img_filename)
        image = Image.open(img_path).convert('RGB')
        return image
    
    def filter_df(self, df, sample_size=2000):
        filtered_df = df[~df['rejected'].str.contains("</s>") & ~df['chosen'].str.contains("</s>") & ~df['question'].str.contains("</s>")]
        if sample_size:
            filtered_df = filtered_df.sample(sample_size, random_state=42)
        return filtered_df
    
    def _convert_to_llava_answer_turn(self, answer):
        return {"from": "gpt", "value": answer}
    
    def _convert_to_llava_question_turn(self, question):
        return {"from": "human", "value": question}

    def convert_dataframe_row_to_source(self, row):
        dict_output = {'question': self._convert_to_llava_question_turn(row['question']), 
                       # Chosen and rejected based on the sum of the first three quality score
                    'chosen': self._convert_to_llava_answer_turn(row['chosen']),
                    'rejected': self._convert_to_llava_answer_turn(row['rejected'])}
        return dict_output

    def __len__(self):
        return len(self.data)
    
preference_torch_dataset = PreferenceInferenceDataset(inference_data_path,
                           tokenizer=tokenizer,
                           img_processor=image_processor,
                           use_im_start_end=False)
preference_torch_dataset[0]

dataset = preference_torch_dataset
collate_fn = partial(preference_collator_fn, pad_token_id=tokenizer.pad_token_id)
dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                    num_workers=5, shuffle=False)
win_logp_list = []
rej_logp_list = []

win_avg_logp_list = []
rej_avg_logp_list = []

win_per_token_logp_list = []
rej_per_token_logp_list = []

with torch.inference_mode():
    for batch in tqdm.tqdm(dataloader, miniters=50):
        for key in ['win', 'rej']:
            input_ids = batch[f'{key}_input_ids'].cuda()
            labels = batch[f'{key}_labels'].cuda()
            attention_mask = batch[f'{key}_attention_mask'].cuda()

            output = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                images=None,
            )
            per_token_logp, log_prob, average_log_prob = get_batch_logps(output.logits, labels, return_all=True)

            # print(per_token_logp.shape, input_ids.shape, labels.shape, flush=True)
            assert per_token_logp.size(1) >= input_ids.size(1) - 1
            per_token_logp = per_token_logp.tolist()
            # per_token_logp = [x[:input_ids[i].ne(tokenizer.pad_token_id).sum().item()] for i, x in enumerate(per_token_logp)]
            log_prob = log_prob.tolist()
            average_log_prob = average_log_prob.tolist()

            if key == 'win':
                win_logp_list += log_prob
                win_avg_logp_list += average_log_prob
                win_per_token_logp_list += per_token_logp
            else:
                rej_logp_list += log_prob
                rej_avg_logp_list += average_log_prob
                rej_per_token_logp_list += per_token_logp
            # print(f'{key} logits in {output.logits.shape}, logp in {log_prob.shape} avg_logp in {average_log_prob.shape}')

df = pd.read_csv(inference_data_path)
def _convert_to_llava_answer_turn(answer):
    return {"from": "gpt", "value": answer}
# Add each list as a column to the dataframe with its variable name
df['win_logp'] = win_logp_list
df['rej_logp'] = rej_logp_list
df['win_avg_logp'] = win_avg_logp_list
df['rej_avg_logp'] = rej_avg_logp_list
df['win_per_token_logp'] = win_per_token_logp_list
df['rej_per_token_logp'] = rej_per_token_logp_list
exisitng_columns = ['rej_logp', 
                    'win_logp',
                    'rej_avg_logp',
                    'win_avg_logp',
                    'rej_per_token_logp',
                    'win_per_token_logp']
for col in exisitng_columns:
    rename = 'ref_' + col
    df[rename] = df[col]
    del df[col]
df['chosen'] = df['chosen'].apply(_convert_to_llava_answer_turn)
df['rejected'] = df['rejected'].apply(_convert_to_llava_answer_turn)
def _convert_to_llava_question_turn(question):
    return {"from": "human", "value": question}
df['question'] = df['question'].apply(_convert_to_llava_question_turn)
df.to_json(output_path, orient='records')