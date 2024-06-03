import jsonlines
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
import math
import torch
from torch.utils.data import Dataset
from config.utils import *
from config.options import *
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from transformers import BertTokenizer
import clip

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def init_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  
    return tokenizer


class ScoreDataset(Dataset):
    def __init__(self, jsonl_file, dim=224, image_prefix="/home/eprakash/diffusers/examples/controlnet/train_data/"):
        self.data = []
        self.image_prefix = image_prefix
        self.dim = dim
        self.transform = _transform(dim)
        self.tokenizer = init_tokenizer()
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                self.data.append(obj)
        self.iters_per_epoch = int(math.ceil(len(self.data)*1.0/opts.batch_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        def load_image(image_path):
            if image_path != "N/A":
                image = Image.open(self.image_prefix + image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
            else:
                image = ToTensor()(Image.new('RGB', (self.dim, self.dim), color=(0, 0, 0)))
            return image

        def convert_rank(rank):
            if rank != "N/A":
                return int(rank)
            return -1

        # Load images if not "N/A"
        image_1 = load_image(item['image_1'])
        conditioning_image = load_image(item['conditioning_image'])
        image_2 = load_image(item['image_2'])

        # Convert ranks to int if not "N/A"
        rank_t_1 = convert_rank(item['rank_t_1'])
        rank_m_1 = convert_rank(item['rank_m_1'])
        rank_t_2 = convert_rank(item['rank_t_2'])
        rank_m_2 = convert_rank(item['rank_m_2'])
        
        text_input = self.tokenizer(item['text'], padding='max_length', truncation=True, max_length=77, return_tensors="pt")
        text = clip.tokenize(item['text'], truncate=True)
        text_ids = text_input.input_ids
        text_mask = text_input.attention_mask
        
        # Create the output dictionary
        sample = {
            'orig_text': item['text'],
            'orig_conditioning_image': item['conditioning_image'],
            'text': text,
            'text_ids': text_ids,
            'text_mask': text_mask,
            'image_1': image_1,
            'conditioning_image': conditioning_image,
            'image_2': image_2,
            'rank_t_1': rank_t_1,
            'rank_m_1': rank_m_1,
            'rank_t_2': rank_t_2,
            'rank_m_2': rank_m_2
        }
        return sample
'''
# Instantiate the dataset
dataset = ScoreDataset('train_dpo_fixed.jsonl')

# Instantiate the dataloader
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

# Example of iterating through the dataloader
for batch in dataloader:
    print(batch)
    break
'''
