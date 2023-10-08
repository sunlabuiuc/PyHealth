import re
import os
import numpy as np
from tqdm import tqdm
import pickle
import json
import retrying
from transformers import AutoTokenizer, AutoModel
from pyhealth import BASE_CACHE_PATH
from collections import defaultdict

def clean_text(x):
    y = re.sub(r'\[(.*?)\]', '', x)  # remove de-identified brackets"
    y = re.sub(r'[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub(r'dr\.', 'doctor', y)
    y = re.sub(r'm\.d\.', 'md', y)
    y = re.sub(r'admission date:', '', y)
    y = re.sub(r'discharge date:', '', y)
    y = re.sub(r'--|__|==', '', y)
    return y


MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "note_embedding")
if not os.path.exists(MODULE_CACHE_PATH):
    os.makedirs(MODULE_CACHE_PATH)
def add_embedding(samples):
    model_name = 'clinical_bert'
    local_filepath = os.path.join(MODULE_CACHE_PATH, model_name + '.json')
    if not os.path.exists(local_filepath):
        text2emb = defaultdict(list)
    else:
        text2emb = json.load(open(local_filepath, 'r'))
        
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    new_samples = []
    for sample in tqdm(samples):
        text = sample['text']
        if text not in text2emb:
            inputs = tokenizer(text, padding=True, truncation=True,
                                max_length=512, return_tensors="pt")
            # Get the model's output 
            outputs = model(**inputs)
            # Extract the embeddings
            embedding = outputs.last_hidden_state.mean(dim=1)
            embedding = embedding.detach().numpy().tolist()[0]
            text2emb[text] = embedding
            sample['embedding'] = embedding
            new_samples.append(sample)
        
        else:
            embedding = text2emb[text]
            sample['embedding'] = embedding
            new_samples.append(sample)
    
    json.dump(text2emb, open(local_filepath, 'w'))

    return samples