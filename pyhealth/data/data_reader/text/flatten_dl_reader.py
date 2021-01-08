# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os.path
import pandas as pd
import numpy as np
import torch
import pickle

from pyhealth.data.expdata_generator import textdata as expdata_generator
from pyhealth.models.text.tool import get_embedding


def convert_examples_to_features_nonchar(tokenizer, text, maxlength_seq):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens[:maxlength_seq - 2] + ["[SEP]"]
    value_len = len(tokens)
    pad_len = maxlength_seq - value_len if value_len < maxlength_seq else 0
    tokens = tokens + pad_len * ["[PAD]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    return value_len, input_ids

def convert_examples_to_features_char(tokenizer, indexer, text, maxlength_seq):
    tokens = tokenizer.basic_tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens[:maxlength_seq - 2] + ["[SEP]"]
    value_len = len(tokens)
    pad_len = maxlength_seq - value_len if value_len < maxlength_seq else 0
    tokens = [tokens + pad_len * ["[PAD]"]]
    input_ids = indexer.as_padded_tensor(tokens)
    return value_len, input_ids

class DatasetReader:

    def __init__(self, data, embed_type = 'BERT', device = torch.device("cpu"), maxlength_seq = 200):
        
        '''
        
        current supported embed_type ['BERT','BioBERT','CharBERT','BioCharBERT']
        
        '''

        assert embed_type in ['BERT','BioBERT','CharBERT','BioCharBERT'], 'please fill in correct embed_type'
        self.feat_info = data['x']
        self.label_list = data['y']
        self.embed_type = embed_type
        self.indexer, self.tokenizer, self.model = get_embedding(self.embed_type)
        self.model.to(device)
        self.device = device
        self.maxlength_seq = maxlength_seq

    def __getitem__(self, index):
        feats = np.zeros([self.maxlength_seq, 768])
        masks = np.zeros([self.maxlength_seq])
        cur_masks = np.zeros([self.maxlength_seq])
        with open(self.feat_info[index], 'r') as f:
            sens = f.readlines()
            real_f = []
            doc = ''
            for idx, sen in enumerate(sens):
                doc += sen + ' '
            if 'Char' not in self.embed_type:
                value_len, embed_sen_ids = convert_examples_to_features_nonchar(self.tokenizer, doc, self.maxlength_seq)
                encoder_opt, _ = self.model(embed_sen_ids.to(self.device), output_all_encoded_layers=False)
                encoder_opt = encoder_opt.squeeze(0).detach().cpu().numpy()
            else:
                value_len, embed_sen_ids = convert_examples_to_features_char(self.tokenizer, self.indexer, doc, self.maxlength_seq)
                encoder_opt, _ = self.model(embed_sen_ids.to(self.device))
                encoder_opt = encoder_opt.squeeze(0).detach().cpu().numpy()
            
            feats[:len(encoder_opt), :] = encoder_opt
            masks[:value_len] = 1
            cur_masks[value_len-1] = 1
        label = self.label_list[index]

        return {'X': np.array(feats), 'M': np.array(masks), 'cur_M': np.array(cur_masks), 'Y': np.array(label)}

    def __len__(self):
        return len(self.feat_info)

#expdata_id = '2021.0105.text'
#
#cur_dataset = expdata_generator(expdata_id, root_dir='./')
#cur_dataset.load_exp_data()
#
#device = torch.device("cuda")
#
#_dataset = DatasetReader(cur_dataset.valid, embed_type = 'CharBERT', device = device, maxlength_seq = 100)
#
#print (len(_dataset.label_list))
#
#dtype = 'train'
#
#_loader = torch.utils.data.DataLoader(_dataset,
#                                      batch_size=2,
#                                      drop_last=True,
#                                      shuffle=True if dtype == 'train' else False)
#
#for batch_idx, databatch in enumerate(_loader):
#    print (databatch['X'].shape)
#    print (databatch['M'].shape)
#    print (databatch['cur_M'].shape)
#    print (databatch['Y'].shape)
#    break

    

