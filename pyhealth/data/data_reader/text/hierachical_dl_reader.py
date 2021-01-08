# -*- coding: utf-8 -*-

# Author: Zhi Qiao <mingshan_ai@163.com>

# License: BSD 2 clause

import os.path
import pandas as pd
import numpy as np
import torch
import pickle

from pyhealth.data.expdata_generator import textdata as expdata_generator

import pytorch_pretrained_bert
from pytorch_pretrained_bert import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pyhealth.utils.characterbertmain.modeling.character_bert import CharacterBertModel
from pyhealth.utils.characterbertmain.utils.character_cnn import CharacterIndexer

def get_bert(embed_type):
    if embed_type == 'BioBERT':
        model_loc = './auxiliary/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/'
        tokenizer = BertTokenizer.from_pretrained(model_loc, do_lower_case=True)
        cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
        model = BertModel.from_pretrained(model_loc, cache_dir=cache_dir)
    elif embed_type == 'BERT':
        model_loc = './auxiliary/pretrained_bert_tf/bert_pretrain_output_all_notes_150000/'
        tokenizer = BertTokenizer.from_pretrained(model_loc, do_lower_case=True)
        cache_dir = os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
        model = BertModel.from_pretrained(model_loc, cache_dir=cache_dir)
    elif embed_type == 'CharBERT':
        model_loc = './auxiliary/pretrained_character_bert/general_character_bert/'
        model = CharacterBertModel.from_pretrained(model_loc)
        tokenizer = BertTokenizer.from_pretrained('./auxiliary/pretrained_bert_tf/bert_pretrain_output_all_notes_150000/')
    elif embed_type == 'BioCharBERT':
        model_loc = './auxiliary/pretrained_character_bert/medical_character_bert/'
        model = CharacterBertModel.from_pretrained(model_loc)
        tokenizer = BertTokenizer.from_pretrained('./auxiliary/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000/')
    return tokenizer, model

def convert_examples_to_features_nonchar(tokenizer, text, maxlength_seq):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens[:maxlength_seq] + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = torch.tensor([input_ids], dtype=torch.long)
    return input_ids

def convert_examples_to_features_char(tokenizer, text, maxlength_seq):
    tokens = tokenizer.basic_tokenizer.tokenize(text)
    tokens = [["[CLS]"] + tokens[:maxlength_seq] + ["[SEP]"]]
    indexer = CharacterIndexer()
    input_ids = indexer.as_padded_tensor(tokens)
    return input_ids

def scan_data(feat_info, tokenizer, embed_type):
    doc_len = 0
    sen_len = 0
    for file_path in feat_info:
        with open(file_path, 'r') as f:
            sens = f.readlines()
            if len(sens) > doc_len:
                doc_len = len(sens)
    return doc_len

class DatasetReader:

    def __init__(self, data, embed_type = 'BERT', maxlength_seq = 200):
        
        '''
        
        current supported embed_type ['BERT','BioBERT','CharBERT','BioCharBERT']
        
        '''

        assert embed_type in ['BERT','BioBERT','CharBERT','BioCharBERT'], 'please fill in correct embed_type'
        self.feat_info = data['x']
        self.label_list = data['y']
        self.embed_type = embed_type
        self.tokenizer, self.model = get_bert(self.embed_type)
        self.seq_len = scan_data(self.feat_info, self.tokenizer, self.embed_type)
        self.word_len = maxlength_seq + 2
        self.maxlength_seq = maxlength_seq
        print (self.seq_len, self.word_len)
        
    def __getitem__(self, index):
        feats = np.zeros([self.seq_len, self.word_len, 768])
        masks = np.zeros([self.seq_len, self.word_len])
        with open(self.feat_info[index], 'r') as f:
            sens = f.readlines()
            for idx, sen in enumerate(sens):
                if 'Char' not in self.embed_type:
                    embed_sen_ids = convert_examples_to_features_nonchar(self.tokenizer, sen, self.maxlength_seq)
                    encoder_opt, _ = self.model(embed_sen_ids, output_all_encoded_layers=False)
                    encoder_opt = encoder_opt.squeeze(0).detach().cpu().numpy()
                    feats[idx, :len(encoder_opt), :] = encoder_opt
                    masks[idx, :len(encoder_opt)] = 1
                else:
                    embed_sen_ids = convert_examples_to_features_char(self.tokenizer, sen, self.maxlength_seq)
                    encoder_opt, _ = self.model(embed_sen_ids)
                    encoder_opt = encoder_opt.squeeze(0).detach().cpu().numpy()
                    feats[idx, :len(encoder_opt), :] = encoder_opt
                    masks[idx, :len(encoder_opt)] = 1
        label = self.label_list[index]

        return {'X': np.array(feats), 'M': np.array(masks), 'Y': np.array(label)}

    def __len__(self):
        return len(self.feat_info)

#expdata_id = '2021.0102.text'
#
#cur_dataset = expdata_generator(expdata_id, root_dir='./')
#cur_dataset.load_exp_data()
#
#_dataset = DatasetReader(cur_dataset.valid)
#
#print (len(_dataset.label_list))
#
#dtype = 'train'
#_loader = torch.utils.data.DataLoader(_dataset,
#                                      batch_size=5,
#                                      drop_last=True,
#                                      shuffle=True if dtype == 'train' else False)
#for batch_idx, databatch in enumerate(_loader):
#    print (databatch['X'].shape)
#    print (databatch['M'].shape)
#    print (databatch['Y'].shape)
#    break

    
