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

def scan_data(feat_info, tokenizer, embed_type, maxlength_seq):
    doc_len = 0
    sen_len = 0
    for file_path in feat_info:
        with open(file_path, 'r') as f:
            sens = f.readlines()
            if len(sens) > doc_len:
                doc_len = len(sens)
                for sen in sens:
                    if 'Char' not in embed_type:
                        embed_sen = convert_examples_to_features_nonchar(tokenizer, sen, maxlength_seq)[0]
                    else:
                        embed_sen = convert_examples_to_features_char(tokenizer, sen, maxlength_seq)[0]
                    if len(embed_sen) > sen_len:
                        sen_len = len(embed_sen)          
    return doc_len, sen_len

class DatasetReader:

    def __init__(self, data, embed_type = 'BERT', device = torch.device("cpu"), task_type = None, maxlength_seq = 10):
        
        '''
        
        current supported embed_type ['BERT','BioBERT','CharBERT','BioCharBERT']
        
        '''

        assert embed_type in ['BERT','BioBERT','CharBERT','BioCharBERT'], 'please fill in correct embed_type'
        self.feat_info = data['x']
        self.label_list = data['y']
        self.embed_type = embed_type
        self.tokenizer, self.model = get_bert(self.embed_type)
        self.seq_len, self.word_len = scan_data(self.feat_info, self.tokenizer, self.embed_type, maxlength_seq)
        self.maxlength_seq = maxlength_seq
        self.model.to(device)
        self.device = device
        self.task_type = task_type
    
    def get_device(self):
        if self.use_gpu:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print('use GPU recource')
            else:
                device = torch.device("cpu")
                print('not find effcient GPU, use CPU recource')
        else:
            device = torch.device("cpu")
            print('use CPU recource')
        return device


    def get_data(self):
        if self.task_type is None:
            raise Exception('fill in correct task-type xxx from [\'binaryclass\', \'multiclass\', \'multilabel\', \'regression\']')
        target_x = []
        label_y = []
        for index in range(len(self.feat_info)):
            feat = []
            with open(self.feat_info[index], 'r') as f:
                sens = f.readlines()
                for idx, sen in enumerate(sens):
                    if 'Char' not in self.embed_type:
                        embed_sen_ids = convert_examples_to_features_nonchar(self.tokenizer, sen, self.maxlength_seq).to(self.device)
#                       print (len(embed_sen_ids), len(embed_sen_ids[0]))
                        encoder_opt, _ = self.model(embed_sen_ids.to(self.device), output_all_encoded_layers=False)
                        encoder_opt = encoder_opt.squeeze(0).detach().cpu().numpy()
                        encoder_opt = np.mean(encoder_opt, 0)
                        feat.append(encoder_opt)
                    else:
                        embed_sen_ids = convert_examples_to_features_char(self.tokenizer, sen, self.maxlength_seq).to(self.device)
                        encoder_opt, _ = self.model(embed_sen_ids.to(self.device))
                        encoder_opt = encoder_opt.squeeze(0).detach().cpu().numpy()
                        encoder_opt = np.mean(encoder_opt, 0)
                        feat.append(encoder_opt)
            feat = np.mean(np.stack(feat, 0), 0)
            label = self.label_list[index]
            target_x.append(feat)
            label_y.append(label)
        target_x = np.stack(target_x, 0)
        label_y = np.stack(label_y, 0)
        if self.task_type == 'multilabel':
            label_y = np.array(self.label_list)
        else:
            labels = []
            if self.task_type == 'multiclass':
                for rowlabel in self.label_list:
                    labels.append(np.argmax(np.array(rowlabel)))
                labels = np.array(labels)
            else:
                labels = np.array(self.label_list)
            label_y = labels.reshape(-1, 1)
        return {'X': target_x, 'Y': label_y}

#
#expdata_id = '2021.0105.text'
#
#cur_dataset = expdata_generator(expdata_id, root_dir='./')
#cur_dataset.load_exp_data()
#
#_dataset = DatasetReader(cur_dataset.valid)
#_dataset.get_data()

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

    

