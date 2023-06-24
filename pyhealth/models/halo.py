'''
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
'''
from collections import defaultdict
from datetime import timedelta
import itertools
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import copy
import math
from typing import Any, Callable, Dict, List, Tuple, Type, Union
import pandas
from pyhealth.utils import print_dict
from sklearn.metrics import r2_score
from tqdm import tqdm
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

from pyhealth import datasets
from pyhealth.data import Event 
from pyhealth.datasets.base_ehr_dataset import BaseEHRDataset
from pyhealth.datasets.eicu import eICUDataset



"""
model configuration

required fields are non-optional for instantiating the HALO model 
"""
class Config(object):
    required = [
        "n_positions", "n_ctx", "n_embd", "n_layer", "n_head", "layer_norm_epsilon", "initializer_range"
    ]
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        
        missing = []
        for r in self.required:
            if (r not in self.__dict__):
                missing.append(r)
        if missing != []:
            raise Exception(f"Incorrect HALO config provided. Required fields missing: {', '.join(missing)}")

"""
model definition & building blocks
"""
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        assert n_state % config.n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = nn.Softmax(dim=-1)(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = gelu

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, config, scale=False):
        super(Block, self).__init__()
        nx = config.n_embd
        self.ln_1 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.attn = Attention(nx, n_ctx, config, scale)
        self.ln_2 = LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class CoarseTransformerModel(nn.Module):
    def __init__(self, config):
        super(CoarseTransformerModel, self).__init__()
        self.n_layer = config.n_layer
        self.n_embd = config.n_embd
        self.n_vocab = config.total_vocab_size

        self.vis_embed_mat = nn.Linear(config.total_vocab_size, config.n_embd, bias=False)
        self.pos_embed_mat = nn.Embedding(config.n_positions, config.n_embd)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_visits, position_ids=None, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_visits.size(1) + past_length, dtype=torch.long,
                                        device=input_visits.device)
            position_ids = position_ids.unsqueeze(0).expand(input_visits.size(0), input_visits.size(1))

        inputs_embeds = self.vis_embed_mat(input_visits)
        position_embeds = self.pos_embed_mat(position_ids)
        hidden_states = inputs_embeds + position_embeds
        for block, layer_past in zip(self.h, past):
            hidden_states, _ = block(hidden_states, layer_past)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states

class AutoregressiveLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.tril(torch.ones(in_features, out_features)).int())
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class FineAutoregressiveHead(nn.Module):
    def __init__(self, config):
        super(FineAutoregressiveHead, self).__init__()
        self.auto1 = AutoregressiveLinear(config.n_embd + config.total_vocab_size, config.n_embd + config.total_vocab_size)
        self.auto2 = AutoregressiveLinear(config.n_embd + config.total_vocab_size, config.n_embd + config.total_vocab_size)
        self.n_embd = config.n_embd
        self.tot_vocab = config.total_vocab_size

    def forward(self, history, input_visits):
        history = history[:,:-1,:]
        input_visits = input_visits[:,1:,:]
        code_logits = self.auto2(torch.relu(self.auto1(torch.cat((history, input_visits), dim=2))))[:,:,self.n_embd-1:-1]
        return code_logits

    def sample(self, history, input_visits):
        history = history[:,:-1,:]
        input_visits = input_visits[:,1:,:]
        currVisit = torch.cat((history, input_visits), dim=2)[:,-1,:].unsqueeze(1)
        code_logits = self.auto2(torch.relu(self.auto1(currVisit)))[:,:,self.n_embd-1:-1]
        return code_logits

class HALOModel(nn.Module):
    def __init__(self, 
            n_ctx,
            total_vocab_size,
            device,
            config: Config = None
        ):
        super(HALOModel, self).__init__()
        
        if config == None:
            config = Config(
                # user defined
                n_ctx=n_ctx,
                total_vocab_size=total_vocab_size,
                device=device,

                # defaults provided by HALO implementors
                n_positions=20,
                n_embd=768,
                n_layer=12,
                n_head=12,
                layer_norm_epsilon=1e-5, 
                initializer_range=0.02,
            )

        self.transformer = CoarseTransformerModel(config)
        self.ehr_head = FineAutoregressiveHead(config)

    def forward(self, input_visits, position_ids=None, ehr_labels=None, ehr_masks=None, past=None):
        hidden_states = self.transformer(input_visits, position_ids, past)
        code_logits = self.ehr_head(hidden_states, input_visits)
        sig = nn.Sigmoid()
        code_probs = sig(code_logits)
        if ehr_labels is not None:    
            bce = nn.BCELoss()
            shift_labels = ehr_labels[..., 1:, :].contiguous()
            if ehr_masks is not None:
                code_probs = code_probs * ehr_masks
                shift_labels = shift_labels * ehr_masks

            loss = bce(code_probs, shift_labels)
            return loss, code_probs, shift_labels
        
        return code_probs

    def sample(self, input_visits, random=True):
        sig = nn.Sigmoid()
        hidden_states = self.transformer(input_visits)
        i = 0
        while i < self.ehr_head.tot_vocab:
            next_logits = self.ehr_head.sample(hidden_states, input_visits)
            next_probs = sig(next_logits)
            if random:
                visit = torch.bernoulli(next_probs)
            else:
                visit = torch.round(next_probs)
            
            remaining_visit = visit[:,0,i:]
            nonzero = torch.nonzero(remaining_visit, as_tuple=True)[1]
            if nonzero.numel() == 0:
                break

            first_nonzero = nonzero.min()
            input_visits[:,-1,i + first_nonzero] = visit[:,0,i + first_nonzero]
            i = i + first_nonzero + 1
        
        return input_visits
    
"""
produce a discrete value given a continuous one
strict_range: when a value is outside of the range, do you throw error, or do you just make the value the range maximum? Similar for minimum. 
"""
class Discretizer:
    
    def __init__(self, strategy: str, strict_range: bool = False, **kwargs):
        if strategy == 'uniform':
            self.start = kwargs['start']
            self.end = kwargs['end']
            step = kwargs['step']
            
            intervals = [self.start]
            while (intervals[-1] < self.end):
                # the last interval may be shorter than all the others, but never exceed range
                intervals.append(min(intervals[-1] + step, self.end))
            
            self.clip_overflow = strict_range
            self.intervals = intervals
            self.to_discrete = self.uniform
        else: 
            # not implemented or discretization strategy is not a correct value
            raise Exception('Invalid discretization strategy provided')
        
    # currently unimplemented
    # todo: bdanek, would be good to write this algorithm without any bugs... & with unit tests    
    def uniform(self, value: float):
        if self.strict_range:
            assert value <= self.end or value >= self.start, f'Strict range enforced; value: {value} not in range [{self.start}, {self.end}].'
        
        return value

"""
preprocessor

- metadata level labels
    - available on the Patient
- phenotype level labels
    - present in visit set

"""
class HALOProcessor():
    """
    Class used to process a PyHealth dataset so that it can be used in HALO. 

    Args:
        dataset: Dataset to process.
        use_tables: Tables to use during processing. If none, all available tables in the dataset are used. 
        
        event_handlers: A dictionary of handlers for unpacking, or accessing fields of a pyhealth.data.Event. 
            The dict key should be a table name, value is a Callable which must accept a pyhealth.data.Event to unpack.
        continuous_value_handlers: A dictionary of handlers for converting an event from a continuous value to a bucketed/categorical one. 
            This handler is applied after the event handler. Dict key is table name, and should return an integer representing which bucket the value falls in.
        continuous_value_handlers_inverter: An optional dictionary of handlers for inverting the operation performed in `continuous_value_handlers`. 
            The dict key is a table name, and the value is a Callable which accepts a vector (list of integers)
        time_handler: 
            A function which converts a timedelta into a multihot vector representation. 
        time_hanlder_inverter: 
            An optional function which converts the multihot time vector represention into a human readbile value. 
        time_vector_length: 
            The integer representing the length of the multihot time vector produced by `time_handler`
        max_visits: 
            The maximum visits to use for modeling. If not provided, the maximum number of visits present in the source dataset is used.
        label_fn: 
            A function which accepts the keyword argument `patient_data: pyhealth.data.Patient` and produces a vector representation of the patient label.
        label_vector_len: 
            The length of a patient label vector.
        invert_label: 
            An optional function for converting the label vector into a human readable patient label

    """
    
    # visit dimension (dim 1)
    SPECIAL_VOCAB = ('start_code', 'last_visit_code', 'pad_code') 
    START_INDEX = 0
    LABEL_INDEX = 1
    VISIT_INDEX = 2

    # code dimension (dim 2)
    SPECIAL_VISITS = ('start_visit', 'label_visit')

    # the key for the inter_visit_gap handler
    TEMPORAL_INTER_VISIT_GAP = 'inter_visit_gap'
        
    def __init__(
        self,
        dataset: BaseEHRDataset,
        use_tables: List[str],
        
        # allows unpacking/handling patient records into events
        event_handlers: Dict[str, Callable[[Type[Event]], Any]] = {},

        # used to handle continuous values
        continuous_value_handlers: Dict[str, Callable[..., int]] = {},
        continuous_value_handlers_inverter: Dict[str, Callable[[List[int]], Any]] = {},

        # used to discretize time
        time_handler: Callable[[Type[timedelta]], Any] = None,
        time_hanlder_inverter: Callable[[List[int]], Any] = None,
        time_vector_length: int = -1,
        
        max_visits: Union[None, int] = None,
        label_fn: Callable[..., List[int]] = None, 
        label_vector_len: int = -1,
        invert_label: Callable[..., Any] = None,
    ) -> None:
        
        self.dataset = dataset
        
        # whitelisted tables
        self.valid_dataset_tables = use_tables 

        # handle processing of event types
        self.event_handlers = event_handlers 
        
        # generate a HALO label based on a patient record
        assert label_fn != None, "Define the label_fn."
        assert label_vector_len >= 0, "Nonnegative vector_len required. May be due to user error, or value is not defined."
        self.label_fn = label_fn
        self.label_vector_len = label_vector_len
        self.invert_label = invert_label

        self.continuous_value_handlers = continuous_value_handlers
        self.continuous_value_handlers_inverter = continuous_value_handlers_inverter

        assert time_handler != None, "Defining time_handler is not optional. This field converts time values to a discrete one hot/multi hot vector representation."
        self.time_handler = time_handler

        # optional
        self.time_hanlder_inverter = time_hanlder_inverter
        
        assert time_vector_length != None, "Defining time_vector_length is not optional. This field is equivalent to the number of buckets required to discretie time."
        self.time_vector_length = time_vector_length

        self.max_visits = max_visits

        # init the indeces & dynamically computed utility variables used in HALO training later
        self.set_indeces()


    def set_indeces(self) -> None:
        # set aggregate indeces
        self.global_events: Dict = {}
        self.aggregate_event_indeces()

        # assert the processor works as expected
        assert len(self.global_events) % 2 == 0, "Event index processor not bijective"

        # bidirectional mappings
        self.num_global_events = self.time_vector_length + len(self.global_events) // 2

        # define the tokens in the event dimension (visit dimension already specified)
        self.label_start_index = self.num_global_events
        self.label_end_index = self.num_global_events + self.label_vector_len
        self.start_token_index = self.num_global_events + self.label_vector_len
        self.end_token_index = self.num_global_events + self.label_vector_len + 1
        self.pad_token_index = self.num_global_events + self.label_vector_len + 2

        # parameters for generating batch vectors
        self.total_vocab_size = self.num_global_events + self.label_vector_len + len(self.SPECIAL_VOCAB)
        self.total_visit_size = len(self.SPECIAL_VISITS) + self.max_visits

    """
    its necessary to aggregate global event data, prior to transforming the dataset
    """
    def aggregate_event_indeces(self) -> None:

        # two way mapping from global identifier to index & vice-versa
        # possible since index <> global identifier is bijective
        # type: ((table_name: str, event_value: any): index) or (index: (table_name: str, event_value: any))
        max_visits: int = 0
        min_birth_datetime = pandas.Timestamp.now()
        
        for pdata in tqdm(list(self.dataset), desc="HALOAggregator generating indeces"):
            
            max_visits = max(max_visits, len(pdata.visits))
            if pdata.birth_datetime != None:
                min_birth_datetime = min(min_birth_datetime, pdata.birth_datetime)

            # compute global event
            for vid, vdata in pdata.visits.items():

                for table in vdata.available_tables:

                    # valid_tables == None signals we want to use all tables
                    # otherwise, omit any table not in the whitelist
                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    
                    for te_raw in table_events_raw:
                        
                        te = event_handler(te_raw) if event_handler else te_raw.code

                        if table in self.continuous_value_handlers:
                            te = self.continuous_value_handlers[table](te)

                        global_event = (table, te)
                    
                        if global_event not in self.global_events:
                            # keys 0 - self.time_vector_length are reserved for inter-visit information
                            index_of_global_event = self.time_vector_length + (len(self.global_events) // 2)
                            self.global_events[global_event] = index_of_global_event
                            self.global_events[index_of_global_event] = global_event

        
        # if the user does not provide, infer from dataset
        if self.max_visits == None:
            self.max_visits = max_visits
        
        self.min_birth_datetime = min_birth_datetime
    """
    similar to dataset.set_task(...)
    - produce a sampleEHRDataset
    """  
    def process_batch(self, batch) -> Tuple:
        batch_size = len(batch)
        
        # dim 0: batch
        # dim 1: visit vectors
        # dim 2: concat(event multihot, label onehot, metadata)
        sample_multi_hot = np.zeros((batch_size, self.total_visit_size, self.total_vocab_size)) # patient data the model reads
        sample_mask = np.zeros((batch_size, self.total_visit_size, 1)) # visits that are unlabeled
        
        for pidx, pdata in enumerate(batch):

            previous_time = pdata.birth_datetime if pdata.birth_datetime != None else self.min_birth_datetime
            # build multihot vector for patient events
            for visit_index, vid,  in enumerate(pdata.visits):
                
                vdata = pdata.visits[vid]

                # set temporal attributes
                current_time = vdata.encounter_time
                time_since_last_visit = current_time - previous_time                
                
                # vector representation of the gap between last visit and current one
                inter_visit_gap_vector = self.time_handler(time_since_last_visit)
                sample_multi_hot[pidx, self.VISIT_INDEX, :self.time_vector_length] = inter_visit_gap_vector

                # the next timedelta is previous current visit - discharge of previous visit
                previous_time = vdata.discharge_time

                sample_mask[pidx, self.VISIT_INDEX + visit_index] = 1
                
                for table in vdata.available_tables:

                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    continuous_value_handler = self.continuous_value_handlers[table] if table in self.continuous_value_handlers else None

                    for te_raw in table_events_raw:
                        
                        te = event_handler(te_raw) if event_handler else te_raw.code
                        
                        if continuous_value_handler:
                            te = continuous_value_handler(te)

                        global_event = (table, te)
                        event_as_index = self.global_events[global_event]
                        
                        # set table events
                        sample_multi_hot[pidx, self.VISIT_INDEX + visit_index, event_as_index] = 1            
            
            # set patient label
            global_label_vector = self.label_fn(patient_data=pdata)
            sample_multi_hot[pidx, self.LABEL_INDEX, self.num_global_events: self.num_global_events + self.label_vector_len] = global_label_vector
            
            # set the end token
            sample_multi_hot[pidx, self.VISIT_INDEX + (len(pdata.visits) - 1), self.end_token_index] = 1

            # set the remainder of the visits to pads if needed
            sample_multi_hot[pidx, (self.VISIT_INDEX + (len(pdata.visits) - 1)) + 1:, self.pad_token_index] = 1
            
        # set the start token
        sample_multi_hot[:, self.START_INDEX, self.start_token_index] = 1

        # set the mask to include the labels
        sample_mask[:, self.LABEL_INDEX] = 1
        
        # "shift the mask to match the shifted labels & predictions the model will return"
        sample_mask = sample_mask[:, 1:, :]
            
        res = (sample_multi_hot, sample_mask)
        
        return res
    
    def get_batch(self, data_subset: BaseEHRDataset, batch_size: int = 16,):

        batch_size = min(len(data_subset), batch_size)

        batch_offset = 0
        while (batch_offset + batch_size <= len(data_subset)):
            
            batch = data_subset[batch_offset: batch_offset + batch_size]
            batch_offset += batch_size # prepare for next iteration
            
            yield self.process_batch(batch)

"""
trainer for generative models
"""
class HALOTrainer:
    def __init__(self, 
            dataset: datasets, 
            model: nn.Module,
            processor: HALOProcessor, 
            optimizer: Optimizer,
            checkpoint_path: str,
        ) -> None:
        self.dataset = dataset
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.checkpoint_path = checkpoint_path
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
    
    """
    Set class variables of a 0.8, 0.1, 0.1 split for train, test, eval sets respectivley.
    returns the splits for convenience
    """
    def set_basic_splits(self):
        train, test, eval = self.split()
        self.train_dataset = train
        self.test_dataset = test
        self.eval_dataset = eval
        
        return self.train_dataset, self.test_dataset, self.eval_dataset
        
    def split(self, splits: List[float] = [0.8, 0.1, 0.1], shuffle: bool = False):
        if shuffle:
            self.dataset = random.random.shuffle(self.dataset)
            
        if sum(splits) != 1:
            raise Exception(f"splits don't sum to the full dataset. sum(splits) = {sum(splits)}")
        
        n = len(self.dataset.patients)
        dataset_splits = []
        start_offset = 0
        for s in splits:
            n_split = math.ceil(n * s) # size of the current split
            
            # the last subset will be the smallest
            subset = self.dataset[start_offset: min(start_offset + n_split, n)]
           
            dataset_splits.append(subset)
            start_offset += n_split
            
        return dataset_splits
    
    def make_checkpoint(self, iteration):
        state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': iteration
            }
        torch.save(state, f"{self.checkpoint_path}.pkl")
        print('\n------------ Save best model ------------\n')

    def eval(self, batch_size: int, current_epoch: int = 0, current_iteration: int = 0, patience: int = 0, save=True):
        self.model.eval()
        
        with torch.no_grad():
            
            global_loss = 1e10
            val_l = []
            current_patience = 0
            for batch_ehr, batch_mask in self.processor.get_batch(self.eval_dataset, batch_size):
                
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)

                val_loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                val_l.append((val_loss).cpu().detach().numpy())
                
                cur_val_loss = np.mean(val_l)
                if current_epoch:
                    print("Epoch %d Validation Loss:%.7f"%(current_epoch, cur_val_loss))
                else:
                    print("Validation Loss:%.7f"%(cur_val_loss))

                # make checkpoint
                if save and cur_val_loss < global_loss:
                    global_loss = cur_val_loss
                    patience = 0
                    self.make_checkpoint(iteration=current_iteration)
                
                current_patience += 1
                if current_patience == patience: break
    
    def validate(self, batch_size):
        self.eval(batch_size=batch_size, current_epoch=0, current_iteration=0, patience=None)

    def train(self, batch_size: int, epoch: int, patience: int, eval_period: int) -> None:        
        
        for e in tqdm(range(epoch), desc="Training HALO model"):
            
            self.model.train()
            
            for i, (batch_ehr, batch_mask) in enumerate(self.processor.get_batch(self.train_dataset, batch_size)):
                
                batch_ehr = torch.tensor(batch_ehr, dtype=torch.float32).to(self.device)
                batch_mask = torch.tensor(batch_mask, dtype=torch.float32).to(self.device)
                
                self.optimizer.zero_grad()
                
                loss, _, _ = self.model(batch_ehr, position_ids=None, ehr_labels=batch_ehr, ehr_masks=batch_mask)
                
                loss.backward()
                self.optimizer.step()
                
                # the eval period may never be reached if there aren't enough batches
                if i % min(eval_period, len(self.train_dataset)//batch_size) == 0:
                    print("Epoch %d, Iter %d: Training Loss:%.7f"%(e, i, loss))
                    self.eval(current_epoch=e, current_iteration=i, batch_size=batch_size)

        self.eval(batch_size=batch_size, current_epoch=epoch, current_iteration=-1, patience=patience)

class HALOGenerator:

    VISITS = 'visits'
    TIME = 'inter-visit_gap'
    LABEL = 'label'

    def __init__(
            self,
            model: nn.Module,
            processor: HALOProcessor,
            batch_size: int, # it is recommended to use the same batch size as that for training
            save_path: str,
            device: str,
        ) -> None:
        
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.save_path = save_path
        self.device = device

    # generate context vector, and the probablility of the label occurrence in the dataset
    def generate_context(self, label_vector) -> List:
        stoken = np.zeros((1, processor.total_vocab_size))
        stoken[0, processor.start_token_index] = 1
        
        if label_vector is None:
            return stoken # probability of label occurrence in dataset
        
        ltoken = np.zeros((1, processor.total_vocab_size))
        ltoken[0, self.processor.label_start_index: self.processor.label_end_index] = label_vector

        context = np.concatenate((stoken, ltoken), axis=0)
        context = context[:, np.newaxis, :]
        return context

    # get batches of context vectors with a probability
    def get_contexts(self, contexts, batch_size: int, probability: float):
        idx = np.random.choice(len(contexts), batch_size, replace = True, p = probability) # random selection to generate contexts*batch_size seems inefficient
        return np.array([contexts[i] for i in idx])

    def sample_sequence(self, context, batch_size, sample=True, visit_type=-1):
        empty = torch.zeros((1,1,processor.total_vocab_size), device=self.device, dtype=torch.float32).repeat(batch_size, 1, 1)
        prev = torch.tensor(context, device=self.device, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(self.processor.max_visits - (len(['start_token', 'label_token']))): # visits - (start vector, label vector); iterate # of ti

                prev = self.model.sample(torch.cat((prev,empty), dim=1), sample)

                if torch.sum(torch.sum(prev[:, :, processor.end_token_index], dim=1).bool().int(), dim=0).item() == batch_size: # why do we do this?
                    break

        samples = prev.cpu().detach().numpy()

        return samples


    # handle conversion from HALO vector output to samples
    def convert_samples_to_ehr(self, samples) -> List[Dict]:
        ehr_outputs = []
        for i in range(len(samples)):
            sample_as_ehr = []
            sample_time_gaps = []
            sample = samples[i]

            # labels need to be hashable, so we convert them back to tuple representation
            labels_output = tuple(sample[self.processor.LABEL_INDEX][self.processor.label_start_index: self.processor.label_end_index])
            parsed_labels_output = self.processor.invert_label(labels_output) if self.processor.invert_label != None else labels_output

            for j in range(self.processor.VISIT_INDEX, len(sample)):
                
                visit = sample[j]

                # handle inter-visit gaps
                visit_time = visit[:self.processor.time_vector_length]
                convert_to_time = self.processor.time_hanlder_inverter
                time_gap = convert_to_time(visit_time) if convert_to_time != None else visit_time
                sample_time_gaps.append(time_gap)

                # handle visit event codes
                visit_events = visit[self.processor.time_vector_length: self.processor.num_global_events]
                visit_code_indices = np.nonzero(visit_events)[0]
                visit_ehr_codes = [self.processor.global_events[self.processor.time_vector_length + index] for index in visit_code_indices]
                sample_as_ehr.append(visit_ehr_codes)

                end = bool(sample[j, self.processor.end_token_index])
                if end: break
            
            ehr_outputs.append({self.VISITS: sample_as_ehr, self.TIME: sample_time_gaps, self.LABEL: parsed_labels_output})

        return ehr_outputs

    def generate_conditioned(self, labels: List[Tuple[any, int]]):
        synthetic_ehr_dataset = []
        for (label, count_per_label) in tqdm(labels, desc=f"Generating samples for labels"):
            context_vectors = self.generate_context(label)
            for i in tqdm(range(0, count_per_label, self.batch_size), leave=False):
                amount_remaining = count_per_label - i
                bs = min(amount_remaining, self.batch_size)
                context = self.get_contexts(context_vectors, bs, probability=None)
                
                batch_synthetic_ehrs = self.sample_sequence(
                    context=context, 
                    batch_size=bs, 
                    sample=True
                )
                
                batch_synthetic_ehrs = self.convert_samples_to_ehr(batch_synthetic_ehrs)
                synthetic_ehr_dataset += batch_synthetic_ehrs
        print("Saving synthetic ehr dataset at:", self.save_path)
        pickle.dump(synthetic_ehr_dataset, open(self.save_path, "wb"))
        return synthetic_ehr_dataset

    def generate_unconditioned(self):    
        pass
    
    def evaluate(self):
        pass

class HALOEvaluator:

    # used to access the output of the evaluate(...) function
    SOURCE_STATS = "source_stats"
    SYNTHETIC_STATS = "synthetic_stats"
    PLOT_PATHS = "plot_paths"

    RECORD_LEN_MEAN = "Record Length Mean"
    RECORD_LEN_STD = "Record Length Standard Deviation"
    VISIT_LEN_MEAN = "Visit Length Mean"
    VISIT_LEN_STD = "Visit Length Standard Deviation"
    TEMPORAL_MEAN = "Inter-visit time Mean"
    TEMPORAL_STD = "Inter-visit time Standard Deviation"
    AGGREGATE = "Aggregate"

    RECORD_CODE_PROB = "Per Record Code Probabilities"
    VISIT_CODE_PROB = "Per Visit Code Probabilities"
    RECORD_BIGRAM_PROB = "Per Record Bigram Probabilities"
    VISIT_BIGRAM_PROB = "Per Visit Bigram Probabilities"
    RECORD_SEQUENTIAL = "Per Record Sequential Visit Bigram Probabilities"
    VISIT_SEQUENTIAL = "Per Visit Sequential Visit Bigram Probabilities"
    PROBABILITIES = "Probabilities"
    LABEL_PROBABILITIES = "Label Probabilities"

    PROBABILITY_DENSITIES = [
        RECORD_CODE_PROB,
        VISIT_CODE_PROB,
        RECORD_BIGRAM_PROB,
        VISIT_BIGRAM_PROB,
        RECORD_SEQUENTIAL,
        VISIT_SEQUENTIAL
    ]

    def __init__(
            self,
            generator: HALOGenerator,
            processor: HALOProcessor,
        ):
        self.generator = generator
        self.processor = processor

        # all ones, 1 index longer than any other label
        self.ALL_LABELS = tuple(np.ones(self.processor.label_vector_len + 1))
    
    def default_path_fn(self, plot_type, label_vector):
        label_string = str(tuple(label_vector))
        path = f"./pyhealth_halo_eval_{plot_type}_{label_string}"
        return path.replace('.', '').replace(' ', '').lower()

    def evaluate(self, source, synthetic, compare_label: List = None, get_plot_path_fn: Callable = None, print_overall: bool = True):
        halo_labels, halo_ehr_stats = self.generate_statistics(ehr_dataset=synthetic)
        
        source_as_vectors = self.to_evaluation_format(source)
        train_erh_labels, train_ehr_stats = self.generate_statistics(ehr_dataset=source_as_vectors)
        
        assert halo_labels, "No labels present in HALO Dataset, this is likely because the dataset schema is incorrect."
        assert train_erh_labels, "No labels present in Training Dataset, this is likely because the dataset schema is incorrect."

        if print_overall:
            print("source (train)")
            print_dict(train_ehr_stats[self.ALL_LABELS][self.AGGREGATE])
            print_dict(train_ehr_stats[self.LABEL_PROBABILITIES])
            print("synthetic")
            print_dict(halo_ehr_stats[self.ALL_LABELS][self.AGGREGATE])
            print_dict(halo_ehr_stats[self.LABEL_PROBABILITIES])

        # Plot per-code statistics
        plot_paths = self.generate_plots(train_ehr_stats, halo_ehr_stats, "Source Data", "Synthetic Data", get_plot_path_fn=get_plot_path_fn, compare_labels=compare_label)

        return {self.SOURCE_STATS: train_ehr_stats, self.SYNTHETIC_STATS: halo_ehr_stats, self.PLOT_PATHS: plot_paths}
    
    def to_evaluation_format(self, dataset: BaseEHRDataset) -> List[Dict]:
        """
        computing probability densities is more straighforward on a vector dataset.
        This method processes a pyhealth dataset into the HALO vector format
        """
        
        converted_samples = []
        for batch_ehr, _ in self.processor.get_batch(dataset, batch_size):
            converted_sample_batch = self.generator.convert_samples_to_ehr(samples=batch_ehr)    
            converted_samples += (converted_sample_batch)
        
        return converted_samples

    def generate_statistics(self, ehr_dataset) -> Dict:
        """Compute basic statistics and probability densities of code occurrences (unigram, bigram, sequential bigram)"""
        
        # compute all available lables in the dataset
        labels = set()
        for sample in ehr_dataset: labels.add(sample[self.generator.LABEL])

        # used in plot generation later
        dataset_labels = tuple(labels)

        # generate overall statistics
        labels.add(self.ALL_LABELS)

        # collect stats for the current label
        stats = {}
        label_counts = {}
        for label in sorted(list(labels)):
            
            # select the current subset to generate stats for
            ehr_subset = []
            if label != self.ALL_LABELS:
                for sample in ehr_dataset:
                    if sample[self.generator.LABEL] == label:
                        ehr_subset.append(sample)
            else:
                ehr_subset = ehr_dataset

            # compute stats per label
            label_subset = ehr_dataset
            label_counts[label] = len(label_subset)

            label_stats = {}

            # compute aggregate stats
            record_lens = []
            visit_lens = []
            visit_gaps = []
            for sample in label_subset:
                visits = sample[self.generator.VISITS]
                timegap = sample[self.generator.TIME]
                record_lens.append(len(visits))
                visit_lens += [len(v) for v in visits]
                visit_gaps.append(timegap)

            aggregate_stats = {}
            aggregate_stats[self.RECORD_LEN_MEAN] = np.mean(record_lens)
            aggregate_stats[self.RECORD_LEN_STD] = np.std(record_lens)
            aggregate_stats[self.VISIT_LEN_MEAN] = np.mean(visit_lens)
            aggregate_stats[self.VISIT_LEN_STD] = np.std(visit_lens)
            aggregate_stats[self.TEMPORAL_MEAN] = np.mean(visit_lens)
            aggregate_stats[self.TEMPORAL_STD] = np.std(visit_lens)
            label_stats[self.AGGREGATE] = aggregate_stats

            # compute probability densities
            code_stats = {}
            n_records = len(record_lens)
            n_visits = len(visit_lens)
            record_code_counts = {}
            visit_code_counts = {}
            record_bigram_counts = {}
            visit_bigram_counts = {}
            record_sequential_bigram_counts = {}
            visit_sequential_bigram_counts = {}
            for row in label_subset:
                patient_codes = set()
                patient_bigrams = set()
                sequential_bigrams = set()
                for j, visit in enumerate(row[self.generator.VISITS]):
                    v = list(set(visit)) # remove duplicates
                    for c in v:
                        visit_code_counts[c] = 1 if c not in visit_code_counts else visit_code_counts[c] + 1
                        patient_codes.add(c)
                    for cs in itertools.combinations(v,2):
                        cs = list(cs)
                        cs.sort()
                        cs = tuple(cs)
                        visit_bigram_counts[cs] = 1 if cs not in visit_bigram_counts else visit_bigram_counts[cs] + 1
                        patient_bigrams.add(cs)
                    if j > 0:
                        v0 = list(set(row[self.generator.VISITS][j - 1]))
                        for c0 in v0:
                            for c in v:
                                sc = (c0, c)
                                visit_sequential_bigram_counts[sc] = 1 if sc not in visit_sequential_bigram_counts else visit_sequential_bigram_counts[sc] + 1
                                sequential_bigrams.add(sc)
                for c in patient_codes:
                    record_code_counts[c] = 1 if c not in record_code_counts else record_code_counts[c] + 1
                for cs in patient_bigrams:
                    record_bigram_counts[cs] = 1 if cs not in record_bigram_counts else record_bigram_counts[cs] + 1
                for sc in sequential_bigrams:
                    record_sequential_bigram_counts[sc] = 1 if sc not in record_sequential_bigram_counts else record_sequential_bigram_counts[sc] + 1
            record_code_probs = {c: record_code_counts[c]/n_records for c in record_code_counts}
            visit_code_probs = {c: visit_code_counts[c]/n_visits for c in visit_code_counts}
            record_bigram_probs = {cs: record_bigram_counts[cs]/n_records for cs in record_bigram_counts}
            visit_bigram_probs = {cs: visit_bigram_counts[cs]/n_visits for cs in visit_bigram_counts}
            record_sequential_bigram_probs = {sc: record_sequential_bigram_counts[sc]/n_records for sc in record_sequential_bigram_counts}
            visit_sequential_bigram_probs = {sc: visit_sequential_bigram_counts[sc]/(n_visits - len(label_subset)) for sc in visit_sequential_bigram_counts}
            
            code_stats[self.RECORD_CODE_PROB] = record_code_probs
            code_stats[self.VISIT_CODE_PROB] = visit_code_probs
            code_stats[self.RECORD_BIGRAM_PROB] = record_bigram_probs
            code_stats[self.VISIT_BIGRAM_PROB] = visit_bigram_probs
            code_stats[self.RECORD_SEQUENTIAL] = record_sequential_bigram_probs
            code_stats[self.VISIT_SEQUENTIAL] = visit_sequential_bigram_probs
            
            label_stats[self.PROBABILITIES] = code_stats
            stats[label] = label_stats
        label_probs = {l: label_counts[l]/n_records for l in label_counts}
        
        stats[self.LABEL_PROBABILITIES] = label_probs
        
        return dataset_labels, stats
    
    def generate_plots(self, stats_a, stats_b, plot_label_a, plot_label_b, get_plot_path_fn: Callable = None, compare_labels: List = None) -> List[str]:
        """Generate plots"""
        if compare_labels == None:
            compare_labels = [self.ALL_LABELS]

        plot_paths = []
        for label in tqdm(compare_labels, desc="Evalutor: generating label plots"):
            data1 = stats_a[label][self.PROBABILITIES]
            data2 = stats_b[label][self.PROBABILITIES]
            for t in self.PROBABILITY_DENSITIES:
                figure_path = get_plot_path_fn(t, label) if get_plot_path_fn != None else self.default_path_fn(t, label)
                print(f"\nLabel stats {figure_path}:")
                probs1 = data1[t]
                probs2 = data2[t]
                keys = set(probs1.keys()).union(set(probs2.keys()))
                values1 = [probs1[k] if k in probs1 else 0 for k in keys]
                values2 = [probs2[k] if k in probs2 else 0 for k in keys]

                plt.clf()
                r2 = r2_score(values1, values2)
                print(f"{t} r-squared = {r2}")
                plt.scatter(values1, values2, marker=".", alpha=0.66)
                maxVal = min(1.1 * max(max(values1), max(values2)), 1.0)
                # maxVal *= (0.3 if 'Sequential' in t else (0.45 if 'Code' in t else 0.3))
                
                plt.xlim([0,maxVal])
                plt.ylim([0,maxVal])
                plt.title(f"{label} {t}")
                plt.xlabel(plot_label_a)
                plt.ylabel(plot_label_b)
                plt.annotate("r-squared = {:.3f}".format(r2), (0.1*maxVal, 0.9*maxVal))
                
                plt.savefig(figure_path)
                plot_paths.append(figure_path)

        return plot_paths


if __name__ == "__main__":

    from pyhealth.datasets import eICUDataset
    from pyhealth.data import Event

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- pyhealth dataset/source ---    
    DATASET_NAME = "eICU-demo"
    # ROOT = "https://storage.googleapis.com/pyhealth/eicu-demo/",
    ROOT = "/home/bdanek2/ai4health/dataset/physionet.org/files/eicu-crd/2.0_unpacked"
    TABLES = ["diagnosis", "medication", "lab", "treatment"]
    CODE_MAPPING = {}
    DEV = True  # not needed when using demo set since its 100 patients large
    REFRESH_CACHE = False

    dataset = eICUDataset(
        dataset_name=DATASET_NAME,
        root=ROOT,
        tables=TABLES,
        code_mapping=CODE_MAPPING,
        dev=DEV,
        refresh_cache=REFRESH_CACHE,
    )

    basedir = '/home/bdanek2/PyHealth'

    # --- processor ---
    save_processor_path = f'{basedir}/model_saves/halo_dev_processor.pkl'
    batch_size = 2048
    
    # define a way to make labels from raw data
    simple_label_fn_output_size = 1
    def simple_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        
        return (1) if pdata.gender == 'Male' else (0)
    
    def handle_measurement(event: Event):
        # ex:  event = discretizer.discretize(event.lab_value)
        return event
    
    # complex case:
    # handle labs of 1 kind in one way, and labs of another in another way
    
    # define a way to handle events that need some special event handling
    # this is where you would define some discrtization strategy
    event_handlers = {}   
    event_handlers['measurement'] =  handle_measurement

    # handle continuous value discretization    
    def handle_lab(x):
        bins = [0, 0.5, 1, 2, 4, 8, 16, 32]
        np.digitize(x, bins)

    continuous_value_handlers = {}
    continuous_value_handlers['lab'] = handle_lab
    
    # handle discretization of time
    bins = [0.5, 1, 1.5, 2, 2.5, 3, 4] # re-admission to icu is releveant only in the short term
    time_vector_length = len(bins) + 1
    def handle_time(t: timedelta):
        year = t.days / 365
        bin = np.digitize(year, bins)
        vect = np.zeros((time_vector_length))
        vect[bin] = 1
        return vect

    processor = HALOProcessor(
        dataset=dataset,
        use_tables=None,
        event_handlers=event_handlers,
        continuous_value_handlers=continuous_value_handlers,
        time_handler=handle_time,
        time_vector_length=time_vector_length,
        label_fn=simple_label_fn,
        label_vector_len=simple_label_fn_output_size,
        invert_label=None
    )
    
    # # save for developement
    # pickle.dump(processor, open(save_processor_path, 'wb'))
    # processor = pickle.load(open(save_processor_path, 'rb'))

    # --- define model & opt ---
    model = HALOModel(
        n_ctx=processor.total_visit_size,
        total_vocab_size=processor.total_vocab_size,
        device=device
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # --- train model ---
    model_save_path = '/home/bdanek2/PyHealth/'
    trainer = HALOTrainer(
        dataset=dataset,
        model=model,
        processor=processor,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        checkpoint_path=f'{basedir}/model_saves/eval_developement.pkl'
    )
    
    trainer.set_basic_splits()
    
    start_time = time.perf_counter()
    # trainer.train(
    #     batch_size=batch_size,
    #     epoch=2000,
    #     patience=5,
    #     eval_period=1000
    # )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print("training time:", run_time, run_time / 60, (run_time / 60) / 60)
    
    # --- generate synthetic dataset using the best model ---
    state_dict = torch.load(f'{basedir}/model_saves/eval_developement.pkl', map_location=device)

    model.load_state_dict(state_dict['model'])
    model.to(device)
    trainer = HALOTrainer(
        dataset=dataset,
        model=model,
        processor=processor,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
        checkpoint_path=f'{basedir}/model_saves/eval_developement.pkl'
    )
    
    trainer.set_basic_splits()
    trainer.eval(batch_size=batch_size, save=False)

    generator = HALOGenerator(
        model=model,
        processor=processor,
        batch_size=batch_size,
        device=device,
        save_path=f"{basedir}/synthetically_generated_data.pkl"
    )

    labels = [((1), 10000), ((0), 10000)]
    synthetic_dataset = generator.generate_conditioned(labels)

    # --- evaluation ---
    labels = {
        (1, ): 'male',
        (0, ): 'female',
    }
    def pathfn(plot_type: str, label: List):
        prefix = f"./halo_eval_plots"

        label = labels[label] if label in labels else 'all_labels'
        label = label.replace('.', '').replace(' ', '').lower()
        path_str = f"{prefix}_{plot_type}_{label}"

        return path_str

    # conduct evaluation of the synthetic data w.r.t. it's source
    evaluator = HALOEvaluator(generator=generator, processor=processor)
    stats = evaluator.evaluate(
        source=trainer.test_dataset,
        synthetic=pickle.load(file=open('./synthetically_generated_data.pkl', 'rb')),
        get_plot_path_fn=pathfn,
        compare_label=list(labels.keys()),
    )

    print("plots at:", '\n'.join(stats[evaluator.PLOT_PATHS]))

    print("done")