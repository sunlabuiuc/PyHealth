'''
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
'''
import numpy as np
import copy
import math
from typing import Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pyhealth.datasets.base_ehr_dataset import BaseEHRDataset
from pyhealth.datasets.eicu import eICUDataset

from pyhealth.datasets.sample_dataset import SampleEHRDataset

"""
model configuration
"""
'''
    code by Brandon Theodorou
    Original GPT-2 Paper and repository here: https://github.com/openai/gpt-2
    Original GPT-2 Pytorch Model: https://github.com/huggingface/pytorch-pretrained-BERT
    GPT-2 Pytorch Model Derived From: https://github.com/graykode/gpt-2-Pytorch
'''
class Config(object):
    def __init__(
            self,
            total_vocab_size=5839,

            code_vocab_size=5812,
            continuous_vocab_size=12,
            label_vocab_size=12,
            special_vocab_size=3,
            
            n_positions=20,
            n_ctx=20,
            n_embd=768,
            n_layer=12,
            n_head=12,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            batch_size=128,
            sample_batch_size=512,
            epoch=100,
            patience=5,
            lr=1e-4,
    ):
        self.total_vocab_size = total_vocab_size
        self.code_vocab_size = code_vocab_size
        self.continuous_vocab_size = continuous_vocab_size
        self.label_vocab_size = label_vocab_size
        self.special_vocab_size = special_vocab_size
        self.n_positions = n_positions
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.epoch = epoch
        self.patience = patience
        self.lr = lr


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
    def __init__(self, config):
        super(HALOModel, self).__init__()
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
class HALOAggregator():
    
    # what does pad_code do again? 
    SPECIAL_VOCAB = ('start_code', 'last_visit_code', 'pad_code')
    
    # visit dimension
    START_INDEX = 0
    LABEL_INDEX = 1
    VISIT_INDEX = 2
    
    """
    discretizator: Table --> Discretizer
    max_visits: the maximum number of visits to include per patient. 
        if `None` is provided, this number will be automatically set
    """
    def __init__(
        self,
        dataset: BaseEHRDataset,
        use_tables: List[str],
        event_handlers: Dict[str, any] = {},
        max_visits: Union[None, int] = None,
        label_fn: callable = lambda data: ''
    ) -> None:
        
        self.dataset = dataset
        
        # whitelisted tables
        self.valid_dataset_tables = use_tables 
        
        # handle processing of event types
        self.event_handlers = event_handlers 
        
        # generate a HALO label based on a patient record
        self.label_fn = label_fn
        
        self.max_visits = max_visits
        
    """
    its necessary to aggregate global event data, prior to trnasforming the dataset
    """
    def aggregate_halo_indeces(self) -> SampleEHRDataset:

        # two way mapping from global identifier to index & vice-versa
        # possible since index <> global identifier is bijective
        # type: ((table_name: str, event_value: any): index) or (index: (table_name: str, event_value: any))
        global_events: Dict = {}
        global_labels: Dict = {}
        max_visits: int = 0
        
        for pid, pdata in tqdm(self.dataset.patients.items(), desc="HALOAggregator generating indeces"):
            
            max_visits = max(max_visits, len(pdata.visits))
            
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
                        global_event = (table, te)
                    
                        index_of_global_event = len(global_events)
                        
                        global_events[global_event] = index_of_global_event
                        global_events[index_of_global_event] = global_event
                        
                        
            # compute global label (method provided by user)
            label = self.label_fn(p_id=pid, patient_data=pdata)
            index_of_global_label = len(global_labels)
            
            global_labels[label] = index_of_global_label
            global_labels[index_of_global_label] = label
            
        # set aggregates for downstream processing
        self.global_events = global_events
        self.global_labels = global_labels
        
        # if the user does not provide, infer from dataset
        if self.max_visits == None:
            self.max_visits = max_visits
            
        # define the tokens in the event dimension (visit dimension already specified)
        self.start_token_index = len(self.global_events) + len(self.global_labels)
        self.end_token_index = len(self.global_events) + len(self.global_labels) + 1
        self.pad_token_index = len(self.global_events) + len(self.global_labels) + 2
            
    """
    similar to dataset.set_task(...)
    - produce a sampleEHRDataset
    """  
    def process(self):
        
        # generate index objects used to make multi hot vectors
        self.aggregate_halo_indeces()
        
        samples = []
        
        for pid, pdata in tqdm(self.dataset.patients.items(), desc="HALOAggregator processing samples"):
            total_vocab_size = len(self.global_events) + len(self.global_labels) + len(self.SPECIAL_VOCAB)
            
            sample_multi_hot = np.zeros((len(self.SPECIAL_VOCAB) + self.max_visits, total_vocab_size)) # patient data the model reads
            sample_mask = np.zeros((len(self.SPECIAL_VOCAB) + self.max_visits, 1)) # visits that are unlabeled
            
            # build multihot vector for patient events
            for visit_index, vid,  in enumerate(pdata.visits):
                vdata = pdata.visits[vid]
                
                sample_mask[visit_index] = 1
                
                for table in vdata.available_tables:

                    if self.valid_dataset_tables != None and table not in self.valid_dataset_tables: continue
                    
                    table_events_raw = vdata.get_event_list(table)
                    
                    event_handler = self.event_handlers[table] if table in self.event_handlers else None
                    
                    for te_raw in table_events_raw:
                        
                        te = event_handler(te_raw) if event_handler else te_raw.code
                        global_event = (table, te)
                        event_as_index = self.global_events[global_event]
                        
                        # set table events
                        sample_multi_hot[self.VISIT_INDEX + visit_index, event_as_index] = 1
            
            # set the start token
            sample_multi_hot[self.START_INDEX, self.start_token_index] = 1
            
            # set patient label
            global_label = self.label_fn(p_id=pid, patient_data=pdata)
            label_as_index = self.global_labels[global_label]
            sample_multi_hot[self.LABEL_INDEX, len(self.global_events) + label_as_index] = 1
            
            # set the end token
            sample_multi_hot[self.VISIT_INDEX + len(pdata.visits), self.end_token_index] = 1
            
            # set the remainder of the visits to pads if needed
            sample_multi_hot[(self.VISIT_INDEX + len(pdata.visits)):, self.pad_token_index] = 1
            
            # set the mask to cover the labels
            sample_mask[self.LABEL_INDEX] = 1
            
            # "shift the mask to match the shifted labels & predictions the model will return"
            sample_mask = sample_mask[1:, :]
            
            samples.append({
                'patient_id': pdata.patient_id,
                'visit_id': ','.join([v for v in pdata.visits]),
                'data_vector': sample_multi_hot.tolist(), 
                'data_mask': sample_mask.tolist()
            })
            
        return SampleEHRDataset(
            samples=samples,
            dataset_name='dataset_as_halo_vectors',
            task_name='HALOAggregator.process'
        )
    
if __name__ == "__main__":
    from pyhealth.datasets import OMOPDataset
    from pyhealth.trainer import Trainer
    from pyhealth.data import Patient, Event
    from pyhealth.datasets import split_by_patient, get_dataloader
    

    
    # # to test the file this path needs to be updated
    # DATASET_NAME = "eICU-demo"
    # ROOT = "https://storage.googleapis.com/pyhealth/eicu-demo/"
    # TABLES = ["diagnosis", "medication", "lab", "treatment", "physicalExam"]
    # CODE_MAPPING = {}
    # DEV = True  # not needed when using demo set since its 100 patients large
    # REFRESH_CACHE = False

    # pyhealth_dataset = eICUDataset(
    #     dataset_name=DATASET_NAME,
    #     root=ROOT,
    #     tables=TABLES,
    #     code_mapping=CODE_MAPPING,
    #     dev=DEV,
    #     refresh_cache=REFRESH_CACHE,
    # )
    
    # DATASET_NAME = "omop-demo"
    # ROOT = "https://storage.googleapis.com/pyhealth/synpuf1k_omop_cdm_5.2.2/"
    # TABLES = [
    #     "condition_occurrence",
    #     "procedure_occurrence",
    #     "drug_exposure",
    #     "measurement",
    # ]
    # CODE_MAPPING = {}
    # DEV = True  # not needed when using demo set since its 100 patients large
    # REFRESH_CACHE = False

    # dataset = OMOPDataset(
    #     dataset_name=DATASET_NAME,
    #     root=ROOT,
    #     tables=TABLES,
    #     code_mapping=CODE_MAPPING,
    #     dev=DEV,
    #     refresh_cache=REFRESH_CACHE,
    # )
    
    DATASET_NAME = "eICU-demo"
    ROOT = "https://storage.googleapis.com/pyhealth/eicu-demo/"
    TABLES = ["diagnosis", "medication", "lab", "treatment", "physicalExam"]
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
    
    # define a way to make labels from raw data
    def simple_label_fn(**kwargs):
        p_id = kwargs['p_id']
        pdata = kwargs['patient_data']
        
        return pdata.gender
    
    def handle_measurement(event: Event):
        
        return event
    
    # define a way to handle events that need some special event handling
    # this is where you would define some discrtization strategy
    event_handlers = {}    
    event_handlers['measurement'] =  handle_measurement
    
    sampleDataset = HALOAggregator(
        dataset=dataset,
        use_tables=None,
        event_handlers=event_handlers,
        label_fn=simple_label_fn
    ).process()
    
    print(sampleDataset)
    
    # dataloader for train, val, test
    # dataset split
    train_ds, val_ds, test_ds = split_by_patient(sampleDataset, [0.8, 0.1, 0.1])

    # obtain train/val/test dataloader, they are <torch.data.DataLoader> object
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)
    
    # get batch function
    
    model = HALOModel(config).to(device)
    
    print(model.mode)
    
    trainer = Trainer(
        model,
        checkpoint_path='/Users/benjamindanek/Code/model_checkpoints',
        metrics=None # //todo: update
    )
    
    trainer.train(
        epochs=1, 
        optimizer_params={"lr": 1e-10},
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
    )