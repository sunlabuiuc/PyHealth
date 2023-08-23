
"""
Model architecture and method from Theodorou, Brandon, Cao Xiao, and Jimeng Sun. “Synthesize Extremely High-Dimensional Longitudinal Electronic Health Records via Hierarchical Autoregressive Language Model.” arXiv, April 4, 2023. http://arxiv.org/abs/2304.02169.
"""
from datetime import timedelta
import datetime
import os
import time
from matplotlib import pyplot as plt
import numpy as np
import copy
import math
from typing import List
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.data import Event 
from pyhealth.datasets.eicu import eICUDataset
from pyhealth.synthetic.halo.evaluator import Evaluator
from pyhealth.synthetic.halo.generator import Generator
from pyhealth.synthetic.halo.processor import Processor
from pyhealth.synthetic.halo.trainer import Trainer

"""model configuration
required fields are non-optional for instantiating the HALO model: "n_positions", "n_ctx", "n_embd", "n_layer", "n_head", "layer_norm_epsilon", "initializer_range"
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

class HALO(nn.Module):
    """HALO LLM Model.
    Based on Theodorou, Brandon, Cao Xiao, and Jimeng Sun. “Synthesize Extremely High-Dimensional Longitudinal Electronic Health Records via Hierarchical Autoregressive Language Model.” arXiv, April 4, 2023. http://arxiv.org/abs/2304.02169.

    Args:
        n_ctx: the number of context vectors the model should expect; equivalent to maximum number of visits to generate
        total_vocab_size: size of the vocabulary
        device: training device
        config: configuration object to instantiate the model. HALO authors provide defaults, but can be modified so long as required fields are included. 
    """
    def __init__(self, 
            n_ctx,
            total_vocab_size,
            device,
            config: Config = None
        ):
        super(HALO, self).__init__()
        
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
    

if __name__ == "__main__":
    from pyhealth.datasets import eICUDataset
    from pyhealth.data import Event

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ROOT = "https://storage.googleapis.com/pyhealth/eicu-demo/"

    ROOT = "/home/bdanek2/data/physionet.org/files/eicu-crd/2.0"
    # ROOT = "/home/bpt3/data/physionet.org/files/eicu-crd/2.0"
    dataset = eICUDataset(
        dataset_name="eICU-demo",
        root=ROOT,
        tables=["diagnosis", "lab"],
        code_mapping={},
        dev=False,
        refresh_cache=False,
    )

    basedir = '/home/bdanek2/halo_developement/testing_1'
    # basedir = '/home/bpt3/code/PyHealth/pyhealth/synthetic/halo/temp'

    # --- processor ---
    batch_size = 512
    
    # define a way to make labels from raw data
    simple_label_fn_output_size = 1
    def simple_label_fn(**kwargs):
        pdata = kwargs['patient_data']
        return (0) if pdata.death_datetime else (1) # 0 for dead, 1 for alive
    
    def handle_diagnosis(event: Event):
        """to reduce the complexity of the model, in this example we will convert granular ICD codes to more broad ones (ie 428.3 --> 428)"""
        split_code = event.code.split('.')
        assert len(split_code) <= 2
        return split_code[0]
    # these values will be used to compute histograms
    def handle_lab(event: Event): 
        value = float(event.attr_dict['lab_result'])
        return value

    # define value handlers; these handlers serve the function of converting an event into a primitive value. 
    event_handlers = {}   
    event_handlers['diagnosis'] =  handle_diagnosis
    event_handlers['lab'] = handle_lab 
    
    # this event handler is called after the histograms have been computed
    def handle_discrete_lab(event: Event, bin_index: int):
        lab_name = event.code
        lab_value = bin_index
        lab_unit = event.attr_dict['lab_measure_name_system']

        return f"{lab_name}_{lab_unit}_{lab_value}"
    
    discrete_event_handlers = {}
    discrete_event_handlers['lab'] = handle_discrete_lab
    
    processor = Processor(
        dataset=dataset,
        use_tables=None,
        event_handlers=event_handlers,
        compute_histograms=['lab'],
        size_per_event_bin={'lab': 10},
        discrete_event_handlers=discrete_event_handlers,
        size_per_time_bin=10,
        label_fn=simple_label_fn,
        label_vector_len=simple_label_fn_output_size,
        name="HALO-Developement-Test",
        refresh_cache=True
    )

    print(f"Processor results in vocab len {processor.total_vocab_size}, max visit num: {processor.total_visit_size}")
    
    model = HALO(
        n_ctx=processor.total_visit_size,
        total_vocab_size=processor.total_vocab_size,
        device=device
    )
    print(model.__call__)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(optimizer.__class__)
    # state_dict = torch.load(open(f'{basedir}/model_saves/eval_developement.pt', 'rb'), map_location=device)
    # model.load_state_dict(state_dict['model'])
    # model.to(device)
    # optimizer.load_state_dict(state_dict['optimizer'])
    # print("loaded previous model from traing; iterations on previous model:", state_dict['iteration'])
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # --- train model ---
    trainer = Trainer(
        dataset=dataset,
        model=model,
        processor=processor,
        optimizer=optimizer,
        checkpoint_dir=f'{basedir}/model_saves',
        model_save_name='eval_developement_test'
    )
    s = trainer.set_basic_splits(from_save=True, save=True)
    print('split lengths', [len(_s) for _s in s])
    
    start_time = time.perf_counter()
    trainer.train(
        batch_size=batch_size,
        epoch=1000,
        patience=3,
        eval_period=float('inf')
    )
    end_time = time.perf_counter()
    run_time = end_time - start_time
    print("training time:", run_time, run_time / 60, (run_time / 60) / 60)
    
    # --- generate synthetic dataset using the best model ---
    state_dict = torch.load(open(trainer.get_model_checkpoint_path(), 'rb'), map_location=device)

    model.load_state_dict(state_dict['model'])
    model.to(device)

    generator = Generator(
        model=model,
        processor=processor,
        batch_size=batch_size,
        device=device,
        save_dir=basedir,
        save_name="synthetically_generated_mortality_data", # save at `synthetically_generated_mortality_data.pkl`
    )

    labels = [((1), 25000), ((0), 25000)]
    synthetic_dataset = generator.generate_conditioned(labels)

    def pathfn(plot_type: str, label: List):
        prefix = os.path.join(generator.save_dir, 'plots')

        label = labels[label] if label in labels else 'all_labels'
        label = label.replace('.', '').replace(' ', '').lower()
        path_str = f"{prefix}_{plot_type}_{label}"

        return path_str
    
    labels = {
        (1, ): 'Alive',
        (0, ): 'Expired',
    }

    # conduct evaluation of the synthetic data w.r.t. it's source
    evaluator = Evaluator(generator=generator, processor=processor)
    stats = evaluator.evaluate(
        source=trainer.test_dataset[:10],
        synthetic=pickle.load(file=open(generator.save_path, 'rb'))[:10],
        get_plot_path_fn=pathfn,
        compare_label=list(labels.keys()),
    )
    print("plots at:", '\n'.join(stats[evaluator.PLOT_PATHS]))

    # --- conversion ---
    print("done")