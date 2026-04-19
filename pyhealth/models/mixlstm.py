import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from collections import abc
from abc import ABC

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

class MLP(nn.Module):

    def __init__(self, neuron_sizes, activation=nn.LeakyReLU, bias=True): 
        super(MLP, self).__init__()
        self.neuron_sizes = neuron_sizes
        
        layers = []
        for s0, s1 in zip(neuron_sizes[:-1], neuron_sizes[1:]):
            layers.extend([
                nn.Linear(s0, s1, bias=bias),
                activation()
            ])
        
        self.classifier = nn.Sequential(*layers[:-1])

    def eval_forward(self, x, y):
        self.eval()
        return self.forward(x)
        
    def forward(self, x):
        x = x.contiguous()
        x = x.view(-1, self.neuron_sizes[0])
        return self.classifier(x)

############################ main models ##################################
class MoE(nn.Module):

    ''' 
    This is a abstract base class for mixture of experts
    
    it supports:
    a) specifiying experts
    b) specifying the gating function (having parameter or not)

    it needs combining functions (either MoO or MoE)
    '''

    def __init__(self, experts, gate):
        super(MoE, self).__init__()        
        self.experts = experts
        self.gate = gate

class MoO(MoE):

    '''
    mixture of outputs
    '''
    def __init__(self, experts, gate, bs_dim=1, expert_dim=0):
        super(MoO, self).__init__(experts, gate)        
        # this is for RNN architecture: bs_dim = 2 for RNN
        self.bs_dim = bs_dim
        self.expert_dim = expert_dim

    def combine(self, o, coef):

        if isinstance(o[0], abc.Sequence): # account for multi_output setting
            return [self.combine(o_, coef) for o_ in zip(*o)]
        else:
            o = torch.stack(o)
            # reshape o to (_, bs, n_expert)  b/c coef is (bs, n_expert)
            o = o.transpose(self.expert_dim, -1)
            o = o.transpose(self.bs_dim, -2) 

            # change back
            res = o * coef
            res = res.transpose(self.expert_dim, -1)
            res = res.transpose(self.bs_dim, -2)
            return res.sum(0)
        
    def forward(self, x, coef=None): # coef is previous coefficient: for IDGate
        coef = self.gate(x, coef) # (bs, n_expert) or n_expert
        self.last_coef = coef
        o = [expert(x) for expert in self.experts]
        return self.combine(o, coef)

class MoW(MoE):

    def forward(self, x, coef=None):
        # assume experts has already been assembled 
        coef = self.gate(x, coef)
        self.last_coef = coef
        return self.experts(x, coef)


################## sample gating functions for get_coefficients ###########
class Gate(ABC, nn.Module):

    '''
    gate function
    '''

    def forward(self, x, coef=None):
        raise NotImplementedError()

class AdaptiveLSTMGate(Gate):

    def __init__(self, input_size, num_experts, normalize=False):
        super(self.__class__, self).__init__()
        self.forward_function = MLP([input_size, num_experts])
        self.normalize = normalize
        
    def forward(self, x, coef=None):
        x, (h, c) = x # h (_, bs, d) 
        o = self.forward_function(h.transpose(0,1)) # (bs, num_experts)
        if self.normalize:            
            return nn.functional.softmax(o, 1)
        else:
            return o
        
class NonAdaptiveGate(Gate):

    def __init__(self, num_experts, coef=None, fixed=False, normalize=False):
        '''
        fixed coefficient: resnet like with predefined not learnable gate values
        normalize: take softmax of the parameters
        '''
        super(self.__class__, self).__init__()
        self.normalize = normalize
        if coef is None: # initialization
            coef = torch.ones(num_experts)
            nn.init.uniform_(coef)
        if fixed:
            coef = nn.Parameter(coef, requires_grad=False)
        else:
            coef = nn.Parameter(coef)

        self.coefficients = coef

    def forward(self, x, coef=None):
        if self.normalize:
            return nn.functional.softmax(self.coefficients, 0)
        else:
            return self.coefficients

class IDGate(Gate): # identity gate

    def forward(self, x, coef): # coef is previous coefficient
        return coef


################ time series example models ################
def moo_linear(in_features, out_features, num_experts, bs_dim=1, expert_dim=0):
    # repeat a linear model for self.num_experts times
    experts = nn.ModuleList()
    for _ in range(num_experts):
        experts.append(nn.Linear(in_features, out_features))

    # tie weights later        
    return MoO(experts, IDGate(), bs_dim=bs_dim, expert_dim=expert_dim) 

class mowLSTM_(nn.Module):

    '''
    helper module for mowLSTM
    '''
    def __init__(self, input_size, hidden_size, num_experts=2, batch_first=False):

        super(mowLSTM_, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.batch_first = batch_first

        # build cell
        self.input_weights = moo_linear(input_size, 4 * hidden_size,
                                        self.num_experts, bs_dim=2) # i,f,g,o
        self.hidden_weights = moo_linear(hidden_size, 4 * hidden_size,
                                         self.num_experts, bs_dim=2)
        # init same as pytorch version
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for m in self.input_weights.experts:
            for name, weight in m.named_parameters():
                nn.init.uniform_(weight, -stdv, stdv)
                # if 'weight' in name:
                #     nn.init.uniform_(weight)
        for m in self.hidden_weights.experts:
            for name, weight in m.named_parameters():
                nn.init.uniform_(weight, -stdv, stdv) 
                # if 'weight' in name:
                #     nn.init.orthogonal_(weight)
                
        # maybe: layer normalization: see jeeheh's code
        # maybe: orthogonal initialization: see jeeheh's code
        # note: pytorch implementation does neither

    def rnn_step(self, x, hidden, coef): # one step of rnn
        bs = x.shape[1]              
        h, c = hidden
        gates = self.input_weights(x, coef) + self.hidden_weights(h, coef)
        # maybe: layer normalization: see jeeheh's code

        ingate, forgetgate, cellgate, outgate = gates.view(bs, -1).chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c = forgetgate * c + ingate * cellgate
        h = outgate * torch.tanh(c) # maybe use layer norm here as well
        return h, c
    
    def forward(self, x, hidden, coef):
        if self.batch_first: # change to seq_len first
            x = x.transpose(0, 1)

        seq_len = x.shape[0]
        output = []
        for t in range(seq_len):
            hidden = self.rnn_step(x[t].unsqueeze(0), hidden, coef)
            output.append(hidden[0]) # seq_len x (_, bs, d)

        output = torch.cat(output, 0)
        return output, hidden

class mowLSTM(nn.Module):

    '''
    helper for mowLSTM, 
    responsible for stacking and bidirectional LSTM
    stack according to 
    https://stackoverflow.com/questions/49224413/difference-between-1-lstm-with-num-layers-2-and-2-lstms-in-pytorch

    '''
    def __init__(self, input_size, hidden_size, num_classes, num_experts=2,
                 num_layers=1, batch_first=False, dropout=0, bidirectional=False,
                 activation=None):

        super(mowLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.batch_first = batch_first
        self.dropouts = nn.ModuleList()

        self.h2o = moo_linear(self.num_directions * self.hidden_size,
                              self.num_classes, self.num_experts, bs_dim=2)
        
        if activation:
            self.activation = activation
        else:
            self.activation = lambda x: x
        
        self.rnns = nn.ModuleList()
        for i in range(num_layers * self.num_directions):
            input_size = input_size if i == 0 else hidden_size
            self.rnns.append(mowLSTM_(input_size, hidden_size, num_experts, batch_first))
            self.dropouts.append(nn.Dropout(p=dropout))

    def forward(self, x, coef):
        x, hidden = x
        self.last_coef = coef
        
        h, c = hidden
        hs, cs = [], []
        for i in range(self.num_layers):
            if i != 0 and i != (self.num_layers - 1):
                x = self.dropouts[i](x) # waste 1 droput out but no problem
            x, hidden = self.rnns[i](x, (h[i].unsqueeze(0), c[i].unsqueeze(0)), coef)
            hs.append(hidden[0])
            cs.append(hidden[1])      

        # todo: bidirectional stacked LSTM, see reference here
        # https://github.com/allenai/allennlp/blob/master/allennlp/modules/stacked_bidirectional_lstm.py; it basically concat layer output

        h = torch.cat(hs, 0)
        c = torch.cat(cs, 0)
        o = x
        # run through prediction layer: o: (seq_len, bs, d)
        o = self.dropouts[0](o)
        o = self.h2o(o, coef)
        o = self.activation(o)

        return o, (h, c)
    

class ExampleMowLSTM(nn.Module):

    '''
    recreate LSTM architectre
    then stack them according to 

    '''
    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, num_directions=1, dropout=0, activation=None):
        super(ExampleMowLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_directions = num_directions
        self.dropout = dropout
        self.activation = activation

    def setKT(self, k, t): # k models t steps
        '''k experts with maximum of t time steps'''
        self.k = k
        self.T = t
        self.cells = nn.ModuleList()

        experts = mowLSTM(self.input_size, self.hidden_size,
                          self.num_classes, num_experts=self.k,
                          num_layers=self.num_layers, dropout=self.dropout,
                          bidirectional = (self.num_directions==2),
                          activation=self.activation)
        self.experts = experts
        
        for _ in range(t):
            gate = NonAdaptiveGate(self.k, normalize=True)
            # gate = AdaptiveLSTMGate(self.hidden_size *\
            #                         self.num_layers *\
            #                         self.num_directions,
            #                         self.k,
            #                         normalize=True)
            self.cells.append(MoW(experts, gate))

    def forward(self, x, hidden):
        seq_len, bs, _ = x.shape
        o = []
        for t in range(seq_len):
            o_, hidden = self.cells[t]((x[t].view(1, bs, -1), hidden))
            o.append(o_)
            
        o = torch.cat(o, 0) # (seq_len, bs, d)
        return o, hidden

    

def orthogonal(shape):
    flat_shape = (int(shape[0]), int(np.prod(shape[1:])))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def lstm_ortho_initializer(shape, scale=1.0):
    size_x = shape[0]
    size_h = int(shape[1]/4) # assumes lstm.
    t = np.zeros(shape)
    t[:, :size_h] = orthogonal([size_x, size_h])*scale
    t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale
    t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
    t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
    return t

class MixLSTM(BaseModel):

    def __init__(self, dataset: SampleDataset, num_experts=2, hidden_size=100):
        super(MixLSTM, self).__init__(dataset)

        #Process dataset to get input dimension and time steps
        input_keys = list(dataset.input_processors.keys())
        # remember the primary input key so Trainer can call model(**batch)
        self.input_key = input_keys[0]
        sample = dataset[0]
        val = sample[input_keys[0]]
        if isinstance(val, (list, tuple)):
            for item in val:
                if torch.is_tensor(item) or isinstance(item, (list, tuple, np.ndarray)):
                    val = item
                    break
        if torch.is_tensor(val):
            input_dim = val.shape[-1] if val.dim() >= 2 else 1
            T = val.shape[0]
        else:
            arr = np.array(val)
            input_dim = arr.shape[-1] if arr.ndim >= 2 else 1
            T = len(val)

        self.input_size = int(input_dim)
        self.time_steps = int(T)
        num_classes = int(self.get_output_size())

        self.model = ExampleMowLSTM(self.input_size, hidden_size,
                                   num_classes, num_layers=1,
                                   num_directions=1, dropout=0,
                                   activation=nn.LogSoftmax(dim=-1))

        self.num_layers = 1
        self.num_directions = 1
        self.hidden_size = hidden_size
        self.model.setKT(num_experts, self.time_steps)
        
    def forward(self, **kwargs):
        # Extract input tensor when called as `model(**batch)` by Trainer.
        x = kwargs.get(self.input_key)

        # change x from (bs, seq_len, d) => (seq_len, bs, d)
        x = x.permute(1, 0, 2)
        batch_size = x.size(1)
        # set initial hidden and cell states on the model device
        device = self.device
        h = torch.zeros(self.num_layers * self.num_directions,
                        batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers * self.num_directions,
                        batch_size, self.hidden_size, device=device)
        
        states = (h, c)
        outputs, states = self.model(x, states)

        # outputs: (seq_len, batch, num_classes) -> (batch, seq_len, num_classes)
        logits_seq = outputs.permute(1, 0, 2)

        # For sequence models used for classification tasks, provide a
        # per-sample logit by selecting the last timestep.
        logits = logits_seq[:, -1, :]

        results = {}
        results["logit"] = logits
        results["y_prob"] = self.prepare_y_prob(logits)

        # If labels were provided in kwargs (Trainer passes them), compute loss
        if hasattr(self, "label_keys") and len(self.label_keys) > 0 and self.label_keys[0] in kwargs:
            y_true = kwargs[self.label_keys[0]].to(self.device)
            loss_fn = self.get_loss_function()
            loss = loss_fn(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        return results

    def after_backward(self):
        return 
 