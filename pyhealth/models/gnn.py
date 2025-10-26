import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy 
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
from tqdm import tqdm 
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from typing import Dict, Optional

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import SequenceProcessor
from pyhealth.models.embedding import EmbeddingModel

torch.manual_seed(3) 
np.random.seed(1)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        if init == 'uniform':
            print("| Uniform Initialization")
            self.reset_parameters_uniform()
        elif init == 'xavier':
            print("| Xavier Initialization")
            self.reset_parameters_xavier()
        elif init == 'kaiming':
            print("| Kaiming Initialization")
            self.reset_parameters_kaiming()
        else:
            raise NotImplementedError

    def reset_parameters_uniform(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # print("adj", adj.dtype, "support", support.dtype)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttention(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttention, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(nn.init.xavier_normal_(torch.empty(in_features, out_features), gain=np.sqrt(2.0)), requires_grad=True)
        self.a1 = nn.Parameter(nn.init.xavier_normal_(torch.empty(out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)
        self.a2 = nn.Parameter(nn.init.xavier_normal_(torch.empty(out_features, 1), gain=np.sqrt(2.0)), requires_grad=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        f_1 = torch.matmul(h, self.a1)
        f_2 = torch.matmul(h, self.a2)
        e = self.leakyrelu(f_1 + f_2.transpose(0,1))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

device = 'cpu'
    


class GCN(BaseModel):
    """GCN model for PyHealth 2.0 datasets.

    This model embeds each feature stream, concatenates the embeddings,
    then applies Graph Convolutional Network on a fully-connected graph
    of patients in the batch.

    Args:
        dataset (SampleDataset): dataset providing processed inputs.
        embedding_dim (int): shared embedding dimension.
        nhid (int): hidden dimension for GCN layers.
        dropout (float): dropout rate applied in GCN.
        init (str): initialization method for GCN layers.
        num_layers (int): number of GCN layers.

    Examples:
        >>> from pyhealth.datasets import SampleDataset, get_dataloader
        >>> samples = [...]
        >>> dataset = SampleDataset(...)
        >>> model = GCN(dataset=dataset)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        nhid: int = 64,
        dropout: float = 0.5,
        init: str = 'xavier',
        num_layers: int = 2,
    ):
        super(GCN, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.nhid = nhid
        self.dropout = dropout
        self.init = init
        self.num_layers = num_layers

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }

        input_dim = len(self.feature_keys) * embedding_dim
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GraphConvolution(input_dim, nhid, init=init))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GraphConvolution(nhid, nhid, init=init))
        self.gcn_layers.append(GraphConvolution(nhid, self.get_output_size(), init=init))

    def _split_temporal(self, feature):
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _pool_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_embs = []
        embedding_inputs: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            x = self._pool_embedding(x)
            x = x.mean(dim=1)
            patient_embs.append(x)

        patient_emb = torch.cat(patient_embs, dim=1)

        batch_size = patient_emb.size(0)
        adj = torch.ones(batch_size, batch_size, device=self.device)

        x = patient_emb
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            if i < len(self.gcn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, self.dropout, training=self.training)

        logits = x
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = x
        return results


class GAT(BaseModel):
    """GAT model for PyHealth 2.0 datasets.

    This model embeds each feature stream, concatenates the embeddings,
    then applies Graph Attention Network on a fully-connected graph
    of patients in the batch.

    Args:
        dataset (SampleDataset): dataset providing processed inputs.
        embedding_dim (int): shared embedding dimension.
        nhid (int): hidden dimension for GAT layers.
        dropout (float): dropout rate applied in GAT.
        nheads (int): number of attention heads.
        num_layers (int): number of GAT layers.

    Examples:
        >>> from pyhealth.datasets import SampleDataset, get_dataloader
        >>> samples = [...]
        >>> dataset = SampleDataset(...)
        >>> model = GAT(dataset=dataset)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> output = model(**batch)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        nhid: int = 64,
        dropout: float = 0.5,
        nheads: int = 1,
        num_layers: int = 2,
    ):
        super(GAT, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.nhid = nhid
        self.dropout = dropout
        self.nheads = nheads
        self.num_layers = num_layers

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.feature_processors = {
            feature_key: self.dataset.input_processors[feature_key]
            for feature_key in self.feature_keys
        }

        input_dim = len(self.feature_keys) * embedding_dim
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GraphAttention(input_dim, nhid, dropout=dropout, alpha=0.2, concat=True))
        for _ in range(num_layers - 1):
            self.gat_layers.append(GraphAttention(nhid, nhid, dropout=dropout, alpha=0.2, concat=True))
        self.gat_layers.append(GraphAttention(nhid, self.get_output_size(), dropout=dropout, alpha=0.2, concat=False))

    def _split_temporal(self, feature):
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _pool_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        patient_embs = []
        embedding_inputs: Dict[str, torch.Tensor] = {}

        for feature_key in self.feature_keys:
            _, value = self._split_temporal(kwargs[feature_key])
            value_tensor = self._ensure_tensor(feature_key, value)
            embedding_inputs[feature_key] = value_tensor

        embedded = self.embedding_model(embedding_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]
            x = self._pool_embedding(x)
            x = x.mean(dim=1)
            patient_embs.append(x)

        patient_emb = torch.cat(patient_embs, dim=1)

        batch_size = patient_emb.size(0)
        adj = torch.ones(batch_size, batch_size, device=self.device)

        x = F.dropout(patient_emb, self.dropout, training=self.training)
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, adj)
            if i < len(self.gat_layers) - 1:
                x = F.dropout(F.elu(x), self.dropout, training=self.training)

        logits = x
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = x
        return results




if __name__ == "__main__":
    from pyhealth.datasets import SampleDataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "diagnoses": ["A", "B", "C"],
            "procedures": ["X", "Y"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-0",
            "diagnoses": ["D", "E"],
            "procedures": ["Z"],
            "label": 0,
        },
    ]

    input_schema = {"diagnoses": "sequence", "procedures": "sequence"}
    output_schema = {"label": "binary"}

    dataset = SampleDataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = GCN(dataset=dataset, embedding_dim=64, nhid=32, num_layers=1)

    data_batch = next(iter(train_loader))

    result = model(**data_batch)
    print(result)

    result["loss"].backward()











