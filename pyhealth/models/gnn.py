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

"""Graph Neural Network models for PyHealth.

This module provides implementations of Graph Convolutional Network (GCN) and
Graph Attention Network (GAT) models for healthcare data analysis. These models
are designed to work with PyHealth 2.0 datasets and can be used for various
prediction tasks in medical data.

The module includes:
- GraphConvolution: Basic GCN layer implementation.
- GraphAttention: Basic GAT layer implementation.
- GCN: Full GCN model for patient-level predictions.
- GAT: Full GAT model for patient-level predictions.
"""

def _to_tensor(
    value,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Converts supported array-like values to tensors on the desired device/dtype."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.tensor(value, device=device, dtype=dtype)


def _prepare_feature_adj(
    adj_value,
    *,
    batch_size: int,
    num_features: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Normalizes feature-level adjacency to shape [batch_size, num_features, num_features]."""
    if adj_value is None:
        return None
    adj = _to_tensor(adj_value, device=device, dtype=dtype)
    if adj.dim() == 2:
        if adj.shape[0] != num_features or adj.shape[1] != num_features:
            raise ValueError(
                f"feature_adj must be of shape [{num_features}, {num_features}] "
                f"or [batch_size, {num_features}, {num_features}]"
            )
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
    elif adj.dim() == 3:
        if (
            adj.shape[0] != batch_size
            or adj.shape[1] != num_features
            or adj.shape[2] != num_features
        ):
            raise ValueError(
                f"feature_adj with 3 dimensions must match "
                f"[batch_size, {num_features}, {num_features}]"
            )
    else:
        raise ValueError("feature_adj must be either 2D or 3D tensor")
    return adj


def _prepare_visit_adj(
    adj_value,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Normalizes visit-level adjacency to shape [batch_size, batch_size]."""
    if adj_value is None:
        return None
    adj = _to_tensor(adj_value, device=device, dtype=dtype)
    if adj.dim() != 2 or adj.shape[0] != batch_size or adj.shape[1] != batch_size:
        raise ValueError(
            f"visit_adj must be a 2D tensor of shape [{batch_size}, {batch_size}]"
        )
    return adj


class GraphConvolution(Module):
    """Simple GCN layer, similar to https://arxiv.org/abs/1609.02907.

    This layer implements a basic graph convolutional operation that aggregates
    node features from neighboring nodes using a learnable weight matrix.
    """

    def __init__(self, in_features, out_features, bias=True, init='xavier'):
        """Initializes the GraphConvolution layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            bias: Whether to include bias term. Defaults to True.
            init: Initialization method ('uniform', 'xavier', 'kaiming').
                Defaults to 'xavier'.
        """
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
        """Resets parameters using uniform initialization."""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        """Resets parameters using Xavier initialization."""
        nn.init.xavier_normal_(self.weight.data, gain=0.02) # Implement Xavier Uniform
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def reset_parameters_kaiming(self):
        """Resets parameters using Kaiming initialization."""
        nn.init.kaiming_normal_(self.weight.data, a=0, mode='fan_in')
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, input, adj):
        """Performs forward pass of the GraphConvolution layer.

        Args:
            input: Input features tensor.
            adj: Adjacency matrix tensor.

        Returns:
            Output features tensor after convolution.
        """
        support = torch.mm(input, self.weight)
        if adj.layout == torch.strided:
            output = torch.mm(adj, support)
        else:
            sparse_adj = adj if adj.layout == torch.sparse_coo else adj.to_sparse()
            output = torch.sparse.mm(sparse_adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """Returns string representation of the layer."""
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphAttention(nn.Module):
    """Simple GAT layer, similar to https://arxiv.org/abs/1710.10903.

    This layer implements a basic graph attention mechanism that computes
    attention coefficients between nodes and aggregates features using
    learnable attention weights.
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        """Initializes the GraphAttention layer.

        Args:
            in_features: Number of input features.
            out_features: Number of output features.
            dropout: Dropout rate for attention coefficients.
            alpha: LeakyReLU negative slope for attention computation.
            concat: Whether to concatenate attention heads. Defaults to True.
        """
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
        """Performs forward pass of the GraphAttention layer.

        Args:
            input: Input features tensor.
            adj: Adjacency matrix tensor.

        Returns:
            Output features tensor after attention aggregation.
        """
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
        """Returns string representation of the layer."""
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

device = 'cpu'
    


class GCN(BaseModel):
    """GCN model for PyHealth 2.0 datasets.

    This model embeds each feature stream, concatenates the embeddings,
    then applies Graph Convolutional Network on a fully-connected graph
    of patients in the batch.

    Args:
        dataset: Dataset providing processed inputs.
        embedding_dim: Shared embedding dimension. Defaults to 128.
        nhid: Hidden dimension for GCN layers. Defaults to 64.
        dropout: Dropout rate applied in GCN. Defaults to 0.5.
        init: Initialization method for GCN layers. Defaults to 'xavier'.
        num_layers: Number of GCN layers. Defaults to 2.

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
        """Splits temporal features if present.

        Args:
            feature: Feature data, potentially containing temporal information.

        Returns:
            Tuple of (temporal_info, feature_data) or (None, feature_data).
        """
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value) -> torch.Tensor:
        """Ensures the value is a tensor with appropriate dtype.

        Args:
            feature_key: Key identifying the feature type.
            value: Value to convert to tensor.

        Returns:
            Tensor representation of the value.
        """
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _pool_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Pools embedding tensor to reduce dimensions.

        Args:
            x: Input embedding tensor.

        Returns:
            Pooled embedding tensor.
        """
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Performs forward pass of the GCN model.

        Args:
            **kwargs: Input features and labels. Must include feature keys
                and label key. Optionally supports:
                - feature_adj: Tensor/array defining feature-feature adjacency.
                  Shape can be [num_features, num_features] shared across the
                  batch or [batch_size, num_features, num_features].
                - visit_adj: Tensor/array defining visit/patient adjacency of
                  shape [batch_size, batch_size]. If omitted, a fully connected
                  graph is used.

        Returns:
            Dictionary containing loss, predictions, true labels, logits,
            and optionally embeddings.
        """
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

        feature_tensor = torch.stack(patient_embs, dim=1)  # (batch, feature, dim)
        batch_size, num_features, _ = feature_tensor.size()

        feature_adj = _prepare_feature_adj(
            kwargs.get("feature_adj"),
            batch_size=batch_size,
            num_features=num_features,
            device=feature_tensor.device,
            dtype=feature_tensor.dtype,
        )
        if feature_adj is not None:
            feature_tensor = torch.matmul(feature_adj, feature_tensor)

        patient_emb = feature_tensor.reshape(batch_size, -1)

        visit_adj = _prepare_visit_adj(
            kwargs.get("visit_adj"),
            batch_size=batch_size,
            device=patient_emb.device,
            dtype=patient_emb.dtype,
        )
        if visit_adj is None:
            visit_adj = patient_emb.new_ones((batch_size, batch_size))

        x = patient_emb
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, visit_adj)
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
        dataset: Dataset providing processed inputs.
        embedding_dim: Shared embedding dimension. Defaults to 128.
        nhid: Hidden dimension for GAT layers. Defaults to 64.
        dropout: Dropout rate applied in GAT. Defaults to 0.5.
        nheads: Number of attention heads. Defaults to 1.
        num_layers: Number of GAT layers. Defaults to 2.

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
        """Splits temporal features if present.

        Args:
            feature: Feature data, potentially containing temporal information.

        Returns:
            Tuple of (temporal_info, feature_data) or (None, feature_data).
        """
        if isinstance(feature, tuple) and len(feature) == 2:
            return feature
        return None, feature

    def _ensure_tensor(self, feature_key: str, value) -> torch.Tensor:
        """Ensures the value is a tensor with appropriate dtype.

        Args:
            feature_key: Key identifying the feature type.
            value: Value to convert to tensor.

        Returns:
            Tensor representation of the value.
        """
        if isinstance(value, torch.Tensor):
            return value
        processor = self.feature_processors[feature_key]
        if isinstance(processor, SequenceProcessor):
            return torch.tensor(value, dtype=torch.long)
        return torch.tensor(value, dtype=torch.float)

    def _pool_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Pools embedding tensor to reduce dimensions.

        Args:
            x: Input embedding tensor.

        Returns:
            Pooled embedding tensor.
        """
        if x.dim() == 4:
            x = x.sum(dim=2)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return x

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Performs forward pass of the GAT model.

        Args:
            **kwargs: Input features and labels. Must include feature keys
                and label key. Optionally supports:
                - feature_adj: Tensor/array defining feature-feature adjacency.
                  Shape can be [num_features, num_features] shared across the
                  batch or [batch_size, num_features, num_features].
                - visit_adj: Tensor/array defining visit/patient adjacency of
                  shape [batch_size, batch_size]. If omitted, a fully connected
                  graph is used.

        Returns:
            Dictionary containing loss, predictions, true labels, logits,
            and optionally embeddings.
        """
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

        feature_tensor = torch.stack(patient_embs, dim=1)
        batch_size, num_features, _ = feature_tensor.size()

        feature_adj = _prepare_feature_adj(
            kwargs.get("feature_adj"),
            batch_size=batch_size,
            num_features=num_features,
            device=feature_tensor.device,
            dtype=feature_tensor.dtype,
        )
        if feature_adj is not None:
            feature_tensor = torch.matmul(feature_adj, feature_tensor)

        patient_emb = feature_tensor.reshape(batch_size, -1)

        visit_adj = _prepare_visit_adj(
            kwargs.get("visit_adj"),
            batch_size=batch_size,
            device=patient_emb.device,
            dtype=patient_emb.dtype,
        )
        if visit_adj is None:
            visit_adj = patient_emb.new_ones((batch_size, batch_size))

        x = F.dropout(patient_emb, self.dropout, training=self.training)
        for i, gat_layer in enumerate(self.gat_layers):
            x = gat_layer(x, visit_adj)
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







