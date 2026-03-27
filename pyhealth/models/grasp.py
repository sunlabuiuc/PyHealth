"""GRASP model for health status representation learning.

This module implements the GRASP (Generic fRAmework for health Status
representation learning based on incorporating knowledge from Similar
Patients) model from Ma et al., AAAI 2021.

The model clusters patient representations via k-means, refines
cluster-level knowledge with a graph convolutional network, and blends
it back into individual patient embeddings through a learned gate.
"""

import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import kneighbors_graph

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.concare import ConCareLayer
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.rnn import RNNLayer


def random_init(
    dataset: torch.Tensor, num_centers: int, device: torch.device
) -> torch.Tensor:
    """Randomly select initial cluster centers from the dataset.

    Args:
        dataset: tensor of shape [num_points, dimension].
        num_centers: number of cluster centers to select.
        device: target device for the output tensor.

    Returns:
        Tensor of shape [num_centers, dimension] with selected centers.
    """
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = min(num_centers, num_points)

    indices = torch.tensor(
        np.array(random.sample(range(num_points), k=num_centers)), dtype=torch.long
    )

    centers = torch.gather(
        dataset, 0, indices.view(-1, 1).expand(-1, dimension).to(device=device)
    )
    return centers


def compute_codes(
    dataset: torch.Tensor, centers: torch.Tensor
) -> torch.Tensor:
    """Assign each data point to its closest cluster center.

    Args:
        dataset: tensor of shape [num_points, dimension].
        centers: tensor of shape [num_centers, dimension].

    Returns:
        Long tensor of shape [num_points] with cluster assignments.
    """
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)

    # 5e8 should vary depending on the free memory on the GPU
    # Ideally, automatically ;)
    chunk_size = int(5e8 / num_centers)
    codes = torch.zeros(num_points, dtype=torch.long)
    centers_t = torch.transpose(centers, 0, 1)
    centers_norms = torch.sum(centers**2, dim=1).view(1, -1)
    for i in range(0, num_points, chunk_size):
        begin = i
        end = min(begin + chunk_size, num_points)
        dataset_piece = dataset[begin:end, :]
        dataset_norms = torch.sum(dataset_piece**2, dim=1).view(-1, 1)
        distances = torch.mm(dataset_piece, centers_t)
        distances *= -2.0
        distances += dataset_norms
        distances += centers_norms
        _, min_ind = torch.min(distances, dim=1)
        codes[begin:end] = min_ind
    return codes


def update_centers(
    dataset: torch.Tensor,
    codes: torch.Tensor,
    num_centers: int,
    device: torch.device,
) -> torch.Tensor:
    """Recompute cluster centers as the mean of assigned data points.

    Args:
        dataset: tensor of shape [num_points, dimension].
        codes: long tensor of shape [num_points] with cluster assignments.
        num_centers: number of clusters.
        device: target device for the output tensor.

    Returns:
        Tensor of shape [num_centers, dimension] with updated centers.
    """
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    centers = torch.zeros(num_centers, dimension, dtype=torch.float).to(device=device)
    cnt = torch.zeros(num_centers, dtype=torch.float)
    centers.scatter_add_(
        0, codes.view(-1, 1).expand(-1, dimension).to(device=device), dataset
    )
    cnt.scatter_add_(0, codes, torch.ones(num_points, dtype=torch.float))
    # Avoiding division by zero
    # Not necessary if there are no duplicates among the data points
    cnt = torch.where(cnt > 0.5, cnt, torch.ones(num_centers, dtype=torch.float))
    centers /= cnt.view(-1, 1).to(device=device)
    return centers


def cluster(
    dataset: torch.Tensor, num_centers: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run k-means clustering until convergence or 1000 iterations.

    Args:
        dataset: tensor of shape [num_points, dimension].
        num_centers: number of clusters.
        device: target device for computation.

    Returns:
        Tuple of (centers, codes) where centers has shape
        [num_centers, dimension] and codes has shape [num_points].
    """
    centers = random_init(dataset, num_centers, device)
    codes = compute_codes(dataset, centers)
    num_iterations = 0
    while True:
        num_iterations += 1
        centers = update_centers(dataset, codes, num_centers, device)
        new_codes = compute_codes(dataset, centers)
        # Waiting until the clustering stops updating altogether
        # This is too strict in practice
        if torch.equal(codes, new_codes):
            break
        if num_iterations > 1000:
            break
        codes = new_codes
    return centers, codes


class GraphConvolution(nn.Module):
    """Single-layer graph convolution (Kipf & Welling, ICLR 2017).

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: if ``True``, adds a learnable bias. Default: ``True``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).float())
        else:
            self.register_parameter("bias", None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, adj, x, device):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float().to(device=device)
        else:
            return output


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class GRASPLayer(nn.Module):
    """GRASPLayer layer.

    Paper: Liantao Ma et al. GRASP: generic framework for health status representation learning based on incorporating knowledge from similar patients. AAAI 2021.

    This layer is used in the GRASP model. But it can also be used as a
    standalone layer.

    Args:
        input_dim: dynamic feature size.
        static_dim: static feature size, if 0, then no static feature is used.
        hidden_dim: hidden dimension of the GRASP layer, default 128.
        cluster_num: number of clusters, default 12. The cluster_num should be no more than the number of samples.
        dropout: dropout rate, default 0.5.
        block: the backbone model used in the GRASP layer
            ('ConCare', 'LSTM' or 'GRU'), default 'ConCare'.

    Examples:
        >>> from pyhealth.models import GRASPLayer
        >>> x = torch.randn(3, 128, 64)  # [batch, seq_len, feature_size]
        >>> layer = GRASPLayer(64, cluster_num=2)
        >>> c = layer(x)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        cluster_num: int = 2,
        dropout: float = 0.5,
        block: str = "ConCare",
    ):
        super(GRASPLayer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cluster_num = cluster_num
        self.dropout = dropout
        self.block = block

        if self.block == "ConCare":
            self.backbone = ConCareLayer(
                input_dim, static_dim, hidden_dim, hidden_dim, dropout=0
            )
        elif self.block == "GRU":
            self.backbone = RNNLayer(input_dim, hidden_dim, rnn_type="GRU", dropout=0)
        elif self.block == "LSTM":
            self.backbone = RNNLayer(input_dim, hidden_dim, rnn_type="LSTM", dropout=0)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.weight1 = nn.Linear(self.hidden_dim, 1)
        self.weight2 = nn.Linear(self.hidden_dim, 1)
        self.GCN = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN.initialize_parameters()
        self.GCN_2 = GraphConvolution(self.hidden_dim, self.hidden_dim, bias=True)
        self.GCN_2.initialize_parameters()
        self.A_mat = None

        self.bn = nn.BatchNorm1d(self.hidden_dim)

    def sample_gumbel(self, shape, eps=1e-20):
        U = torch.rand(shape)

        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature, device):
        y = logits + self.sample_gumbel(logits.size()).to(device=device)
        return torch.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, device, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = self.gumbel_softmax_sample(logits, temperature, device)

        if not hard:
            return y.view(-1, self.cluster_num)

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard

    def grasp_encoder(
        self,
        input: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode patient sequences with backbone + cluster-aware GCN.

        Args:
            input: tensor of shape [batch_size, seq_len, input_dim].
            static: optional static features [batch_size, static_dim].
            mask: optional mask [batch_size, seq_len].

        Returns:
            Tensor of shape [batch_size, hidden_dim].
        """
        if self.block == "ConCare":
            hidden_t, _ = self.backbone(input, mask=mask, static=static)
        else:
            _, hidden_t = self.backbone(input, mask)

        centers, codes = cluster(hidden_t, self.cluster_num, input.device)

        if self.A_mat is None:
            A_mat = np.eye(self.cluster_num)
        else:
            A_mat = kneighbors_graph(
                np.array(centers.detach().cpu().numpy()),
                20,
                mode="connectivity",
                include_self=False,
            ).toarray()

        adj_mat = torch.tensor(A_mat).to(device=input.device)

        e = self.relu(torch.matmul(hidden_t, centers.transpose(0, 1)))  # b clu_num

        scores = self.gumbel_softmax(e, temperature=1, device=input.device, hard=True)
        digits = torch.argmax(scores, dim=-1)  #  b

        h_prime = self.relu(self.GCN(adj_mat, centers, input.device))
        h_prime = self.relu(self.GCN_2(adj_mat, h_prime, input.device))

        clu_appendix = torch.matmul(scores, h_prime)

        weight1 = torch.sigmoid(self.weight1(clu_appendix))
        weight2 = torch.sigmoid(self.weight2(hidden_t))
        weight1 = weight1 / (weight1 + weight2)
        weight2 = 1 - weight1

        final_h = weight1 * clu_appendix + weight2 * hidden_t
        out = final_h
        return out

    def forward(
        self,
        x: torch.tensor,
        static: Optional[torch.tensor] = None,
        mask: Optional[torch.tensor] = None,
    ) -> torch.tensor:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input_dim].
            static: a tensor of shape [batch size, static_dim].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid.

        Returns:
            output: a tensor of shape [batch size, fusion_dim] representing the
                patient embedding.
        """
        # rnn will only apply dropout between layers
        out = self.grasp_encoder(x, static, mask)
        out = self.dropout(out)
        return out


class GRASP(BaseModel):
    """GRASP model.

    Paper: Liantao Ma et al. GRASP: generic framework for health status
        representation learning based on incorporating knowledge from
        similar patients. AAAI 2021.

    This model applies a separate GRASP layer for each feature, and then
    concatenates the outputs. The concatenated representations are fed into
    a fully connected layer to make predictions.

    The GRASP layer encodes patient sequences with a backbone (ConCare, GRU,
    or LSTM), clusters patients via k-means, refines cluster representations
    with a 2-layer GCN, and blends cluster-level knowledge back into
    individual patient representations via a learned gating mechanism.

    Args:
        dataset (SampleDataset): the dataset to train the model. It is used
            to query certain information such as the set of all tokens.
        static_key (str): optional key in samples to use as static features,
            e.g. "demographics". Only numerical static features are supported.
            Default is None.
        embedding_dim (int): the embedding dimension. Default is 128.
        hidden_dim (int): the hidden dimension. Default is 128.
        **kwargs: other parameters for the GRASPLayer
            (e.g., cluster_num, dropout, block).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86", "cond-80"],
        ...         "procedures": ["proc-12", "proc-45"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-12", "cond-52"],
        ...         "procedures": ["proc-23"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "sequence",
        ...         "procedures": "sequence",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test",
        ... )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = GRASP(
        ...     dataset=dataset,
        ...     embedding_dim=128,
        ...     hidden_dim=64,
        ...     cluster_num=2,
        ... )
        >>>
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(GRASP, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.static_key = static_key

        # validate kwargs for GRASP layer
        if "input_dim" in kwargs:
            raise ValueError("input_dim is determined by embedding_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Determine static feature dimension
        self.static_dim = 0
        if self.static_key is not None:
            first_sample = dataset[0]
            if self.static_key in first_sample:
                static_val = first_sample[self.static_key]
                if isinstance(static_val, torch.Tensor):
                    self.static_dim = (
                        static_val.shape[-1] if static_val.dim() > 0 else 1
                    )
                elif isinstance(static_val, (list, tuple)):
                    self.static_dim = len(static_val)
                else:
                    self.static_dim = 1

        # Dynamic feature keys (exclude static key)
        self.dynamic_feature_keys = [
            k for k in self.feature_keys if k != self.static_key
        ]

        # one GRASPLayer per dynamic feature
        self.grasp = nn.ModuleDict()
        for feature_key in self.dynamic_feature_keys:
            self.grasp[feature_key] = GRASPLayer(
                input_dim=embedding_dim,
                static_dim=self.static_dim,
                hidden_dim=hidden_dim,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(
            len(self.dynamic_feature_keys) * self.hidden_dim, output_size
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each
        patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient
                    embeddings if requested.
        """
        patient_emb = []
        embedded = self.embedding_model(kwargs)

        # Extract static features if configured
        static = None
        if self.static_key is not None and self.static_key in kwargs:
            static = kwargs[self.static_key]
            if isinstance(static, (list, tuple)):
                static = torch.tensor(static, dtype=torch.float)
            static = static.to(self.device)

        for feature_key in self.dynamic_feature_keys:
            x = embedded[feature_key]
            mask = (torch.abs(x).sum(dim=-1) != 0).int()
            x = self.grasp[feature_key](x, static=static, mask=mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": ["proc-12", "proc-45"],
            "label": 1,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "conditions": ["cond-12", "cond-52"],
            "procedures": ["proc-23"],
            "label": 0,
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "conditions": "sequence",
            "procedures": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = GRASP(
        dataset=dataset,
        embedding_dim=32,
        hidden_dim=32,
        cluster_num=2,
    )

    data_batch = next(iter(train_loader))
    ret = model(**data_batch)
    print(ret)

    ret["loss"].backward()
