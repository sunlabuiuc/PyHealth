import copy
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sklearn.neighbors import kneighbors_graph

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.concare import ConCareLayer
from pyhealth.models.embedding import EmbeddingModel
from pyhealth.models.rnn import RNNLayer
from pyhealth.models.utils import get_last_visit


def random_init(dataset, num_centers, device):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    # print("random size", dataset.size())
    # print("numcenter", num_centers)

    indices = torch.tensor(
        np.array(random.sample(range(num_points), k=num_centers)), dtype=torch.long
    )

    centers = torch.gather(
        dataset, 0, indices.view(-1, 1).expand(-1, dimension).to(device=device)
    )
    return centers


# Compute for each data point the closest center
def compute_codes(dataset, centers):
    num_points = dataset.size(0)
    dimension = dataset.size(1)
    num_centers = centers.size(0)

    # print("size:", dataset.size(), centers.size())
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


# Compute new centers as means of the data points forming the clusters
def update_centers(dataset, codes, num_centers, device):
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


def cluster(dataset, num_centers, device):
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
    def __init__(self, in_features, out_features, bias=True):
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
        block: the backbone model used in the GRASP layer ('ConCare', 'LSTM' or 'GRU'), default 'ConCare'.

    Examples:
        >>> from pyhealth.models import GRASPLayer
        >>> input = torch.randn(3, 128, 64)  # [batch size, sequence len, feature_size]
        >>> layer = GRASPLayer(64, cluster_num=2)
        >>> c = layer(input)
        >>> c.shape
        torch.Size([3, 128])
    """

    def __init__(
        self,
        input_dim: int,
        static_dim: int = 0,
        hidden_dim: int = 128,
        cluster_num: int = 2,
        dropout: int = 0.5,
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

    def grasp_encoder(self, input, static=None, mask=None):

        if self.block == "ConCare":
            hidden_t, _ = self.backbone(input, mask=mask, static=static)
        else:
            _, hidden_t = self.backbone(input, mask)
        hidden_t = torch.squeeze(hidden_t, 0)

        centers, codes = cluster(hidden_t, self.cluster_num, input.device)

        if self.A_mat == None:
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
    """GRASP model for EHR-based prediction tasks.

    GRASP (Generic framework for health status Representation learning
    bAsed on incorporating knowledge from Similar Patients) uses graph-based
    clustering to capture patient similarity and enhance temporal modeling.

    Paper: Liantao Ma et al. GRASP: generic framework for health status
        representation learning based on incorporating knowledge from
        similar patients. AAAI 2021.

    Note:
        We use separate GRASP layers for different feature_keys.
        The model automatically handles different input formats through the
        EmbeddingModel.

    Args:
        dataset: The dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        static_key: The key in samples to use as static features, e.g.
            "demographics". Default is None. Only numerical static features
            are supported.
        embedding_dim: The embedding dimension. Default is 128.
        hidden_dim: The hidden dimension. Default is 128.
        **kwargs: Other parameters for the GRASP layer (cluster_num, block,
            dropout).

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "list_codes": ["505800458", "50580045810", "50580045811"],
        ...         "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
        ...         "demographic": [0.0, 2.0, 1.5],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-1",
        ...         "list_codes": ["55154191800", "551541928", "55154192800"],
        ...         "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7]],
        ...         "demographic": [0.0, 2.0, 1.5],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = SampleDataset(
        ...     samples=samples,
        ...     input_schema={"list_codes": "sequence", "list_vectors": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>> from pyhealth.models import GRASP
        >>> model = GRASP(
        ...     dataset=dataset,
        ...     static_key="demographic",
        ...     embedding_dim=64,
        ...     hidden_dim=64,
        ...     cluster_num=2,
        ... )
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>> ret = model(**data_batch)
        >>> print(ret["loss"])
        tensor(..., grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        static_key: Optional[str] = None,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(GRASP, self).__init__(dataset=dataset)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.static_key = static_key

        # validate kwargs for GRASP layer
        if "feature_size" in kwargs:
            raise ValueError("feature_size is determined by embedding_dim")

        cluster_num = kwargs.get("cluster_num", 12)
        if len(dataset) < cluster_num:
            raise ValueError(
                f"cluster_num ({cluster_num}) must be no larger than "
                f"dataset size ({len(dataset)})"
            )

        assert len(self.label_keys) == 1, (
            "Only one label key is supported for GRASP"
        )
        self.label_key = self.label_keys[0]

        # EmbeddingModel handles all feature embedding automatically
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Determine static dimension
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

        # Get dynamic feature keys (excluding static key)
        self.dynamic_feature_keys = [
            k for k in self.feature_keys
            if k != self.static_key
        ]

        # GRASP layers for each dynamic feature
        self.grasp = nn.ModuleDict()
        for feature_key in self.dynamic_feature_keys:
            self.grasp[feature_key] = GRASPLayer(
                input_dim=embedding_dim,
                static_dim=self.static_dim,
                hidden_dim=self.hidden_dim,
                **kwargs,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(
            len(self.dynamic_feature_keys) * self.hidden_dim, output_size
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the final loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
        """
        patient_emb = []

        embedded, masks = self.embedding_model(kwargs, output_mask=True)

        # Get static features if available
        static = None
        if self.static_key is not None and self.static_key in kwargs:
            static_data = kwargs[self.static_key]
            if isinstance(static_data, torch.Tensor):
                static = static_data.float().to(self.device)
            else:
                static = torch.tensor(
                    static_data, dtype=torch.float, device=self.device
                )

        for feature_key in self.dynamic_feature_keys:
            x = embedded[feature_key]
            mask = masks[feature_key]
            x = self.grasp[feature_key](x, static=static, mask=mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        # Compute loss and predictions
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "list_codes": ["505800458", "50580045810", "50580045811"],
            "list_vectors": [[1.0, 2.55, 3.4], [4.1, 5.5, 6.0]],
            "list_list_codes": [["A05B", "A05C", "A06A"], ["A11D", "A11E"]],
            "label": 1,
            "demographic": [1.0, 2.0, 1.3],
        },
        {
            "patient_id": "patient-0",
            "visit_id": "visit-1",
            "list_codes": [
                "55154191800",
                "551541928",
                "55154192800",
                "705182798",
                "70518279800",
            ],
            "list_vectors": [[1.4, 3.2, 3.5], [4.1, 5.9, 1.7], [4.5, 5.9, 1.7]],
            "list_list_codes": [["A04A", "B035", "C129"]],
            "label": 0,
            "demographic": [1.0, 2.0, 1.3],
        },
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={
            "list_codes": "sequence",
            "list_vectors": "sequence",
            "list_list_codes": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="test",
    )

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    model = GRASP(
        dataset=dataset,
        static_key="demographic",
        embedding_dim=64,
        hidden_dim=64,
        cluster_num=2,
    )

    data_batch = next(iter(train_loader))
    ret = model(**data_batch)
    print(ret)
    ret["loss"].backward()
