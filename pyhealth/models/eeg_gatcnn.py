"""Shallow EEG-GAT model wrapped for PyHealth 2.0.

Two GATConv layers followed by a two-layer classification head. 

Expects a SampleDataset built with:
    input_schema  = {"node_features": "tensor", "adj_matrix": "tensor"}
    output_schema = {"label": "binary"}

Example:
    >>> from pyhealth.datasets import EEGGCNNDataset
    >>> from pyhealth.tasks import EEGGCNNClassification
    >>> from pyhealth.models import EEGGATConvNet
    >>> dataset = EEGGCNNDataset(root="path/to/data")
    >>> sample_ds = dataset.set_task(EEGGCNNClassification())
    >>> model = EEGGATConvNet(dataset=sample_ds, dropout=0.2)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm, global_add_pool
from torch_geometric.utils import dense_to_sparse

from pyhealth.models.base_model import BaseModel


class EEGGATConvNet(BaseModel):
    """Shallow EEG-GAT model for PyHealth 2.0.

    Two GATConv layers use multi-head attention over the electrode graph to
    aggregate node features, with adjacency-matrix edge weights biasing the
    attention scores.  Global pooling and a two-layer classifier follow.

    Architecture (input → output features per node):
        conv1: GATConv(6  → 16, heads=4, concat=True)  →  64
        conv2: GATConv(64 → 20, heads=1, concat=False) + BatchNorm  →  20
        global_add_pool → dropout → fc1(20→10) → fc2(10→1)

    Attributes:
        dropout: Dropout probability applied before the classifier.
        conv1: First graph attention layer.
        conv2: Second graph attention layer.
        conv2_bn: Batch normalisation applied after conv2.
        fc1: First fully-connected classification layer.
        fc2: Output fully-connected layer.

    Args:
        dataset: SampleDataset returned by EEGGCNNDataset.set_task().
        num_node_features: Number of PSD bands per node. Defaults to 6.
        dropout: Dropout probability applied before the classifier.
            Defaults to 0.2.
        **kwargs: Additional keyword arguments forwarded to BaseModel.
    """

    def __init__(
        self,
        dataset,
        num_node_features: int = 6,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset, **kwargs)
        self.dropout = dropout

        output_size = self.get_output_size()  # 1 for binary

        self.conv1 = GATConv(
            num_node_features, 16,
            heads=4, concat=True, negative_slope=0.2,
            dropout=0.0, add_self_loops=False, bias=True, edge_dim=1,
        )
        self.conv2 = GATConv(
            64, 20,
            heads=1, concat=False, negative_slope=0.2,
            dropout=0.0, add_self_loops=False, bias=True, edge_dim=1,
        )
        self.conv2_bn = BatchNorm(
            20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, output_size)

        for fc in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(fc.weight, gain=1)

        self.double()

    def _build_graph_batch(
        self,
        node_features_list: List[torch.Tensor],
        adj_matrix_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert per-sample dense graphs into a single batched PyG graph.

        Iterates over paired node-feature and adjacency-matrix tensors,
        converts each dense adjacency matrix to sparse (COO) format, reshapes
        edge weights to (E, 1) for GATConv's edge_dim=1, offsets node indices
        for batching, and concatenates everything into four batch-level tensors.

        Args:
            node_features_list: Length-B list of (N, F) float64 tensors, one
                per sample, where N is the number of nodes and F the number
                of node features.
            adj_matrix_list: Length-B list of (N, N) float64 tensors containing
                the dense adjacency (edge-weight) matrices.

        Returns:
            A tuple of (x, edge_index, edge_attr, batch) where:
                x: (B*N, F) node feature matrix on self.device.
                edge_index: (2, E) edge index tensor on self.device.
                edge_attr: (E, 1) edge attribute tensor on self.device.
                batch: (B*N,) batch assignment vector on self.device.
        """
        all_edge_index: List[torch.Tensor] = []
        all_edge_attr: List[torch.Tensor] = []
        all_x: List[torch.Tensor] = []
        batch: List[int] = []

        offset = 0
        for i, (x, adj) in enumerate(zip(node_features_list, adj_matrix_list)):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float64)
            if not isinstance(adj, torch.Tensor):
                adj = torch.tensor(adj, dtype=torch.float64)
            x = x.double()
            adj = adj.double()

            ei, ew = dense_to_sparse(adj)       # edge weights from adjacency matrix, shape: (E,)
            all_edge_index.append(ei + offset)
            all_edge_attr.append(ew.unsqueeze(-1))  # (E, 1) for edge_dim=1
            all_x.append(x)
            batch.extend([i] * x.shape[0])
            offset += x.shape[0]

        edge_index = torch.cat(all_edge_index, dim=1)
        edge_attr = torch.cat(all_edge_attr, dim=0)
        x = torch.cat(all_x, dim=0)
        batch_tensor = torch.tensor(batch, dtype=torch.long)

        device = self.device
        return (
            x.to(device),
            edge_index.to(device),
            edge_attr.to(device),
            batch_tensor.to(device),
        )

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Run a forward pass through the model.

        Called by PyHealth's Trainer as ``model(**batch_dict)`` where
        batch_dict contains keys matching the dataset schemas.

        Args:
            **kwargs: Batch dictionary produced by the DataLoader. Expected
                keys are:
                    node_features: (B, 8, 6) float tensor of PSD features.
                    adj_matrix: (B, 8, 8) float tensor of edge weights.
                    label: (B, 1) float tensor of binary labels produced by
                        BinaryLabelProcessor.

        Returns:
            A dictionary with keys:
                loss: Scalar training loss.
                y_prob: (B, 1) sigmoid probabilities for the positive class.
                y_true: (B, 1) ground-truth labels.
                logit: (B, 1) raw logits.
        """
        node_features = kwargs["node_features"]
        adj_matrices = kwargs["adj_matrix"]
        label_key = self.label_keys[0]
        labels = kwargs[label_key]

        if isinstance(node_features, torch.Tensor) and node_features.dim() == 3:
            node_features = list(node_features)
            adj_matrices = list(adj_matrices)

        x, edge_index, edge_attr, batch = self._build_graph_batch(
            node_features, adj_matrices
        )

        x = F.leaky_relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x, edge_index, edge_attr=edge_attr)))

        out = global_add_pool(x, batch=batch)

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.leaky_relu(self.fc1(out))
        logits = self.fc2(out)

        y_prob = self.prepare_y_prob(logits)
        y_true = labels.to(self.device).double()
        loss = self.get_loss_function()(logits, y_true)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
