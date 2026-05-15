"""Shallow EEG-GCNN model wrapped for PyHealth 2.0.

Two GCN layers followed by a two-layer classification head.  Shallower than
the deep variant (4 GCN layers, 3-layer head) but otherwise identical in its
PyHealth integration.

Expects a SampleDataset built with:
    input_schema  = {"node_features": "tensor", "adj_matrix": "tensor"}
    output_schema = {"label": "binary"}

Example:
    >>> from pyhealth.datasets import EEGGCNNDataset
    >>> from pyhealth.tasks import EEGGCNNClassification
    >>> from pyhealth.models import EEGGraphConvNet
    >>> dataset = EEGGCNNDataset(root="path/to/data")
    >>> sample_ds = dataset.set_task(EEGGCNNClassification())
    >>> model = EEGGraphConvNet(dataset=sample_ds)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool
from torch_geometric.utils import dense_to_sparse

from pyhealth.models.base_model import BaseModel


class EEGGraphConvNet(BaseModel):
    """Shallow EEG-GCNN model for PyHealth 2.0.

    Two GCNConv layers aggregate node features weighted by adjacency-matrix
    edge weights, followed by global pooling and a two-layer classifier.

    Architecture:
        conv1: GCNConv(6  → 32)
        conv2: GCNConv(32 → 20) + BatchNorm
        global_add_pool → dropout(0.2) → fc1(20→10) → fc2(10→1)

    Attributes:
        conv1: First graph convolutional layer.
        conv2: Second graph convolutional layer.
        conv2_bn: Batch normalisation applied after conv2.
        fc1: First fully-connected classification layer.
        fc2: Output fully-connected layer.

    Args:
        dataset: SampleDataset returned by EEGGCNNDataset.set_task().
        num_node_features: Number of PSD bands per node. Defaults to 6.
        **kwargs: Additional keyword arguments forwarded to BaseModel.
    """

    def __init__(
        self,
        dataset,
        num_node_features: int = 6,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset, **kwargs)

        output_size = self.get_output_size()  # 1 for binary

        self.conv1 = GCNConv(
            num_node_features, 32, improved=True, cached=False, normalize=False
        )
        self.conv2 = GCNConv(32, 20, improved=True, cached=False, normalize=False)
        self.conv2_bn = BatchNorm(
            20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )

        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, output_size)

        for fc in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(fc.weight, gain=1)

    def _build_graph_batch(
        self,
        node_features_list: List[torch.Tensor],
        adj_matrix_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert per-sample dense graphs into a single batched PyG graph.

        Iterates over paired node-feature and adjacency-matrix tensors,
        converts each dense adjacency matrix to sparse (COO) format, offsets
        node indices for batching, and concatenates everything into four
        batch-level tensors ready for PyTorch Geometric.

        Args:
            node_features_list: Length-B list of (N, F) float tensors, one
                per sample, where N is the number of nodes and F the number
                of node features.
            adj_matrix_list: Length-B list of (N, N) float tensors containing
                the dense adjacency (edge-weight) matrices.

        Returns:
            A tuple of (x, edge_index, edge_weight, batch) where:
                x: (B*N, F) node feature matrix on self.device.
                edge_index: (2, E) edge index tensor on self.device.
                edge_weight: (E,) edge weight tensor on self.device.
                batch: (B*N,) batch assignment vector on self.device.
        """
        all_edge_index: List[torch.Tensor] = []
        all_edge_weight: List[torch.Tensor] = []
        all_x: List[torch.Tensor] = []
        batch: List[int] = []

        offset = 0
        for i, (x, adj) in enumerate(zip(node_features_list, adj_matrix_list)):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(adj, torch.Tensor):
                adj = torch.tensor(adj, dtype=torch.float32)
            x = x.float()
            adj = adj.float()

            ei, ew = dense_to_sparse(adj)
            all_edge_index.append(ei + offset)
            all_edge_weight.append(ew)
            all_x.append(x)
            batch.extend([i] * x.shape[0])
            offset += x.shape[0]

        edge_index = torch.cat(all_edge_index, dim=1)
        edge_weight = torch.cat(all_edge_weight)
        x = torch.cat(all_x, dim=0)
        batch_tensor = torch.tensor(batch, dtype=torch.long)

        device = self.device
        return (
            x.to(device),
            edge_index.to(device),
            edge_weight.to(device),
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

        x, edge_index, edge_weight, batch = self._build_graph_batch(
            node_features, adj_matrices
        )

        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x, edge_index, edge_weight)))

        out = global_add_pool(x, batch=batch)

        out = F.dropout(out, p=0.2, training=self.training)
        out = F.leaky_relu(self.fc1(out))
        logits = self.fc2(out)

        y_prob = self.prepare_y_prob(logits)
        y_true = labels.to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
