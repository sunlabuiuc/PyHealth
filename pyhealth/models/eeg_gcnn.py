"""Shallow EEG-GCNN model wrapped for PyHealth 2.0.

Two GCN layers followed by a two-layer classification head.  Shallower than
the deep variant (4 GCN layers, 3-layer head) but otherwise identical in its
PyHealth integration.

Expects a SampleDataset built with:
    input_schema  = {"node_features": "tensor", "adj_matrix": "tensor"}
    output_schema = {"label": "binary"}

Key differences from the PyHealth 1.x version:
  - BaseModel.__init__ accepts only ``dataset``; feature_keys and label_keys
    are derived automatically from the dataset schemas.
  - Labels arrive from the DataLoader as (B, 1) float32 tensors (already
    processed by BinaryLabelProcessor) — no label tokenizer or prepare_labels
    step is needed.
  - get_output_size() and get_loss_function() are called without arguments in
    the 2.0 BaseModel API.
"""

from pyhealth.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, global_add_pool
from torch_geometric.utils import dense_to_sparse

NUM_NODES = 8


class EEGGraphConvNet(BaseModel):
    """Shallow EEG-GCNN model for PyHealth 2.0.

    Architecture:
        conv1: GCNConv(6  → 32)
        conv2: GCNConv(32 → 20) + BatchNorm
        global_add_pool → dropout(0.2) → fc1(20→10) → fc2(10→1)

    Args:
        dataset:           SampleDataset returned by EEGGCNNDataset.set_task().
        num_node_features: Number of PSD bands per node. Default: 6.
    """

    def __init__(self, dataset, num_node_features: int = 6, **kwargs):
        super().__init__(dataset=dataset, **kwargs)

        output_size = self.get_output_size()   # 1 for binary

        # Shallow GCN backbone
        self.conv1 = GCNConv(
            num_node_features, 32, improved=True, cached=False, normalize=False
        )
        self.conv2 = GCNConv(32, 20, improved=True, cached=False, normalize=False)
        self.conv2_bn = BatchNorm(20, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)

        # Classification head
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, output_size)

        # Xavier initialization for fully connected layers
        for fc in [self.fc1, self.fc2]:
            nn.init.xavier_normal_(fc.weight, gain=1)

    def _build_graph_batch(self, node_features_list, adj_matrix_list):
        """Convert a list of per-sample dense graphs into a single batched PyG graph.

        Returns (x, edge_index, edge_weight, batch) all on self.device.
        """
        all_edge_index  = []
        all_edge_weight = []
        all_x  = []
        batch  = []

        offset = 0
        for i, (x, adj) in enumerate(zip(node_features_list, adj_matrix_list)):
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            if not isinstance(adj, torch.Tensor):
                adj = torch.tensor(adj, dtype=torch.float32)
            x   = x.float()
            adj = adj.float()

            ei, ew = dense_to_sparse(adj)
            all_edge_index.append(ei + offset)
            all_edge_weight.append(ew)
            all_x.append(x)
            batch.extend([i] * x.shape[0])
            offset += x.shape[0]

        edge_index  = torch.cat(all_edge_index, dim=1)
        edge_weight = torch.cat(all_edge_weight)
        x           = torch.cat(all_x, dim=0)
        batch       = torch.tensor(batch, dtype=torch.long)

        device = self.device
        return (
            x.to(device),
            edge_index.to(device),
            edge_weight.to(device),
            batch.to(device),
        )

    def forward(self, **kwargs):
        """Forward pass.

        Called by PyHealth's Trainer as ``model(**batch_dict)`` where batch_dict
        contains keys matching the dataset schemas.

        Expected kwargs
        ---------------
        node_features : (B, 8, 6) float tensor — stacked by DataLoader
        adj_matrix    : (B, 8, 8) float tensor
        label         : (B, 1)   float tensor — produced by BinaryLabelProcessor

        Returns
        -------
        dict with keys: loss, y_prob, y_true, logit
        """
        node_features = kwargs["node_features"]
        adj_matrices  = kwargs["adj_matrix"]
        label_key     = self.label_keys[0]   # "label"
        labels        = kwargs[label_key]

        # Unpack batch dimension into a list for _build_graph_batch
        if isinstance(node_features, torch.Tensor) and node_features.dim() == 3:
            node_features = list(node_features)
            adj_matrices  = list(adj_matrices)

        # Build batched PyG graph
        x, edge_index, edge_weight, batch = self._build_graph_batch(
            node_features, adj_matrices
        )

        # Shallow GCN forward pass
        x = F.leaky_relu(self.conv1(x, edge_index, edge_weight))
        x = F.leaky_relu(self.conv2_bn(self.conv2(x, edge_index, edge_weight)))

        # Global pooling: one vector per graph in the batch
        out = global_add_pool(x, batch=batch)

        # Classifier
        out    = F.dropout(out, p=0.2, training=self.training)
        out    = F.leaky_relu(self.fc1(out))
        logits = self.fc2(out)   # (B, 1)

        # Convert logits to probabilities for the positive class
        y_prob = self.prepare_y_prob(logits)

        y_true = labels.to(self.device)
        loss   = self.get_loss_function()(logits, y_true)
        return {
            "loss":   loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit":  logits,
        }
