"""DearLLM: LLM-deduced feature correlations as a patient graph prior.

Contributor: Kavish Shah (GitHub: thekavishshah)

Paper: Yongxin Xu, Xinke Jiang, Xu Chu, Rihong Qiu, Yujie Feng, Hongxin
Ding, Junfeng Zhao, Yasha Wang, Bing Xie. DearLLM: Enhancing Personalized
Healthcare via Large Language Models-Deduced Feature Correlations. AAAI 2025,
pages 941-949. https://doi.org/10.1609/aaai.v39i1.32079

Official code: https://github.com/Artessay/DearLLM (no license file; this is
an independent re-implementation for PyHealth 2.x, written from the paper and
checked against the official code's behaviour).

Description:
    DearLLM builds one small directed graph per patient. Its nodes are the
    patient's diagnosis codes. The weight of the edge ``c_i -> c_j`` comes
    from a medical LLM: given the patient's visit history and the text
    "<c_i> is related to", the conditional perplexity of the LLM on the name
    of ``c_j`` is low when the two codes are strongly related. Perplexities
    are clipped to percentile bounds and mapped to ``[0, 1]`` (low perplexity
    becomes a high weight); see
    :func:`pyhealth.tasks.dearllm_mortality.normalize_perplexity_graphs`.

    The model has three parts:

    1. An EHR encoder (GRU, LSTM or Transformer) over the visit sequence,
       giving the patient vector ``h_T``.
    2. A graph encoder: code embeddings, then ``L`` GCN layers over the
       weighted patient graph (:class:`DearLLMGraphEncoder`).
    3. Feature-frequency-aware pooling: the node vectors are summed with
       weights ``softmax(CF-IPF)``, where CF-IPF is a TF-IDF style score
       that is high for codes frequent in this patient but rare in the
       training set. This gives the graph vector ``h_s``.

    The prediction is ``Linear(dropout([h_T, h_s]))``.

    The LLM step is an *input* to this model, not part of it: the graph is
    precomputed and passed in through three sample fields (see
    :class:`DearLLM`). This keeps training cheap and testable offline.
"""

import itertools
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from ..processors import NestedSequenceProcessor, SequenceProcessor
from .base_model import BaseModel
from .rnn import RNNLayer
from .transformer import TransformerLayer


class DearLLMGraphEncoder(nn.Module):
    """Batched GCN encoder with feature-frequency-aware pooling.

    Works on dense, zero-padded batches so it needs no graph library. For
    each patient ``b`` the weighted adjacency is ``A[b, i, j] = w`` for an
    edge ``i -> j``. Each layer applies, in this order (as in the official
    code): ``BatchNorm`` over the real (non-padding) nodes, then a GCN
    convolution with symmetric degree normalisation, then ``ReLU``:

    ``H' = ReLU(A_hat^T (BN(H) W) + b)`` with
    ``A_hat = D^{-1/2} A D^{-1/2}`` and ``D`` the weighted in-degree.

    Messages flow from source to target, which matches
    ``torch_geometric.nn.GCNConv`` on a directed edge list. With
    ``add_self_loops=True``, ``A`` is replaced by ``A + I`` first, as in the
    paper's Eq. 4; the official code uses ``add_self_loops=False``.

    The graph vector is ``sum_i pi_i H_i`` with
    ``pi = softmax(node_weights)`` over the real nodes (``pooling="cfipf"``),
    or the plain mean of the real nodes (``pooling="mean"``, the paper's
    DearLLM_f- ablation).

    Args:
        num_nodes: Size of the node vocabulary, including padding index 0.
        node_dim: Size of the node (code) embedding. Default is 64.
        hidden_dims: Output size of each GCN layer. Default is (64, 64).
        add_self_loops: Whether to add self-loops before normalising.
            Default is False (official code).
        pooling: ``"cfipf"`` (paper) or ``"mean"``. Default is ``"cfipf"``.

    Examples:
        >>> import torch
        >>> encoder = DearLLMGraphEncoder(num_nodes=6, node_dim=4,
        ...                               hidden_dims=(4, 4))
        >>> node_ids = torch.tensor([[2, 3, 4], [5, 0, 0]])
        >>> node_weights = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.0]])
        >>> edges = torch.tensor([[[0.0, 1.0, 0.5], [1.0, 2.0, 0.9]],
        ...                       [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        >>> graph_emb, pi = encoder(node_ids, node_weights, edges)
        >>> graph_emb.shape, pi.shape
        (torch.Size([2, 4]), torch.Size([2, 3]))
    """

    def __init__(
        self,
        num_nodes: int,
        node_dim: int = 64,
        hidden_dims: Sequence[int] = (64, 64),
        add_self_loops: bool = False,
        pooling: str = "cfipf",
    ):
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims needs at least one GCN layer.")
        if pooling not in ("cfipf", "mean"):
            raise ValueError(f"pooling must be 'cfipf' or 'mean', got {pooling!r}")
        self.add_self_loops = add_self_loops
        self.pooling = pooling
        self.output_dim = hidden_dims[-1]

        self.node_embedding = nn.Embedding(num_nodes, node_dim, padding_idx=0)
        dims = [node_dim] + list(hidden_dims)
        self.norms = nn.ModuleList(nn.BatchNorm1d(d) for d in dims[:-1])
        # GCN layer = linear map without bias, propagation, then bias
        # (the same order as GCNConv).
        self.convs = nn.ModuleList(
            nn.Linear(d_in, d_out, bias=False)
            for d_in, d_out in itertools.pairwise(dims)
        )
        self.conv_biases = nn.ParameterList(
            nn.Parameter(torch.zeros(d_out)) for d_out in dims[1:]
        )
        self.relu = nn.ReLU()

    @staticmethod
    def build_adjacency(edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Turns padded edge rows into a dense weighted adjacency matrix.

        Args:
            edges: Float tensor of shape ``(batch, num_edges, 3)``; each row
                is ``[source, target, weight]`` with node positions local to
                the patient. Zero rows (padding) add nothing.
            num_nodes: Number of node slots ``N`` in the batch.

        Returns:
            Tensor of shape ``(batch, N, N)`` with ``A[b, i, j]`` the summed
            weight of edges ``i -> j``.

        Examples:
            >>> import torch
            >>> edges = torch.tensor([[[0.0, 1.0, 0.5]]])
            >>> DearLLMGraphEncoder.build_adjacency(edges, 2)
            tensor([[[0.0000, 0.5000],
                     [0.0000, 0.0000]]])
        """
        batch_size = edges.shape[0]
        src = edges[..., 0].long()
        dst = edges[..., 1].long()
        weight = edges[..., 2].to(torch.float32)
        if src.numel() > 0 and (
            src.min() < 0
            or dst.min() < 0
            or src.max() >= num_nodes
            or dst.max() >= num_nodes
        ):
            raise ValueError(
                "Edge endpoints must index the patient's node list "
                f"(0 <= index < {num_nodes})."
            )
        adj = edges.new_zeros(batch_size, num_nodes, num_nodes, dtype=torch.float32)
        batch_idx = torch.arange(batch_size, device=edges.device)
        batch_idx = batch_idx.unsqueeze(1).expand_as(src)
        adj.index_put_((batch_idx, src, dst), weight, accumulate=True)
        return adj

    def normalize_adjacency(
        self, adj: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """Symmetric GCN normalisation ``D^{-1/2} A D^{-1/2}``.

        Args:
            adj: Weighted adjacency of shape ``(batch, N, N)``.
            node_mask: Bool tensor ``(batch, N)``, True for real nodes.

        Returns:
            Normalised adjacency of the same shape. Nodes with zero
            in-degree get zero rows and columns.

        Examples:
            >>> import torch
            >>> encoder = DearLLMGraphEncoder(num_nodes=3, node_dim=2,
            ...                               hidden_dims=(2,))
            >>> adj = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
            >>> encoder.normalize_adjacency(adj, torch.tensor([[True, True]]))
            tensor([[[0., 1.],
                     [1., 0.]]])
        """
        if self.add_self_loops:
            adj = adj + torch.diag_embed(node_mask.to(adj.dtype))
        # GCNConv uses the weighted in-degree of every node for both ends.
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(torch.isinf(deg_inv_sqrt), 0.0)
        return deg_inv_sqrt.unsqueeze(2) * adj * deg_inv_sqrt.unsqueeze(1)

    def _masked_batch_norm(
        self, norm: nn.BatchNorm1d, x: torch.Tensor, node_mask: torch.Tensor
    ) -> torch.Tensor:
        """BatchNorm over real nodes only, so padding does not skew stats."""
        valid = x[node_mask]
        if valid.shape[0] == 0:
            return x
        if self.training and valid.shape[0] < 2:
            # BatchNorm cannot estimate a variance from one value; fall back
            # to the running statistics for this batch.
            valid = F.batch_norm(
                valid,
                norm.running_mean,
                norm.running_var,
                norm.weight,
                norm.bias,
                training=False,
                eps=norm.eps,
            )
        else:
            valid = norm(valid)
        out = torch.zeros_like(x)
        out[node_mask] = valid
        return out

    def forward(
        self,
        node_ids: torch.Tensor,
        node_weights: torch.Tensor,
        edges: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encodes a batch of patient graphs.

        Args:
            node_ids: Long tensor ``(batch, N)`` of node vocabulary indices,
                0 for padding.
            node_weights: Float tensor ``(batch, N)`` of CF-IPF scores.
            edges: Float tensor ``(batch, E, 3)`` of ``[source, target,
                weight]`` rows (see :meth:`build_adjacency`).

        Returns:
            A tuple ``(graph_emb, pi)``: the graph vectors of shape
            ``(batch, output_dim)`` and the pooling weights of shape
            ``(batch, N)`` (zero on padding; all zero for an empty graph).

        Examples:
            >>> import torch
            >>> encoder = DearLLMGraphEncoder(num_nodes=4, node_dim=4,
            ...                               hidden_dims=(4,))
            >>> emb, pi = encoder(torch.tensor([[1, 2]]),
            ...                   torch.tensor([[0.5, 0.5]]),
            ...                   torch.tensor([[[0.0, 1.0, 1.0]]]))
            >>> pi
            tensor([[0.5000, 0.5000]])
        """
        node_mask = node_ids != 0
        num_nodes = node_ids.shape[1]
        if edges.dim() != 3 or edges.shape[-1] != 3:
            raise ValueError(
                "edges must have shape (batch, num_edges, 3), got "
                f"{tuple(edges.shape)}."
            )
        adj = self.build_adjacency(edges, num_nodes)
        adj = adj * node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        a_hat = self.normalize_adjacency(adj, node_mask)

        x = self.node_embedding(node_ids)
        mask_f = node_mask.unsqueeze(-1).to(x.dtype)
        for norm, conv, bias in zip(self.norms, self.convs, self.conv_biases):
            x = self._masked_batch_norm(norm, x, node_mask)
            # out[j] = sum_i a_hat[i, j] * (x W)[i]: messages go source -> target
            x = torch.bmm(a_hat.transpose(1, 2), conv(x)) + bias
            x = self.relu(x) * mask_f

        if self.pooling == "cfipf":
            scores = node_weights.to(x.dtype).masked_fill(~node_mask, float("-inf"))
            pi = torch.softmax(scores, dim=1)
        else:
            pi = mask_f.squeeze(-1) / mask_f.sum(dim=1).clamp(min=1.0)
        # A patient without nodes gives softmax over all -inf = NaN; use 0.
        pi = torch.nan_to_num(pi, nan=0.0)
        graph_emb = torch.bmm(pi.unsqueeze(1), x).squeeze(1)
        return graph_emb, pi


class DearLLM(BaseModel):
    """DearLLM model (Xu et al., AAAI 2025).

    Combines an EHR sequence encoder with a GCN over a per-patient graph of
    LLM-deduced code correlations, pooled with CF-IPF weights. See the module
    docstring for the method.

    Inputs. The dataset's ``input_schema`` must contain the three graph
    fields below; every other input field is an EHR feature (``"sequence"``
    or ``"nested_sequence"`` of codes) and gets its own encoder:

    * ``node_key`` (``"sequence"``): the patient's distinct codes, i.e. the
      graph nodes. Their order defines the node positions used in edges.
    * ``node_weight_key`` (``"tensor"``): one CF-IPF score per node.
    * ``edge_key`` (``"tensor"``): a list of ``[source, target, weight]``
      rows. A patient without edges must still have one row; use the
      zero-weight row ``[0, 0, 0.0]``, which has no effect.

    :class:`~pyhealth.tasks.DearLLMMortalityPredictionMIMIC3` and
    :class:`~pyhealth.tasks.DearLLMMortalityPredictionMIMIC4` produce these
    fields.

    Training. The paper trains the graph convolutions with a smaller
    learning rate than the rest; :meth:`parameter_groups` returns optimizer
    groups for that.

    Args:
        dataset: The dataset to train the model on.
        embedding_dim: Size of the EHR code embeddings. Default is 128.
        hidden_dim: Hidden size of the GRU/LSTM encoder (the Transformer
            keeps ``embedding_dim``). Default is 128.
        backbone: EHR encoder, one of ``"gru"``, ``"lstm"`` or
            ``"transformer"``. Default is ``"gru"``.
        num_layers: Number of encoder layers. Default is 1.
        heads: Attention heads of the Transformer encoder. Default is 4.
        node_dim: Size of the graph node embeddings. Default is 64.
        graph_hidden_dims: Output size of each GCN layer. Default is
            ``(64, 64)``.
        add_self_loops: Add self-loops in the GCN (paper Eq. 4: True;
            official code: False). Default is False.
        pooling: ``"cfipf"`` (paper) or ``"mean"`` (ablation). Default is
            ``"cfipf"``.
        dropout: Dropout before the classifier and inside the EHR encoder.
            Default is 0.5.
        node_key: Name of the node field. Default is ``"graph_nodes"``.
        node_weight_key: Name of the CF-IPF field. Default is
            ``"graph_node_weights"``.
        edge_key: Name of the edge field. Default is ``"graph_edges"``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "conditions": [["a", "b"], ["b", "c"]],
        ...         "graph_nodes": ["a", "b", "c"],
        ...         "graph_node_weights": [0.1, 0.2, 0.3],
        ...         "graph_edges": [[0, 1, 0.5], [1, 2, 0.9]],
        ...         "mortality": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "conditions": [["d"]],
        ...         "graph_nodes": ["d"],
        ...         "graph_node_weights": [0.4],
        ...         "graph_edges": [[0, 0, 0.0]],
        ...         "mortality": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "nested_sequence",
        ...         "graph_nodes": "sequence",
        ...         "graph_node_weights": "tensor",
        ...         "graph_edges": "tensor",
        ...     },
        ...     output_schema={"mortality": "binary"},
        ... )
        >>> model = DearLLM(dataset, embedding_dim=8, hidden_dim=8,
        ...                 node_dim=4, graph_hidden_dims=(4, 4))
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> out = model(**batch)
        >>> sorted(out.keys())
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        backbone: str = "gru",
        num_layers: int = 1,
        heads: int = 4,
        node_dim: int = 64,
        graph_hidden_dims: Sequence[int] = (64, 64),
        add_self_loops: bool = False,
        pooling: str = "cfipf",
        dropout: float = 0.5,
        node_key: str = "graph_nodes",
        node_weight_key: str = "graph_node_weights",
        edge_key: str = "graph_edges",
    ):
        super().__init__(dataset=dataset)
        if len(self.label_keys) != 1:
            raise ValueError("DearLLM supports exactly one label key.")
        self.label_key = self.label_keys[0]
        backbone = backbone.lower()
        if backbone not in ("gru", "lstm", "transformer"):
            raise ValueError(
                f"backbone must be 'gru', 'lstm' or 'transformer', got {backbone!r}"
            )
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.node_key = node_key
        self.node_weight_key = node_weight_key
        self.edge_key = edge_key

        processors = self.dataset.input_processors
        for key in (node_key, node_weight_key, edge_key):
            if key not in processors:
                raise ValueError(
                    f"DearLLM needs the graph field {key!r} in the input schema."
                )
        if not isinstance(processors[node_key], SequenceProcessor):
            raise TypeError(f"{node_key!r} must use the 'sequence' processor.")

        graph_keys = {node_key, node_weight_key, edge_key}
        self.ehr_keys: list[str] = [k for k in self.feature_keys if k not in graph_keys]
        if not self.ehr_keys:
            raise ValueError("DearLLM needs at least one EHR feature field.")

        # EHR encoder: one embedding table and one sequence encoder per feature.
        self.embeddings = nn.ModuleDict()
        self.encoders = nn.ModuleDict()
        for key in self.ehr_keys:
            processor = processors[key]
            if not isinstance(processor, (SequenceProcessor, NestedSequenceProcessor)):
                raise TypeError(
                    f"EHR feature {key!r} must use 'sequence' or 'nested_sequence'."
                )
            self.embeddings[key] = nn.Embedding(
                len(processor.code_vocab), embedding_dim, padding_idx=0
            )
            if backbone == "transformer":
                self.encoders[key] = TransformerLayer(
                    embedding_dim, heads=heads, dropout=dropout, num_layers=num_layers
                )
            else:
                self.encoders[key] = RNNLayer(
                    embedding_dim,
                    hidden_dim,
                    rnn_type=backbone.upper(),
                    num_layers=num_layers,
                    dropout=dropout,
                )
        encoder_dim = embedding_dim if backbone == "transformer" else hidden_dim
        patient_dim = len(self.ehr_keys) * encoder_dim

        self.graph_encoder = DearLLMGraphEncoder(
            num_nodes=len(processors[node_key].code_vocab),
            node_dim=node_dim,
            hidden_dims=graph_hidden_dims,
            add_self_loops=add_self_loops,
            pooling=pooling,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(
            patient_dim + self.graph_encoder.output_dim,
            self.label_processor.output_size(),
        )

    def _encode_feature(self, key: str, codes: torch.Tensor) -> torch.Tensor:
        """Encodes one code feature into a ``(batch, encoder_dim)`` vector."""
        codes = codes.to(self.device)
        x = self.embeddings[key](codes)
        if codes.dim() == 3:
            # nested_sequence (batch, visits, codes): sum the codes of a visit.
            x = x.sum(dim=2)
            mask = (codes != 0).any(dim=2)
        else:
            mask = codes != 0
        # Keep at least one step per patient: an all-padding row would give
        # a zero-length sequence (GRU/LSTM) or an all-masked attention row.
        mask[:, 0] = True
        # RNNLayer returns the last valid step, TransformerLayer the first.
        _, patient_vec = self.encoders[key](x, mask.int())
        return patient_vec

    def parameter_groups(
        self, lr: float = 1e-3, graph_lr: float = 1e-4, weight_decay: float = 1e-4
    ) -> list[dict]:
        """Optimizer parameter groups with a separate GCN learning rate.

        The official code trains the GCN layers with ``graph_lr`` and all
        other parameters with ``lr``; biases and normalisation weights get no
        weight decay.

        Args:
            lr: Learning rate for everything except the GCN layers.
            graph_lr: Learning rate for the GCN layers.
            weight_decay: L2 penalty (the paper's ``eta``).

        Returns:
            A list of dicts to pass to a ``torch.optim`` optimizer.

        Examples:
            >>> import torch
            >>> groups = model.parameter_groups(lr=1e-3, graph_lr=1e-4)
            >>> optimizer = torch.optim.Adam(groups)
        """
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
            (g, d): [] for g in (True, False) for d in (True, False)
        }
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_graph = name.startswith("graph_encoder.conv")
            decay = param.dim() > 1
            groups[(is_graph, decay)].append(param)
        return [
            {
                "params": params,
                "lr": graph_lr if is_graph else lr,
                "weight_decay": weight_decay if decay else 0.0,
            }
            for (is_graph, decay), params in groups.items()
            if params
        ]

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **kwargs: A batch from :func:`~pyhealth.datasets.get_dataloader`:
                the EHR features, the three graph fields and (optionally) the
                label. Pass ``embed=True`` to also return the fused patient
                vector and the node pooling weights.

        Returns:
            A dict with ``logit`` and ``y_prob``; ``loss`` and ``y_true`` if
            the label is given; ``embed`` and ``node_weights`` if
            ``embed=True``.

        Examples:
            >>> out = model(**batch)
            >>> out["y_prob"].shape
            torch.Size([2, 1])
        """
        patient_vecs = [
            self._encode_feature(key, self._value(kwargs[key])) for key in self.ehr_keys
        ]
        graph_vec, node_weights = self.graph_encoder(
            self._value(kwargs[self.node_key]).to(self.device),
            self._value(kwargs[self.node_weight_key]).to(self.device),
            self._value(kwargs[self.edge_key]).to(self.device),
        )
        embed = torch.cat(patient_vecs + [graph_vec], dim=1)
        logits = self.fc(self.dropout(embed))

        results = {
            "logit": logits,
            "y_prob": self.label_processor.activation(logits),
        }
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            results["loss"] = self.label_processor.loss(logits, y_true)
            results["y_true"] = y_true
        if kwargs.get("embed", False):
            results["embed"] = embed
            results["node_weights"] = node_weights
        return results

    @staticmethod
    def _value(feature) -> torch.Tensor:
        """Returns the value tensor of a feature (processors may give tuples)."""
        if isinstance(feature, tuple):
            return feature[-1]
        return feature
