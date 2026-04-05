"""SHy: Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction.

Paper: Ruijie Yu et al. Self-Explaining Hypergraph Neural Networks for
Diagnosis Prediction. CHIL 2025.
Paper link: https://proceedings.mlr.press/v287/yu25a.html

This module implements SHy as a PyHealth BaseModel. SHy builds a per-patient
hypergraph (diseases as nodes, visits as hyperedges), applies UniGIN message
passing to produce personalized disease embeddings, extracts temporal phenotypes
via Gumbel-Softmax sampling, and predicts future diagnoses through a weighted
combination of per-phenotype predictions.

Key components:
    - HierarchicalEmbedding: Embeds ICD codes using multi-level hierarchy.
    - UniGINConv: Hypergraph message passing layer.
    - UniGATConv: Attention-based hypergraph message passing layer.
    - HGNN: Stacked hypergraph neural network.
    - HSLEncoder: Hypergraph structure learning with false-negative
      augmentation and phenotype extraction.
    - HypergraphEmbeddingAggregator: GRU-based aggregation of phenotype
      embeddings.
    - FinalClassifier: Self-attention weighted combination of per-phenotype
      predictions.
    - SHyLayer: Standalone layer combining all components.
    - SHy: Full PyHealth model inheriting from BaseModel.

TODO: Review and polish all docstrings below to ensure they follow
Google style consistently. Add Examples sections to inner classes
where missing. Verify all Args and Returns are documented.
See rubric Section 1 "Documentation" (5 pts).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


def _glorot(tensor: torch.Tensor) -> None:
    """Xavier uniform initialization."""
    if tensor is not None:
        stdv = (6.0 / (tensor.size(-2) + tensor.size(-1))) ** 0.5
        tensor.data.uniform_(-stdv, stdv)


class HierarchicalEmbedding(nn.Module):
    """Hierarchical embedding for ICD-9 codes.

    Embeds each level of the ICD-9 hierarchy separately and concatenates
    the results. This allows related diseases to share sub-embeddings at
    coarser hierarchy levels.

    Args:
        code_levels: Tensor of shape (num_codes, num_levels) mapping each
            code to its ancestor at each hierarchy level.
        code_num_in_levels: List of vocabulary sizes at each level.
        code_dims: List of embedding dimensions at each level.
    """

    def __init__(
        self,
        code_levels: torch.Tensor,
        code_num_in_levels: List[int],
        code_dims: List[int],
    ):
        super().__init__()
        self.level_num = len(code_num_in_levels)
        self.register_buffer("code_levels", code_levels)
        self.level_embeddings = nn.ModuleList(
            [
                nn.Embedding(code_num, code_dim)
                for code_num, code_dim in zip(code_num_in_levels, code_dims)
            ]
        )

    def forward(self) -> torch.Tensor:
        """Returns: Tensor of shape (num_codes, total_embed_dim)."""
        embeddings = [
            self.level_embeddings[level](self.code_levels[:, level] - 1)
            for level in range(self.level_num)
        ]
        return torch.cat(embeddings, dim=1)


class UniGINConv(nn.Module):
    """UniGIN convolution layer for hypergraph message passing.

    Implements the UniGIN aggregation scheme: messages are passed from nodes
    to hyperedges (mean aggregation) and back to nodes (sum aggregation),
    with a learnable epsilon for self-loop weighting.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        heads: Number of attention heads (multiplies out_channels).
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.eps = nn.Parameter(torch.tensor([0.1]))

    def forward(
        self, X: torch.Tensor, vertex: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            X: Node features, shape (num_nodes, in_channels).
            vertex: Vertex indices of incident pairs.
            edges: Hyperedge indices of incident pairs.

        Returns:
            Updated node features, shape (num_nodes, heads * out_channels).
        """
        N = X.shape[0]
        Xve = X[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce="mean")
        Xev = Xe[edges]
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        X = (1 + self.eps) * X + Xv
        return self.W(X)


class UniGATConv(nn.Module):
    """UniGAT convolution layer for hypergraph message passing.

    Uses attention-based aggregation instead of GIN-style aggregation.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension per head.
        heads: Number of attention heads.
    """

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.heads = heads
        self.out_channels = out_channels
        self.attn_drop = nn.Dropout(0.0)
        self.leaky_relu = nn.LeakyReLU()
        _glorot(self.att_e)

    def forward(
        self, X: torch.Tensor, vertex: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        H, C, N = self.heads, self.out_channels, X.shape[0]
        X0 = self.W(X)
        X_view = X0.view(N, H, C)
        Xve = X_view[vertex]
        Xe = scatter(Xve, edges, dim=0, reduce="mean")
        alpha_e = (Xe * self.att_e).sum(-1)
        a_ev = alpha_e[edges]
        alpha = self.leaky_relu(a_ev)
        # softmax over edges per vertex
        alpha_max = scatter(alpha, vertex, dim=0, reduce="max", dim_size=N)
        alpha = torch.exp(alpha - alpha_max[vertex])
        alpha_sum = scatter(alpha, vertex, dim=0, reduce="sum", dim_size=N)
        alpha = alpha / (alpha_sum[vertex] + 1e-8)
        alpha = self.attn_drop(alpha)
        Xev = Xe[edges] * alpha.unsqueeze(-1)
        Xv = scatter(Xev, vertex, dim=0, reduce="sum", dim_size=N)
        return (Xv.view(N, H * C) + X0)


class HGNN(nn.Module):
    """Stacked hypergraph neural network.

    Applies multiple layers of hypergraph convolutions (UniGIN or UniGAT)
    with LeakyReLU activations and dropout.

    Args:
        nfeat: Input feature dimension.
        nhid: Hidden dimension.
        nclass: Output dimension.
        nlayer: Number of intermediate layers (0 = single output layer).
        nhead: Number of attention heads per layer.
        dropout_p: Dropout probability.
        conv_type: Type of convolution ("UniGINConv" or "UniGATConv").
    """

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        nlayer: int,
        nhead: int,
        dropout_p: float,
        conv_type: str = "UniGINConv",
    ):
        super().__init__()
        self.nlayer = nlayer
        conv_cls = UniGINConv if conv_type == "UniGINConv" else UniGATConv

        self.convs = nn.ModuleList(
            [conv_cls(nfeat, nhid, heads=nhead)]
            + [
                conv_cls(nhid * nhead, nhid, heads=nhead)
                for _ in range(max(0, nlayer - 1))
            ]
        )
        in_dim = nhid * nhead if nlayer > 0 else nfeat
        self.conv_out = conv_cls(in_dim, nclass, heads=1)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self, X: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        if self.nlayer > 0:
            for conv in self.convs:
                X = self.dropout(self.act(conv(X, V, E)))
        X = self.conv_out(X, V, E)
        return F.leaky_relu(X)


class HSLPart1(nn.Module):
    """Computes incident mask probabilities for phenotype extraction.

    For each (node, hyperedge) pair, outputs a probability indicating
    whether the node belongs to this phenotype's sub-hypergraph.

    Args:
        emb_dim: Embedding dimension after HGNN.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp1 = nn.Linear(emb_dim * 2, 256)
        self.act = nn.ReLU()
        self.mlp2 = nn.Linear(256, 1)

    def forward(
        self, X: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        eX = scatter(X[V], E, dim=0, reduce="mean")
        combined = torch.cat(
            [
                X.unsqueeze(1).expand(X.shape[0], eX.shape[0], X.shape[-1]),
                eX.repeat(X.shape[0], 1, 1),
            ],
            dim=-1,
        )
        prob = torch.sigmoid(self.mlp2(self.act(self.mlp1(combined)))).squeeze(-1)
        if prob.dim() == 1:
            prob = prob.unsqueeze(1)
        return prob


class HSLPart2(nn.Module):
    """False-negative augmentation and Gumbel-Softmax node sampling.

    Identifies likely missing disease-visit pairs using cosine similarity,
    adds them to the hypergraph, then samples nodes via relaxed Bernoulli.

    Args:
        n_c: Number of cosine weight vectors.
        emb_dim: Embedding dimension.
        add_ratio: Ratio of false-negative pairs to add.
        temperature: Temperature for Gumbel-Softmax sampling.
    """

    def __init__(
        self, n_c: int, emb_dim: int, add_ratio: float, temperature: float
    ):
        super().__init__()
        self.cos_weight = nn.Parameter(torch.randn(n_c, emb_dim))
        self.add_ratio = add_ratio
        self.temperature = temperature

    def forward(
        self,
        X: torch.Tensor,
        H: torch.Tensor,
        V: torch.Tensor,
        E: torch.Tensor,
        incident_mask_prob: torch.Tensor,
    ) -> torch.Tensor:
        # False-negative augmentation via cosine similarity
        eX = scatter(X[V], E, dim=0, reduce="mean")
        node_fc = F.normalize(
            (X.unsqueeze(1) * self.cos_weight), p=2, dim=-1
        ).permute(1, 0, 2)
        edge_fc = F.normalize(
            (eX.unsqueeze(1) * self.cos_weight), p=2, dim=-1
        ).permute(1, 2, 0)
        S = torch.matmul(node_fc, edge_fc).mean(0)
        S[V, E] = -1e30
        num_add = max(1, int(self.add_ratio * E.shape[0]))
        _, indices = torch.topk(S.flatten(), num_add)
        row = torch.div(indices, S.shape[1], rounding_mode="floor")
        col = indices % S.shape[1]
        delta_H = torch.zeros_like(H)
        delta_H[row, col] = 1.0
        enriched_H = H + delta_H

        # Gumbel-Softmax node sampling
        eps = torch.empty_like(incident_mask_prob).uniform_(1e-6, 1 - 1e-6)
        logit = torch.log(eps) - torch.log(1 - eps)
        logit = (torch.log(incident_mask_prob + 1e-8)
                 - torch.log(1 - incident_mask_prob + 1e-8)
                 + logit)
        soft = torch.sigmoid(logit / self.temperature)
        hard = (soft > 0.5).float()
        incident_mask = hard - soft.detach() + soft  # straight-through

        enriched_H = enriched_H * incident_mask
        return enriched_H


class HypergraphEmbeddingAggregator(nn.Module):
    """Aggregates node embeddings into a single hypergraph embedding.

    Uses a GRU over visit-level embeddings (computed by multiplying the
    incidence matrix transpose with node embeddings), followed by
    attention-weighted summation.

    Args:
        in_channel: Input dimension (node embedding size).
        hid_channel: GRU hidden state dimension.
    """

    def __init__(self, in_channel: int, hid_channel: int):
        super().__init__()
        self.gru = nn.GRU(in_channel, hid_channel, batch_first=False)
        self.attention = nn.Linear(hid_channel, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        visit_emb = torch.matmul(H.T.float(), X)
        hidden_states, _ = self.gru(visit_emb)
        alpha = self.softmax(self.attention(hidden_states).squeeze(-1))
        hg_emb = torch.sum(
            torch.matmul(torch.diag(alpha), hidden_states), dim=0
        )
        return hg_emb


class HSLEncoder(nn.Module):
    """Hypergraph Structure Learning encoder.

    Combines HGNN message passing with multi-channel phenotype extraction.
    Each channel extracts one temporal phenotype via HSL (false-negative
    augmentation + Gumbel-Softmax sampling).

    Args:
        total_emb_dim: Total embedding dimension from HierarchicalEmbedding.
        hgnn_dim: Hidden dimension for HGNN layers.
        after_hgnn_dim: Output dimension after HGNN.
        hgnn_layer_num: Number of HGNN layers.
        nhead: Number of attention heads in HGNN.
        num_tp: Number of temporal phenotypes to extract.
        temperatures: List of Gumbel-Softmax temperatures per phenotype.
        add_ratios: List of false-negative add ratios per phenotype.
        n_c: Number of cosine weight vectors.
        hid_state_dim: GRU hidden state dimension for aggregation.
        dropout: Dropout probability.
        conv_type: HGNN convolution type.
    """

    def __init__(
        self,
        total_emb_dim: int,
        hgnn_dim: int,
        after_hgnn_dim: int,
        hgnn_layer_num: int,
        nhead: int,
        num_tp: int,
        temperatures: List[float],
        add_ratios: List[float],
        n_c: int,
        hid_state_dim: int,
        dropout: float,
        conv_type: str = "UniGINConv",
    ):
        super().__init__()
        self.hgnn_layer_num = hgnn_layer_num
        self.num_tp = num_tp

        if hgnn_layer_num >= 0:
            self.hgnn = HGNN(
                total_emb_dim,
                hgnn_dim,
                after_hgnn_dim,
                hgnn_layer_num,
                nhead,
                dropout,
                conv_type,
            )
        else:
            self.linear_fallback = nn.Linear(total_emb_dim, after_hgnn_dim)

        self.hsl_p1 = nn.ModuleList(
            [HSLPart1(after_hgnn_dim) for _ in range(num_tp)]
        )
        self.hsl_p2 = nn.ModuleList(
            [
                HSLPart2(n_c, after_hgnn_dim, ar, temp)
                for temp, ar in zip(temperatures, add_ratios)
            ]
        )
        self.aggregator = HypergraphEmbeddingAggregator(
            after_hgnn_dim, hid_state_dim
        )

    def forward(
        self, X: torch.Tensor, H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]

        if self.hgnn_layer_num >= 0:
            X_1 = self.hgnn(X, V, E)
        else:
            X_1 = F.leaky_relu(self.linear_fallback(X))

        if self.num_tp > 1:
            probs = torch.stack(
                [p1(X_1, V, E) for p1 in self.hsl_p1]
            )
            tps = torch.stack(
                [
                    self.hsl_p2[k](X_1, H, V, E, probs[k])
                    for k in range(self.num_tp)
                ]
            )
            latent_tps = torch.stack(
                [self.aggregator(X_1, tps[j]) for j in range(self.num_tp)]
            )
        else:
            probs = self.hsl_p1[0](X_1, V, E)
            tps = self.hsl_p2[0](X_1, H, V, E, probs)
            latent_tps = self.aggregator(X_1, tps)

        return tps, latent_tps


class DecoderRNN(nn.Module):
    """GRU-based decoder for hypergraph reconstruction (fidelity loss).

    Args:
        hidden_size: Hidden state dimension.
        output_size: Number of disease codes.
    """

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        X: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = F.relu(torch.matmul(input, X).view(1, -1))
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output[0]))
        return output, hidden


class HSLDecoder(nn.Module):
    """Decoder that reconstructs the original hypergraph from phenotype
    embeddings, used for computing the fidelity loss.

    Args:
        latent_tp_dim: Dimension of each phenotype embedding.
        num_tp: Number of temporal phenotypes.
        proj_dim: Projection dimension for context vector.
        code_num: Total number of disease codes.
    """

    def __init__(
        self, latent_tp_dim: int, num_tp: int, proj_dim: int, code_num: int
    ):
        super().__init__()
        self.to_context = nn.Linear(latent_tp_dim * num_tp, proj_dim)
        self.reconstruct_net = DecoderRNN(proj_dim, code_num)
        self.code_num = code_num

    def forward(
        self,
        latent_tp: torch.Tensor,
        visit_len: int,
        H: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        decoder_hidden = self.to_context(
            torch.reshape(latent_tp, (-1,))
        ).view(1, -1)
        reconstructed_H = torch.zeros(
            visit_len, self.code_num, device=latent_tp.device
        )
        target_tensor = H.T
        decoder_input = torch.zeros(self.code_num, device=latent_tp.device)
        for di in range(visit_len):
            output, decoder_hidden = self.reconstruct_net(
                decoder_input, decoder_hidden, X
            )
            reconstructed_H[di] = output[0]
            decoder_input = target_tensor[di]  # teacher forcing
        return reconstructed_H.T


class FinalClassifier(nn.Module):
    """Prediction head combining per-phenotype predictions.

    Uses self-attention to compute importance weights for each phenotype's
    prediction, then combines them as a weighted sum.

    Args:
        in_channel: Phenotype embedding dimension.
        code_num: Number of output disease codes.
        key_dim: Key/query/value dimension for self-attention.
        sa_head: Number of self-attention heads.
        num_tp: Number of temporal phenotypes.
    """

    def __init__(
        self,
        in_channel: int,
        code_num: int,
        key_dim: int,
        sa_head: int,
        num_tp: int,
    ):
        super().__init__()
        self.num_tp = num_tp
        if num_tp > 1:
            self.w_key = nn.Linear(in_channel, key_dim)
            self.w_query = nn.Linear(in_channel, key_dim)
            self.w_value = nn.Linear(in_channel, key_dim)
            self.multihead_attn = nn.MultiheadAttention(key_dim, sa_head)
            self.tp_attention = nn.Linear(key_dim, 1, bias=False)
        self.classifier = nn.Linear(in_channel, code_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self, latent_tp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.num_tp > 1:
            keys = self.w_key(latent_tp)
            querys = self.w_query(latent_tp)
            values = self.w_value(latent_tp)
            sa_output, _ = self.multihead_attn(
                querys, keys, values, need_weights=False
            )
            alpha = self.softmax(self.tp_attention(sa_output).squeeze(-1))
            separate_pred = self.softmax(self.classifier(latent_tp))
            final_pred = torch.sum(
                separate_pred
                * alpha.unsqueeze(-1).expand_as(separate_pred),
                dim=-2,
            )
            return final_pred, alpha
        else:
            final_pred = self.softmax(self.classifier(latent_tp))
            alpha = torch.ones(1, device=latent_tp.device)
            return final_pred, alpha


class SHyLayer(nn.Module):
    """SHy layer combining all components.

    This layer can be used standalone outside of the PyHealth framework.

    Args:
        code_levels: Numpy array of shape (num_codes, num_levels).
        single_dim: Embedding dimension per ICD-9 hierarchy level.
        hgnn_dim: HGNN hidden dimension.
        after_hgnn_dim: HGNN output dimension.
        hgnn_layer_num: Number of HGNN layers (0-indexed from reference).
        nhead: Number of HGNN attention heads.
        num_tp: Number of temporal phenotypes.
        temperatures: Gumbel-Softmax temperatures per phenotype.
        add_ratios: False-negative add ratios per phenotype.
        n_c: Number of cosine weight vectors.
        hid_state_dim: GRU hidden state dimension.
        dropout: Dropout probability.
        key_dim: Self-attention key dimension.
        sa_head: Number of self-attention heads.
        conv_type: HGNN convolution type.

    Examples:
        >>> import numpy as np
        >>> code_levels = np.array([[1, 1, 1], [1, 1, 2], [1, 2, 3]])
        >>> layer = SHyLayer(
        ...     code_levels=code_levels, single_dim=32,
        ...     hgnn_dim=64, after_hgnn_dim=64, hgnn_layer_num=1,
        ...     nhead=2, num_tp=3, temperatures=[0.5, 0.5, 0.5],
        ...     add_ratios=[0.1, 0.1, 0.1], n_c=5, hid_state_dim=64,
        ...     dropout=0.1, key_dim=64, sa_head=2, conv_type="UniGINConv",
        ... )
    """

    def __init__(
        self,
        code_levels: np.ndarray,
        single_dim: int = 32,
        hgnn_dim: int = 256,
        after_hgnn_dim: int = 128,
        hgnn_layer_num: int = 2,
        nhead: int = 4,
        num_tp: int = 5,
        temperatures: Optional[List[float]] = None,
        add_ratios: Optional[List[float]] = None,
        n_c: int = 10,
        hid_state_dim: int = 128,
        dropout: float = 0.001,
        key_dim: int = 256,
        sa_head: int = 8,
        conv_type: str = "UniGINConv",
        output_size: Optional[int] = None,
    ):
        super().__init__()

        if temperatures is None:
            temperatures = [0.5] * num_tp
        if add_ratios is None:
            add_ratios = [0.1] * num_tp

        code_num_in_levels = np.max(code_levels, axis=0).tolist()
        code_levels_t = torch.from_numpy(code_levels).long()
        code_dims = [single_dim] * code_levels.shape[1]
        total_emb_dim = sum(code_dims)
        num_codes = code_levels.shape[0]
        # Output size for classifier: use label dimension if provided,
        # otherwise fall back to number of codes in hierarchy.
        classifier_out = output_size if output_size is not None else num_codes

        self.hier_embed = HierarchicalEmbedding(
            code_levels_t, code_num_in_levels, code_dims
        )
        self.encoder = HSLEncoder(
            total_emb_dim,
            hgnn_dim,
            after_hgnn_dim,
            hgnn_layer_num,
            nhead,
            num_tp,
            temperatures,
            add_ratios,
            n_c,
            hid_state_dim,
            dropout,
            conv_type,
        )
        self.decoder = HSLDecoder(
            hid_state_dim, num_tp, total_emb_dim, num_codes
        )
        self.classifier = FinalClassifier(
            hid_state_dim, classifier_out, key_dim, sa_head, num_tp
        )
        self.num_tp = num_tp

    def forward(
        self,
        Hs: List[torch.Tensor],
        visit_lens: List[int],
    ) -> Tuple[torch.Tensor, list, list, torch.Tensor]:
        """Forward pass.

        Args:
            Hs: List of incidence matrices, each shape (num_codes, max_visits).
            visit_lens: List of actual visit counts per patient.

        Returns:
            pred: Predicted probabilities, shape (batch, num_codes).
            tp_list: List of temporal phenotype incidence matrices.
            recon_H_list: List of reconstructed incidence matrices.
            alphas: Phenotype importance weights, shape (batch, num_tp).
        """
        X = self.hier_embed()
        tp_list = []
        latent_tp_list = []
        recon_H_list = []

        for i in range(len(Hs)):
            H_i = Hs[i][:, : int(visit_lens[i])]
            tp, latent_tp = self.encoder(X, H_i)
            tp_list.append(tp)
            latent_tp_list.append(latent_tp)
            recon_H_list.append(
                self.decoder(latent_tp, int(visit_lens[i]), H_i, X)
            )

        pred, alphas = self.classifier(torch.stack(latent_tp_list))
        return pred, tp_list, recon_H_list, alphas


def shy_loss(
    pred: torch.Tensor,
    label: torch.Tensor,
    original_h: torch.Tensor,
    reconstruction: list,
    tps: list,
    alphas: torch.Tensor,
    visit_lens: List[int],
    obj_r: List[float],
    device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[str]]:
    """Compute the combined SHy loss.

    Args:
        pred: Predicted probabilities.
        label: Ground truth multi-hot labels.
        original_h: Original incidence matrices.
        reconstruction: Reconstructed incidence matrices.
        tps: Temporal phenotype sub-hypergraphs.
        alphas: Phenotype importance weights.
        visit_lens: Visit lengths per patient.
        obj_r: Loss weights [pred, fidelity, distinct, alpha].
        device: Torch device.

    Returns:
        Total loss, list of individual losses, list of loss names.
    """
    criterion = nn.BCELoss()
    # Loss 1: prediction
    loss_pred = criterion(pred, label)
    # Loss 2: fidelity (reconstruction)
    loss_fidelity = torch.tensor(0.0, device=device)
    for k in range(len(tps)):
        recon = reconstruction[k].flatten()
        target = original_h[k][:, : int(visit_lens[k])].flatten()
        loss_fidelity = loss_fidelity + criterion(recon, target)
    loss_fidelity = loss_fidelity / len(tps)

    if len(tps[0].shape) > 2:
        # Loss 3: distinctness
        loss_distinct = torch.tensor(0.0, device=device)
        Q = torch.eye(tps[0].shape[0], device=device)
        for j in range(len(tps)):
            swap_tp = torch.swapaxes(tps[j], 0, -1)
            loss_temp = torch.tensor(0.0, device=device)
            for jj in range(len(swap_tp)):
                one_visit = swap_tp[jj]
                loss_temp = loss_temp + torch.norm(
                    Q - torch.matmul(one_visit.t(), one_visit), p=2
                )
            loss_distinct = loss_distinct + loss_temp / len(swap_tp)
        loss_distinct = loss_distinct / len(tps)
        # Loss 4: alpha regularization
        loss_alpha = torch.mean(
            torch.sqrt(torch.var(alphas, dim=1))
            - torch.norm(alphas, p=2, dim=1)
        )
        total = (
            obj_r[0] * loss_pred
            + obj_r[1] * loss_fidelity
            + obj_r[2] * loss_distinct
            - obj_r[3] * loss_alpha
        )
        return (
            total,
            [loss_pred, loss_fidelity, loss_distinct, loss_alpha],
            ["Prediction", "Fidelity", "Distinctness", "Alpha"],
        )
    else:
        total = obj_r[0] * loss_pred + obj_r[1] * loss_fidelity
        return (
            total,
            [loss_pred, loss_fidelity],
            ["Prediction", "Fidelity"],
        )


class SHy(BaseModel):
    """SHy model for diagnosis prediction in PyHealth.

    Paper: Ruijie Yu et al. Self-Explaining Hypergraph Neural Networks for
    Diagnosis Prediction. CHIL 2025.

    SHy builds a per-patient hypergraph where diseases are nodes and visits
    are hyperedges. It uses UniGIN message passing to produce personalized
    disease embeddings, extracts temporal phenotypes via Gumbel-Softmax,
    and predicts future diagnoses as a weighted combination of per-phenotype
    predictions.

    Note:
        SHy requires ``torch_scatter`` to be installed. Install via:
        ``pip install torch_scatter``.

        This model expects input data as multi-hot coded visit sequences.
        The ``conditions`` feature should be a nested sequence of ICD codes
        representing visits. A ``code_levels`` numpy array mapping each code
        to its ICD-9 hierarchy must be provided.

    Args:
        dataset: The PyHealth SampleDataset.
        code_levels: Numpy array of shape (num_codes, num_hierarchy_levels),
            mapping each disease code to its ancestor at each level of the
            ICD-9 tree.
        single_dim: Embedding dimension per hierarchy level. Default 32.
        hgnn_dim: HGNN hidden dimension. Default 256.
        after_hgnn_dim: HGNN output dimension. Default 128.
        hgnn_layer_num: Number of HGNN layers. Default 2.
        nhead: Number of HGNN attention heads. Default 4.
        num_tp: Number of temporal phenotypes. Default 5.
        temperatures: Gumbel-Softmax temperatures. Default [0.5]*num_tp.
        add_ratios: False-negative add ratios. Default [0.1]*num_tp.
        n_c: Number of cosine weight vectors. Default 10.
        hid_state_dim: GRU hidden state dimension. Default 128.
        dropout: Dropout probability. Default 0.001.
        key_dim: Self-attention key dimension. Default 256.
        sa_head: Number of self-attention heads. Default 8.
        conv_type: HGNN convolution type ("UniGINConv" or "UniGATConv").
            Default "UniGINConv".
        loss_weights: Weights for [pred, fidelity, distinct, alpha] losses.
            Default [1.0, 0.1, 0.01, 0.01].

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> import numpy as np
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": [["001", "401"], ["250"]],
        ...         "label": [1, 0, 1],
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "conditions": [["401", "250"], ["001"]],
        ...         "label": [0, 1, 0],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "nested_sequence"},
        ...     output_schema={"label": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>> code_levels = np.array([[1,1,1],[1,1,2],[1,2,3]])
        >>> model = SHy(
        ...     dataset=dataset,
        ...     code_levels=code_levels,
        ...     single_dim=16, hgnn_dim=32, after_hgnn_dim=32,
        ...     hgnn_layer_num=1, nhead=2, num_tp=2,
        ...     temperatures=[0.5, 0.5], add_ratios=[0.1, 0.1],
        ...     n_c=5, hid_state_dim=32, key_dim=32, sa_head=2,
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        code_levels: np.ndarray,
        single_dim: int = 32,
        hgnn_dim: int = 256,
        after_hgnn_dim: int = 128,
        hgnn_layer_num: int = 2,
        nhead: int = 4,
        num_tp: int = 5,
        temperatures: Optional[List[float]] = None,
        add_ratios: Optional[List[float]] = None,
        n_c: int = 10,
        hid_state_dim: int = 128,
        dropout: float = 0.001,
        key_dim: int = 256,
        sa_head: int = 8,
        conv_type: str = "UniGINConv",
        loss_weights: Optional[List[float]] = None,
    ):
        super(SHy, self).__init__(dataset=dataset)

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        if temperatures is None:
            temperatures = [0.5] * num_tp
        if add_ratios is None:
            add_ratios = [0.1] * num_tp
        if loss_weights is None:
            loss_weights = [1.0, 0.1, 0.01, 0.01]

        self.loss_weights = loss_weights
        self.num_tp = num_tp

        # Get vocabulary size from the processor for the first feature key
        # to determine how many token indices the model must handle.
        # The processor adds <pad> (0) and <unk> (1) before real codes,
        # so we pad code_levels with dummy rows to match vocab size.
        feature_key = self.feature_keys[0]
        processor = dataset.input_processors[feature_key]
        vocab_size = processor.vocab_size()

        # Expand code_levels: prepend dummy rows for special tokens
        # (pad, unk) so that processor token index == code_levels row.
        n_special = vocab_size - code_levels.shape[0]
        if n_special > 0:
            dummy = np.ones(
                (n_special, code_levels.shape[1]), dtype=code_levels.dtype
            )
            code_levels = np.concatenate([dummy, code_levels], axis=0)
        self._vocab_size = vocab_size
        # Output size from label processor (number of label classes)
        self._output_size = self.get_output_size()

        self.shy_layer = SHyLayer(
            code_levels=code_levels,
            single_dim=single_dim,
            hgnn_dim=hgnn_dim,
            after_hgnn_dim=after_hgnn_dim,
            hgnn_layer_num=hgnn_layer_num,
            nhead=nhead,
            num_tp=num_tp,
            temperatures=temperatures,
            add_ratios=add_ratios,
            n_c=n_c,
            hid_state_dim=hid_state_dim,
            dropout=dropout,
            key_dim=key_dim,
            sa_head=sa_head,
            conv_type=conv_type,
            output_size=self._output_size,
        )

    def _build_hypergraphs(
        self,
        conditions: torch.Tensor,
        num_codes: int,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Convert batched condition sequences into per-patient hypergraphs.

        Each patient's visit history is converted into an incidence matrix
        H of shape (num_codes, num_visits) where H[i, j] = 1 means
        disease i was present in visit j.

        Args:
            conditions: Batched condition tensor from the dataloader.
                Shape: (batch, max_visits, max_codes_per_visit) for
                nested sequences from NestedSequenceProcessor.
            num_codes: Total number of codes (including pad/unk from
                processor vocabulary).

        Returns:
            Hs: List of incidence matrices, each (num_codes, num_visits).
            visit_lens: List of actual visit counts per patient.
        """
        Hs = []
        visit_lens = []
        batch_size = conditions.shape[0]

        for i in range(batch_size):
            patient = conditions[i]

            if patient.dim() == 1:
                H = torch.zeros(num_codes, 1, device=conditions.device)
                for code_idx in patient:
                    idx = int(code_idx.item())
                    if 0 < idx < num_codes:
                        H[idx, 0] = 1.0
                # Ensure at least one nonzero entry
                if H.sum() == 0:
                    H[1, 0] = 1.0
                Hs.append(H)
                visit_lens.append(1)

            elif patient.dim() == 2:
                visit_mask = patient.abs().sum(dim=-1) > 0
                n_visits = max(1, int(visit_mask.sum().item()))
                H = torch.zeros(
                    num_codes, n_visits, device=conditions.device
                )
                for v in range(n_visits):
                    has_code = False
                    for code_idx in patient[v]:
                        idx = int(code_idx.item())
                        if 0 < idx < num_codes:
                            H[idx, v] = 1.0
                            has_code = True
                    # Ensure every visit has at least one code
                    if not has_code:
                        H[1, v] = 1.0
                Hs.append(H)
                visit_lens.append(n_visits)

            else:
                raise ValueError(
                    f"Unexpected condition tensor shape: {patient.shape}"
                )

        return Hs, visit_lens

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        The model expects a nested sequence of diagnosis codes as input.
        Each patient's visit history is converted into a hypergraph
        incidence matrix, processed through the SHy pipeline, and
        produces multi-label diagnosis predictions.

        Args:
            **kwargs: Keyword arguments containing feature keys and
                optionally the label key. Features are unpacked from
                processor tuples when necessary.

        Returns:
            A dictionary with keys:
                - logit: raw prediction scores, shape (batch, num_codes).
                - y_prob: predicted probabilities, shape (batch, num_codes).
                - loss: scalar loss (only if label key is present).
                - y_true: true labels (only if label key is present).
                - embed: patient embeddings (only if embed=True).
        """
        feature_key = self.feature_keys[0]
        feature = kwargs[feature_key]

        # Unpack from processor tuple if needed
        if isinstance(feature, (tuple, list)):
            conditions = feature[0]
        else:
            conditions = feature

        num_codes = self._vocab_size
        Hs, visit_lens = self._build_hypergraphs(conditions, num_codes)

        pred, tp_list, recon_H_list, alphas = self.shy_layer(
            Hs, visit_lens
        )

        # pred is already softmax'd by FinalClassifier
        results = {
            "logit": pred,
            "y_prob": pred,
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            # Build per-patient incidence matrices for fidelity loss
            max_vlen = max(visit_lens)
            padded_Hs = torch.stack(
                [
                    F.pad(h, (0, max_vlen - h.shape[1]))
                    for h in Hs
                ]
            )
            loss, _, _ = shy_loss(
                pred,
                y_true,
                padded_Hs,
                recon_H_list,
                tp_list,
                alphas,
                visit_lens,
                self.loss_weights,
                self.device,
            )
            results["loss"] = loss
            results["y_true"] = y_true

        if kwargs.get("embed", False):
            results["embed"] = pred
        return results
