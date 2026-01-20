"""
SHy (Self-Explaining Hypergraph Neural Networks) model for PyHealth.

Name: Hyunsoo Lee
NetId: hyunsoo2
Paper Title: S. R. X et al., "Self-Explaining Hypergraph Neural Networks for
    Diagnosis Prediction", (SHy), 2024.
Paper Link: https://arxiv.org/html/2502.10689v2
Original implementation: https://github.com/ThunderbornSakana/SHy
Description: SHy for Next-Visit Diagnosis Prediction

This example demonstrates how to use the SHy model for next-visit diagnosis
prediction on the MIMIC dataset. SHy learns interpretable temporal phenotypes
(TPs) from patient visit sequences using hypergraph neural networks and structure
learning.

The example is implemented in examples/shy_mimic3_demo.py
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np

from pyhealth.datasets import BaseDataset, SampleDataset
from pyhealth.models import BaseModel

try:
    # SHy relies on scatter operations for hypergraph aggregation.
    from torch_scatter import scatter
except ImportError:
    scatter = None

# ---------------------------------------------------------
# 1. Graph Convolution (UniGINConv Only)
# ---------------------------------------------------------
class UniGINConv(nn.Module):
    """
    Unified Graph Isomorphism Network Convolution (UniGINConv).
    Adapted for Hypergraphs.
    """
    def __init__(self, in_channels, out_channels, heads, dropout=0.):
        super().__init__()
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, X, vertex, edges):
        """Perform a single UniGIN-style hypergraph convolution.

        Args:
            X: Node feature tensor of shape ``(num_nodes, feature_dim)``.
            vertex: 1D index tensor mapping connections to node indices.
            edges: 1D index tensor mapping connections to edge indices.

        Returns:
            Updated node features tensor of shape ``(num_nodes, heads*out_dim)``.
        """
        N = X.shape[0]
        # X: (Num_nodes, Features)

        # 1. Message from Nodes to Edges
        Xve = X[vertex]  # Gather node features for every connection
        Xe = scatter(Xve, edges, dim=0, reduce='mean')  # Aggregate at edges

        # 2. Message from Edges back to Nodes
        Xev = Xe[edges]  # Broadcast edge features back to connections
        Xv = scatter(Xev, vertex, dim=0, reduce='sum',
                     dim_size=N)  # Aggregate at nodes

        # 3. Update
        X = (1 + self.eps) * X + Xv
        X = self.W(X)

        # Guard against numerical instability
        X = torch.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
        return X

# ---------------------------------------------------------
# 2. Hierarchical Embedding
# ---------------------------------------------------------
class HierarchicalEmbedding(nn.Module):
    """Hierarchical code embedding layer.

    This module embeds medical codes (e.g., ICD-9/ICD-10) by combining
    embeddings from multiple hierarchy levels. The hierarchy for each code
    is given by `code_levels`, which indexes into separate embedding matrices
    for each level. The final code embedding is the concatenation of all
    level-specific embeddings.

    Args:
        code_levels: LongTensor of shape (num_codes, num_levels) where each
            entry is the index of the code at that hierarchy level.
        code_num_in_levels: List of ints; number of unique codes at each level.
        code_dims: List of ints; embedding dimension for each hierarchy level.

    Shape:
        - Output: (num_codes, sum(code_dims))
    """
    def __init__(
        self,
        code_levels: torch.Tensor,
        code_num_in_levels: List[int],
        code_dims: List[int],
    ):
        super().__init__()
        assert len(code_num_in_levels) == len(code_dims)
        self.code_levels = code_levels  # registered as buffer by SHy
        self.level_embeddings = nn.ModuleList(
            nn.Embedding(num_codes, dim)
            for num_codes, dim in zip(code_num_in_levels, code_dims)
        )

    def forward(self) -> torch.Tensor:
        """Returns the concatenated embedding for all codes.

        Returns:
            Tensor of shape (num_codes, sum(code_dims)).
        """
        level_embs = []
        for level_idx, emb in enumerate(self.level_embeddings):
            indices = self.code_levels[:, level_idx]
            level_embs.append(emb(indices))
        return torch.cat(level_embs, dim=-1)


# ---------------------------------------------------------
# 3. HGNN (Hypergraph Neural Network)
# ---------------------------------------------------------
class HGNN(nn.Module):
    """Hypergraph Neural Network built from UniGINConv layers.

    This module stacks multiple :class:`UniGINConv` layers followed by a
    final projection. It applies LeakyReLU and dropout between layers.
    """
    def __init__(self, nfeat, nhid, nclass, nlayer, nhead, dropout_p, device):
        super().__init__()
        self.nlayer = nlayer
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout_p)

        # Hardcoded to UniGINConv as requested
        self.convs = nn.ModuleList()

        # Input Layer
        self.convs.append(UniGINConv(
            nfeat, nhid, heads=nhead, dropout=dropout_p))

        # Hidden Layers
        # Note: UniGINConv with heads expands dimension by * heads.
        # We assume subsequent layers take (nhid * nhead) and project back to nhid.
        for _ in range(self.nlayer - 1):
            self.convs.append(UniGINConv(nhid * nhead, nhid,
                              heads=nhead, dropout=dropout_p))

        # Output Layer
        if self.nlayer > 0:
            self.conv_out = UniGINConv(
                nhid * nhead, nclass, heads=1, dropout=dropout_p)
        else:
            self.conv_out = UniGINConv(
                nfeat, nclass, heads=1, dropout=dropout_p)

    def forward(self, X, V, E, H=None):
        """Run the HGNN over a hypergraph.

        Args:
            X: Node features, shape ``(num_nodes, feature_dim)``.
            V: Node indices for non-zero incidences.
            E: Edge indices corresponding to `V`.
            H: Optional incidence matrix (unused by the convs but kept for
               API compatibility).

        Returns:
            Node features after the final projection.
        """
        for conv in self.convs:
            X = conv(X, V, E)
            X = self.act(X)
            X = self.dropout(X)
        X = self.conv_out(X, V, E)
        return F.leaky_relu(X)

# ---------------------------------------------------------
# 4. HSL Layers (Hypergraph Structure Learning)
# ---------------------------------------------------------
class HSL_Layer_Part1(nn.Module):
    """Calculates incident mask probabilities."""

    def __init__(self, emb_dim):
        super(HSL_Layer_Part1, self).__init__()
        self.sample_MLP_1 = nn.Linear(emb_dim * 2, 256)
        self.act = nn.ReLU()
        self.sample_MLP_2 = nn.Linear(256, 1)

    def forward(self, X, V, E):
        """Compute incident mask probabilities from node/edge features.

        Args:
            X: Node embeddings (num_nodes, emb_dim).
            V: Node indices for incidences.
            E: Edge indices for incidences.

        Returns:
            A 2D tensor of Bernoulli probabilities for each (node,edge)
            pair with shape ``(num_nodes, num_edges)``. Values are clamped
            to a stable range for sampling.
        """
        eX = scatter(X[V], E, dim=0, reduce='mean')
        eX = torch.nan_to_num(eX, nan=0.0)

        # Expand features to pair nodes with their edges
        X_expanded = X.unsqueeze(1).expand(
            X.shape[0], eX.shape[0], X.shape[-1])
        eX_expanded = eX.repeat(X.shape[0], 1, 1)
        cat_feat = torch.cat([X_expanded, eX_expanded], dim=-1)

        incident_mask_prob = self.act(self.sample_MLP_1(cat_feat))
        incident_mask_prob = torch.sigmoid(torch.squeeze(
            self.sample_MLP_2(incident_mask_prob), -1))

        # Clamp for numerical stability in Bernoulli sampling
        incident_mask_prob = torch.nan_to_num(incident_mask_prob, nan=0.5)
        incident_mask_prob = torch.clamp(
            incident_mask_prob, min=1e-6, max=1.0-1e-6)

        if incident_mask_prob.ndim == 1:
            incident_mask_prob = incident_mask_prob.unsqueeze(1)
        return incident_mask_prob


class HSL_Layer_Part2(nn.Module):
    """HSL Part 2: node sampling and false-negative edge addition.

    Given:
        - X: Code embeddings (num_codes, emb_dim)
        - H: Incidence matrix (num_codes, num_edges)
        - V, E: Node/edge indices for non-zero entries in H
        - incident_mask_prob: Bernoulli probabilities for each incidence

    This layer:
        1. Computes edge embeddings from incident codes.
        2. Proposes new node–edge incidences (false negatives) based on
           similarity in a learned space.
        3. Samples a mask to drop noisy incidences (false positives).

    Returns:
        Enriched incidence matrix of shape (num_codes, num_edges).
    """

    def __init__(self, n_c: int, emb_dim: int, add_ratio: float, temperature: float):
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
        # Guard: if no active incidences, return H unchanged
        if H.shape[1] == 0 or E.numel() == 0:
            return H

        # -- existing math, unchanged --
        # 1. Edge embeddings
        eX = scatter(X[V], E, dim=0, reduce="mean")

        node_fc = X.unsqueeze(1) * self.cos_weight
        all_node_m = F.normalize(node_fc, p=2, dim=-1, eps=1e-6).permute(1, 0, 2)
        edge_fc = eX.unsqueeze(1) * self.cos_weight
        all_edge_m = F.normalize(edge_fc, p=2, dim=-1, eps=1e-6).permute(1, 2, 0)

        S = torch.matmul(all_node_m, all_edge_m).mean(0)

        mask_val = -1e9
        S[V, E] = mask_val

        k = int(self.add_ratio * E.shape[0])
        if k > 0:
            v, i = torch.topk(S.flatten(), k)
            row = torch.div(i, S.shape[1], rounding_mode="floor")
            col = i % S.shape[1]
            delta_H = torch.zeros_like(H)
            delta_H[row, col] = 1.0
            enriched_H = H + delta_H
        else:
            enriched_H = H

        incident_mask = dist.RelaxedBernoulli(
            temperature=self.temperature,
            probs=incident_mask_prob,
        ).rsample()

        enriched_H = enriched_H * incident_mask
        return enriched_H

class HypergraphEmbeddingAggregator(nn.Module):
    """Aggregates hyperedges/visits over time via a GRU + attention.

    Given:
        - X: Code embeddings of shape (num_codes, emb_dim)
        - H: Incidence matrix (num_codes, num_visits)

    It:
        1. Computes visit embeddings as H^T * X.
        2. Feeds the sequence of visit embeddings into a GRU.
        3. Applies attention over the GRU hidden states to produce a single
           hypergraph embedding vector.

    Returns:
        Tensor of shape (hid_channel,) representing the patient-level embedding.
    """

    def __init__(self, in_channel: int, hid_channel: int):
        super().__init__()
        self.temporal_edge_aggregator = nn.GRU(in_channel, hid_channel, 1)
        self.attention_context = nn.Linear(hid_channel, 1, bias=False)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # H: (num_codes, num_visits)
        if H.dim() != 2 or H.shape[1] == 0:
            # No visits; return a zero embedding
            hid_size = self.temporal_edge_aggregator.hidden_size
            return X.new_zeros(hid_size)

        visit_emb = torch.matmul(H.T.to(torch.float32), X)  # (num_visits, emb_dim)
        if visit_emb.size(0) == 0:
            hid_size = self.temporal_edge_aggregator.hidden_size
            return X.new_zeros(hid_size)

        visit_emb = visit_emb.unsqueeze(1)  # (seq_len, batch=1, emb_dim)
        hidden_states, _ = self.temporal_edge_aggregator(visit_emb)

        attn_scores = self.attention_context(hidden_states)  # (seq_len, 1, 1)
        attn_scores = torch.nan_to_num(attn_scores, nan=-1e9)
        alpha = self.softmax(attn_scores)  # softmax over sequence length

        hg_emb = torch.sum(hidden_states * alpha, dim=0).squeeze(0)
        return hg_emb

class HSLEncoder(nn.Module):
    """Hypergraph + HSL encoder for temporal phenotypes.

    This module takes code embeddings and an incidence matrix H describing
    the patient-specific hypergraph (codes ↔ visits). It runs:

        1. HGNN over the hypergraph to get code embeddings.
        2. HSL (Hypergraph Structure Learner) to refine the hypergraph.
        3. Aggregation over hyperedges/visits to obtain a set of temporal
           phenotypes (TPs) for this patient.

    The output is a set of `num_TP` latent TP vectors per patient, which
    are then consumed by the classifier and decoder.

    Args:
        code_dims: List of embedding dims for each hierarchy level.
        HGNN_dim: Hidden dimension inside HGNN.
        after_HGNN_dim: Output dimension of HGNN before HSL.
        HGNN_layer_num: Number of HGNN layers.
        nhead: Number of attention heads inside HSL.
        num_TP: Number of temporal phenotypes to extract.
        temperature: Temperature for relaxed Bernoulli in HSL sampling.
        add_ratio: Ratio of edges to add in HSL Part 2.
        n_c: Number of clusters / components for HSL.
        hid_state_dim: Dimensionality of hidden state/latent TP.
        dropout: Dropout probability.
        device: Device for internal tensors.
    """
    def __init__(self, code_dims, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, num_TP, temperature, add_ratio, n_c, hid_state_dim, dropout, device):
        super(HSLEncoder, self).__init__()
        self.HGNN_layer_num = HGNN_layer_num
        input_dim = sum(code_dims)

        if HGNN_layer_num >= 0:
            self.firstHGNN = HGNN(
                input_dim, HGNN_dim, after_HGNN_dim, HGNN_layer_num, nhead, dropout, device)
        else:
            self.NoneHGNN = nn.Linear(input_dim, after_HGNN_dim)

        self.num_TP = num_TP
        self.HSL_P1_combo = nn.ModuleList(
            [HSL_Layer_Part1(after_HGNN_dim) for i in range(num_TP)])

        temps = temperature if isinstance(temperature, list) else [
            temperature]*num_TP
        ratios = add_ratio if isinstance(add_ratio, list) else [
            add_ratio]*num_TP

        self.HSL_P2_combo = nn.ModuleList([
            HSL_Layer_Part2(n_c, after_HGNN_dim, r, t)
            for t, r in zip(temps, ratios)
        ])
        self.hyperG_emb_aggregator = HypergraphEmbeddingAggregator(
            after_HGNN_dim, hid_state_dim)

    def forward(self, X, H):
        """Encode a patient's hypergraph and produce temporal phenotypes.

        Args:
            X: Global code embeddings, shape ``(vocab_size, emb_dim)``.
            H: Incidence matrix for this patient, shape
               ``(vocab_size, num_visits)``.

        Returns:
            A tuple ``(TPs, latent_TPs, incident_mask_probs)`` where:
            - ``TPs`` is either a list of per-TP incidence matrices (when
              ``num_TP > 1``) or a single incidence matrix (when ``num_TP==1``).
            - ``latent_TPs`` is a tensor of latent TP vectors
              (stacked when ``num_TP>1``).
            - ``incident_mask_probs`` contains the sampling probabilities
              produced by HSL Part 1.
        """

        # Convert Incidence Matrix H to Vertex (V) and Edge (E) indices
        V = torch.nonzero(H)[:, 0]
        E = torch.nonzero(H)[:, 1]

        if self.HGNN_layer_num >= 0:
            X_1 = self.firstHGNN(X, V, E, H)
        else:
            X_1 = F.leaky_relu(self.NoneHGNN(X))

        # Separate into Temporal Phenotypes (TPs)
        if self.num_TP > 1:
            incident_mask_probs = torch.stack(
                [l(X_1, V, E) for l in self.HSL_P1_combo])
            TPs_list = []
            latent_TPs_list = []
            for k in range(self.num_TP):
                tp_H = self.HSL_P2_combo[k](
                    X_1, H, V, E, incident_mask_probs[k])
                tp_latent = self.hyperG_emb_aggregator(X_1, tp_H)
                TPs_list.append(tp_H)
                latent_TPs_list.append(tp_latent)
            TPs = TPs_list
            latent_TPs = torch.stack(latent_TPs_list)
        else:
            incident_mask_probs = self.HSL_P1_combo[0](X_1, V, E)
            TPs = self.HSL_P2_combo[0](X_1, H, V, E, incident_mask_probs)
            latent_TPs = self.hyperG_emb_aggregator(X_1, TPs)

        return TPs, latent_TPs, incident_mask_probs

# ---------------------------------------------------------
# 5. Decoder & Classifier
# ---------------------------------------------------------
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden, X):
        """Run one step of the decoder RNN.

        Args:
            input: Current decoder input vector (e.g., previous target), shape
                ``(code_dim,)``.
            hidden: Previous GRU hidden state with shape ``(num_layers, batch, hidden_size)``.
            X: Projection matrix or embeddings used to compute the input-to-hidden mapping.

        Returns:
            Tuple ``(output, hidden)`` where ``output`` is the decoder output
            probabilities for the current step (shape ``(1, code_num)``) and
            ``hidden`` is the updated GRU hidden state.
        """
        output = F.relu(torch.matmul(input, X).view(1, -1))
        output = output.unsqueeze(1)
        output, hidden = self.gru(output, hidden)
        output = self.sigmoid(self.out(output[0]))
        return output, hidden


class HSL_Decoder(nn.Module):
    def __init__(self, latent_TP_dim, num_TP, proj_dim, code_num, device):
        super(HSL_Decoder, self).__init__()
        self.to_context = nn.Linear(latent_TP_dim * num_TP, proj_dim)
        self.reconstruct_net = DecoderRNN(proj_dim, code_num)
        self.device = device
        self.code_num = code_num

    def forward(self, latent_TP, visit_len, H, X):
        """Reconstruct visit-level incidence logits from latent TPs.

        Args:
            latent_TP: Latent TP representation for the patient (shape depends
                on encoder; flattened/concatenated here to form initial context).
            visit_len: Number of visits to reconstruct (sequence length).
            H: Original incidence matrix used as teacher-forcing targets.
            X: Global code embeddings used by the decoder projection.

        Returns:
            A tensor of reconstructed visit-level logits/probabilities with
            shape ``(visit_len, code_num)``. If ``visit_len==0`` an empty
            tensor with shape ``(0, code_num)`` is returned.
        """
        decoder_hidden = self.to_context(
            torch.reshape(latent_TP, (-1,))).view(1, -1)
        decoder_hidden = decoder_hidden.unsqueeze(0)

        target_tensor = H.T
        decoder_input = torch.zeros(self.code_num, device=self.device)

        outputs = []
        for di in range(visit_len):
            output, decoder_hidden = self.reconstruct_net(
                decoder_input, decoder_hidden, X)
            outputs.append(output)
            decoder_input = target_tensor[di]

        if len(outputs) == 0:
            return torch.zeros((0, self.code_num), device=self.device)

        return torch.cat(outputs, dim=0)


class FinalClassifier(nn.Module):
    """Final classifier over temporal phenotypes.

    For each patient we have `num_TP` temporal phenotype vectors. This module:
        1. Applies self-attention over TPs to obtain attention weights.
        2. Computes a weighted mixture of per-TP predictions (per-label logits).

    Args:
        in_channel: Dimensionality of each TP vector.
        code_num: Number of output labels/classes.
        key_dim: Dimensionality for attention keys/queries/values.
        SA_head: Number of attention heads.
        num_TP: Number of temporal phenotypes.
    """
    def __init__(
        self,
        in_channel: int,
        code_num: int,
        key_dim: int,
        SA_head: int,
        num_TP: int,
    ):
        super().__init__()
        self.num_TP = num_TP

        self.w_key = nn.Linear(in_channel, key_dim)
        self.w_query = nn.Linear(in_channel, key_dim)
        self.w_value = nn.Linear(in_channel, key_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=key_dim,
            num_heads=SA_head,
            batch_first=False,
        )

        self.tp_attention = nn.Linear(key_dim, 1, bias=False)
        self.classifier = nn.Linear(in_channel, code_num)
        # Attention should normalize over TP dimension
        self.softmax = nn.Softmax(dim=0)

    def forward(
        self, latent_tp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            latent_tp: Tensor of shape (batch, num_TP, in_channel).

        Returns:
            final_pred: Tensor of shape (batch, code_num), logits.
            alpha: Tensor of shape (batch, num_TP), attention weights.
        """
        latent_tp = torch.nan_to_num(latent_tp, nan=0.0)
        latent_tp_t = latent_tp.permute(1, 0, 2)  # (num_TP, batch, d)

        if self.num_TP > 1:
            keys = self.w_key(latent_tp_t)
            queries = self.w_query(latent_tp_t)
            values = self.w_value(latent_tp_t)

            sa_output, _ = self.multihead_attn(queries, keys, values, need_weights=False)
            sa_output = torch.nan_to_num(sa_output, nan=0.0)

            scores = self.tp_attention(sa_output).squeeze(-1)  # (num_TP, batch)
            alpha = self.softmax(scores)                       # softmax over TPs
            alpha = alpha.permute(1, 0)                        # (batch, num_TP)
        else:
            alpha = torch.ones(
                latent_tp.shape[0], self.num_TP, device=latent_tp.device
            )

        separate_pred = self.classifier(latent_tp)             # (batch, num_TP, code_num)
        final_pred = torch.sum(separate_pred * alpha.unsqueeze(-1), dim=1)
        return final_pred, alpha


# ---------------------------------------------------------
# 6. SHy Main Model (PyHealth)
# ---------------------------------------------------------


class SHy(BaseModel):
    """SHy: Self-Explaining Hypergraph Neural Network.

    This is a PyHealth wrapper around the SHy model, adapted from the
    original implementation for use with EHR tasks such as
    `DiagnosisPredictionMIMIC3`.

    The model expects:
        - A feature key corresponding to nested diagnosis sequences
          (e.g., "conditions"): shape [batch, num_visits, max_codes].
        - A multilabel output (e.g., "label") representing next-visit
          diagnosis codes.

    Internally, SHy:
        1. Embeds codes via `HierarchicalEmbedding`.
        2. Builds a patient-specific hypergraph (codes ↔ visits) from sequences.
        3. Runs HGNN + HSL to learn temporal phenotypes.
        4. Uses a decoder to reconstruct the hypergraph (fidelity loss).
        5. Uses `FinalClassifier` to predict next-visit diagnoses.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_keys: List[str],
        label_key: str,
        mode: str = "multilabel",
        embedding_dim: int = 32,
        hgnn_dim: int = 64,
        num_temporal_phenotypes: int = 3,
        dropout: float = 0.5,
        hgnn_layers: int = 2,
        nhead: int = 4,
        n_c: int = 10,
        key_dim: int = 64,
        sa_head: int = 4,
        loss_weights: List[float] = [1.0, 0.003, 0.00025, 0.04],
        **kwargs,
    ):
        if scatter is None:
            raise ImportError(
                "SHy model requires `torch-scatter`. "
                "Install it via: pip install torch-scatter "
                "and refer to https://github.com/rusty1s/pytorch_scatter "
                "for compatible wheels."
            )

        super().__init__(dataset=dataset)

        assert len(feature_keys) >= 1, "SHy expects at least one feature key."
        self.feature_keys = feature_keys
        self.feature_key = feature_keys[0]
        self.label_keys = [label_key]
        self.label_key = label_key
        self.mode = mode

        self.num_tp = num_temporal_phenotypes
        self.loss_weights = loss_weights

        # ------------------------------------------------------------------
        # 1) Vocabulary & label space
        # ------------------------------------------------------------------
        self.vocab = self._get_tokens(dataset, self.feature_key)
        self.vocab_size = self._calculate_vocab_size(self.vocab)

        # If still empty, fall back to 0..(num_labels-1) as code IDs.
        if self.vocab_size == 0:
            num_labels = self.get_output_size()
            self.vocab_size = num_labels
            self.vocab = list(range(self.vocab_size))

        # Label space size from dataset
        self.num_classes = self.get_output_size()

        # Build a simple hierarchy: single level for all codes by default.
        hierarchy = self._build_hierarchy_matrix(self.vocab, self.vocab_size)
        hierarchy_tensor = torch.tensor(hierarchy, dtype=torch.long)
        self.register_buffer("code_levels", hierarchy_tensor, persistent=False)

        num_levels = self.code_levels.shape[1]
        level_dims = [embedding_dim] * num_levels
        code_num_in_levels = [self.vocab_size] * num_levels

        # ------------------------------------------------------------------
        # 2) Hierarchical embedding
        # ------------------------------------------------------------------
        self.hier_embed = HierarchicalEmbedding(
            code_levels=self.code_levels,
            code_num_in_levels=code_num_in_levels,
            code_dims=level_dims,
        )
        input_dim = sum(level_dims)

        # ------------------------------------------------------------------
        # 3) Encoder (HGNN + HSL)
        # ------------------------------------------------------------------
        self.encoder = HSLEncoder(
            code_dims=level_dims,
            HGNN_dim=hgnn_dim,
            after_HGNN_dim=hgnn_dim,
            HGNN_layer_num=hgnn_layers,
            nhead=nhead,
            num_TP=self.num_tp,
            temperature=1.0,
            add_ratio=0.2,
            n_c=n_c,
            hid_state_dim=input_dim,
            dropout=dropout,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # 4) Decoder (hypergraph reconstruction)
        # ------------------------------------------------------------------
        self.decoder = HSL_Decoder(
            latent_TP_dim=input_dim,
            num_TP=self.num_tp,
            proj_dim=input_dim,
            code_num=self.vocab_size,
            device=self.device,
        )

        # ------------------------------------------------------------------
        # 5) Classifier (temporal phenotype mixture)
        # ------------------------------------------------------------------
        self.classifier = FinalClassifier(
            in_channel=input_dim,
            code_num=self.num_classes,
            key_dim=key_dim,
            SA_head=sa_head,
            num_TP=self.num_tp,
        )

    # ----------------------------------------------------------------------
    # Forward & loss
    # ----------------------------------------------------------------------
    def forward(self, **batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            **batch: PyHealth batch dict, should contain:
                - self.feature_key: diagnosis sequences, shape (B, T, L).
                - self.label_key: multilabel targets.

        Returns:
            Dict with keys:
                - "loss": scalar loss tensor.
                - "y_prob": predicted probabilities.
                - "y_true": target labels.
                - "logit": raw logits before activation.
        """
        conditions: torch.Tensor = batch[self.feature_key]
        labels: torch.Tensor = batch[self.label_key]

        # 1) Build incidence matrices per patient
        Hs, visit_lens = self._build_incidence_matrix(conditions)

        # 2) Global code embeddings
        X = self.hier_embed()  # (vocab_size, input_dim)

        # 3) Encode each patient's hypergraph
        latent_tp_list: List[torch.Tensor] = []
        recon_H_list: List[torch.Tensor] = []

        for i, H in enumerate(Hs):
            tp, latent_tp, _ = self.encoder(X, H)
            latent_tp_list.append(latent_tp)
            recon = self.decoder(latent_tp, visit_lens[i], H, X)
            recon_H_list.append(recon)

        batch_latent = torch.stack(latent_tp_list, dim=0)  # (B, num_TP, d)

        # 4) Classification over temporal phenotypes
        logits, alphas = self.classifier(batch_latent)

        # 5) Compute loss
        loss = self.compute_loss(
            logits=logits,
            targets=labels,
            Hs=Hs,
            recon_H_list=recon_H_list,
            alphas=alphas,
            visit_lens=visit_lens,
        )

        # 6) Probabilities (use BaseModel helper)
        y_prob = self.prepare_y_prob(logits)
        y_prob = torch.nan_to_num(y_prob, nan=0.0)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": labels,
            "logit": logits,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        Hs: List[torch.Tensor],
        recon_H_list: List[torch.Tensor],
        alphas: torch.Tensor,
        visit_lens: List[int],
    ) -> torch.Tensor:
        """Compute the SHy loss.

        The total loss is a weighted combination of:
            - prediction loss (multilabel BCE),
            - hypergraph reconstruction loss,
            - sparsity/regularization on attention weights.

        Args:
            logits: Prediction logits, shape (B, num_classes).
            targets: Targets, either indices or one-hot, shape (B, *).
            Hs: List of original incidence matrices, one per patient.
            recon_H_list: List of reconstructed incidence matrices.
            alphas: Attention weights over temporal phenotypes, shape (B, num_TP).
            visit_lens: Number of visits per patient.

        Returns:
            Scalar loss tensor.
        """
        # Prediction loss
        if self.mode == "multilabel":
            if targets.dim() == 2 and targets.shape[1] != logits.shape[1]:
                y_true = torch.zeros_like(logits)
                t_long = targets.long()
                mask = (t_long >= 0) & (t_long < logits.shape[1])
                safe_targets = t_long * mask.long()
                y_true.scatter_(1, safe_targets, 1)
                y_true[:, 0] = 0  # treat 0 as padding
                targets = y_true

            pred_loss = F.binary_cross_entropy_with_logits(
                logits, targets.float()
            )
        else:
            pred_loss = F.cross_entropy(logits, targets)

        # Hypergraph reconstruction loss (MSE on H)
        fid_loss = 0.0
        mse = nn.MSELoss()
        for k in range(len(recon_H_list)):
            v_len = visit_lens[k]
            if v_len > 0:
                target_h = Hs[k][:, :v_len].T        # (v_len, vocab_size)
                recon = recon_H_list[k][:v_len, :]   # (v_len, vocab_size)
                if target_h.numel() == 0 or recon.numel() == 0:
                    continue
                fid_loss += mse(recon, target_h)
        if len(recon_H_list) > 0:
            fid_loss = fid_loss / len(recon_H_list)

        # Attention regularization: encourage spread over TPs
        alpha_loss = torch.mean(torch.sum(alphas ** 2, dim=-1))

        loss = (
            self.loss_weights[0] * pred_loss
            + self.loss_weights[1] * fid_loss
            + self.loss_weights[3] * alpha_loss
        )
        return loss

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _get_tokens(self, dataset: SampleDataset, key: str) -> List[int]:
        """Collect all unique tokens for a given feature key.

        Handles nested sequences like:
            [[code1, code2], [code3, ...]]

        and converts scalar tensors to Python ints so they can be sorted.
        """
        tokens: Iterable[Any] = []
        if hasattr(dataset, "get_all_tokens"):
            tokens = dataset.get_all_tokens(key=key)  # may be empty

            norm_tokens: List[Any] = []
            for t in tokens:
                if torch.is_tensor(t):
                    if t.numel() == 1:
                        norm_tokens.append(int(t.item()))
                    else:
                        norm_tokens.extend(int(x.item()) for x in t.view(-1))
                else:
                    norm_tokens.append(t)
            if norm_tokens:
                return norm_tokens

        token_set: set = set()

        def add_val(val: Any) -> None:
            if torch.is_tensor(val):
                if val.numel() == 1:
                    token_set.add(int(val.item()))
                else:
                    for x in val.view(-1):
                        token_set.add(int(x.item()))
            else:
                token_set.add(val)

        for sample in dataset:
            data = sample[key]
            if isinstance(data, (list, tuple)):
                for sub in data:
                    if isinstance(sub, (list, tuple)):
                        for item in sub:
                            add_val(item)
                    else:
                        add_val(sub)
            else:
                add_val(data)

        return sorted(token_set)

    @staticmethod
    def _calculate_vocab_size(vocab: List[Any]) -> int:
        """Infer vocabulary size from a list of tokens.

        If tokens are integer-like, returns max(token) + 1.
        Otherwise, falls back to len(vocab).
        """
        if not vocab:
            return 0
        try:
            ints = [int(v) for v in vocab]
            return max(ints) + 1
        except Exception:
            return len(vocab)

    def _build_incidence_matrix(
        self, conditions: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Build incidence matrices H for each patient.

        Conditions are assumed to be:
            conditions[b, t, l] = code index or 0 (padding).

        For each patient b:
            - We consider each timestep t as a "visit".
            - We build H_b of shape (vocab_size, num_visits_b) where
              H_b[code, visit] = 1 if the code appears in that visit.

        Returns:
            Hs: List of incidence matrices, one per patient.
            visit_lens: List of num_visits_b for each patient.
        """
        batch_size = conditions.shape[0]
        Hs: List[torch.Tensor] = []
        visit_lens: List[int] = []

        for b in range(batch_size):
            sample = conditions[b]               # (T, L)
            valid_visits: List[torch.Tensor] = []

            for v_idx in range(sample.shape[0]):
                visit = sample[v_idx]
                visit_codes = visit[visit != 0]
                if visit_codes.numel() > 0:
                    valid_visits.append(visit_codes)

            num_visits = len(valid_visits)
            visit_lens.append(num_visits)

            if num_visits == 0:
                H = torch.zeros(
                    (self.vocab_size, 0), device=self.device
                )
            else:
                H = torch.zeros(
                    (self.vocab_size, num_visits), device=self.device
                )
                for j, codes in enumerate(valid_visits):
                    for code in codes:
                        idx = int(code.item())
                        if 0 <= idx < self.vocab_size:
                            H[idx, j] = 1.0

            Hs.append(H)

        return Hs, visit_lens

    @staticmethod
    def _build_hierarchy_matrix(vocab: List[Any], vocab_size: int) -> np.ndarray:
        """Build a simple hierarchy matrix for codes.

        Currently, this is a placeholder that assigns all codes to a single
        hierarchy level. It returns a matrix of ones with shape:

            (vocab_size, 1)

        so that each code has one "level index". This can be extended to
        multiple levels if a real hierarchy (e.g., ICD chapters) is desired.
        """
        return np.ones((vocab_size, 1), dtype=np.int64)