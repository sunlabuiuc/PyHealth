"""
SHy: Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction.

Paper: Leisheng Yu, Yanxiao Cai, Minxing Zhang, and Xia Hu.
    Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction.
    Proceedings of Machine Learning Research (CHIL), 2025.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


def _scatter_mean(src: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    """Average vectors by index."""
    out = torch.zeros(num, src.shape[1], device=src.device)
    count = torch.zeros(num, 1, device=src.device)
    idx = index.unsqueeze(1).expand_as(src)
    out.scatter_add_(0, idx, src)
    count.scatter_add_(
        0, index.unsqueeze(1), torch.ones(index.shape[0], 1, device=src.device)
    )
    return out / count.clamp(min=1)


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, num: int) -> torch.Tensor:
    """Sum vectors by index."""
    out = torch.zeros(num, src.shape[1], device=src.device)
    out.scatter_add_(0, index.unsqueeze(1).expand_as(src), src)
    return out


class UniGINConv(nn.Module):
    """
    One layer of hypergraph message passing (UniGIN).

    Two-stage aggregation:
        1. Node -> Hyperedge: mean of node features
        2. Hyperedge -> Node: sum of hyperedge features
    Then a GIN-style learnable self-loop.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.eps = nn.Parameter(torch.tensor(0.1))

    def forward(
        self, X: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            X: Node features (num_nodes, in_dim).
            V: Node indices (COO format).
            E: Hyperedge indices in COO format.
        """
        num_nodes = X.shape[0]
        if E.numel() == 0:
            return self.W((1 + self.eps) * X)
        num_edges = E.max().item() + 1

        # nodes -> hyperedges (mean)
        edge_emb = _scatter_mean(X[V], E, num_edges)

        # hyperedges -> nodes (sum)
        node_msg = _scatter_sum(edge_emb[E], V, num_nodes)

        return self.W((1 + self.eps) * X + node_msg)


class PhenotypeExtractor(nn.Module):
    """
    Extracts the temporal phenotype from a patient.

    Three steps:
        1. Score each (code, visit) pair for inclusion probability
        2. Add false negatives
        3. Gumbel-Softmax

    Args:
        emb_dim: Code embedding dimension.
        temperature: Gumbel-Softmax temperature.
        add_ratio: Fraction of false-negative connections.
    """

    def __init__(self, emb_dim: int, temperature: float = 1.0, add_ratio: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.add_ratio = add_ratio

        # Score how likely a code belongs to a visit for this phenotype
        self.scorer = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _add_false_negatives(
        self, X: torch.Tensor, H: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """
        Find codes that are probably missing from visits and add them,
        by using cosine similarity.

        Args:
            X: Node features (num_nodes, in_dim).
            H: Incidence matrix (num_nodes, num_visits)
            V: Node indices (COO format).
            E: Hyperedge indices in COO format.
        """
        num_edges = H.shape[1]
        # Mean code embeddings in a visit
        visit_emb = _scatter_mean(X[V], E, num_edges)
        # Cosine similarity between every (code, visit) pairs
        X_norm = F.normalize(X, dim=-1)
        visit_norm = F.normalize(visit_emb, dim=-1)
        sim = X_norm @ visit_norm.T
        sim[H > 0] = -1e16

        # Add the top-k most similar missing (code, visit) pairs
        num_to_add = max(1, int(self.add_ratio * V.shape[0]))
        num_to_add = min(num_to_add, sim.numel())
        _, flat_idx = torch.topk(sim.flatten(), num_to_add)
        rows = flat_idx // sim.shape[1]
        cols = flat_idx % sim.shape[1]

        enriched = H.clone()
        enriched[rows, cols] = 1.0
        return enriched

    def _score_pairs(
        self, X: torch.Tensor, V: torch.Tensor, E: torch.Tensor, num_edges: int
    ) -> torch.Tensor:
        """Score every (code, visit) pair for phenotype inclusion."""
        visit_emb = _scatter_mean(X[V], E, num_edges)

        code_rep = X.unsqueeze(1).expand(-1, num_edges, -1)
        visit_rep = visit_emb.unsqueeze(0).expand(X.shape[0], -1, -1)
        pair_feat = torch.cat([code_rep, visit_rep], dim=-1)

        return torch.sigmoid(self.scorer(pair_feat).squeeze(-1))

    def _gumbel_sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Gumbel-Softmax."""
        # Gumbel noise
        u = torch.rand_like(probs).clamp(1e-16, 1 - 1e-16)
        gumbel = torch.log(u) - torch.log(1 - u)
        logit = torch.log(probs.clamp(1e-16) / (1 - probs).clamp(1e-16))
        soft = torch.sigmoid((logit + gumbel) / self.temperature)
        hard = (soft > 0.5).float()
        return hard - soft.detach() + soft

    def forward(
        self, X: torch.Tensor, H: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract one phenotype sub-hypergraph.

        Args:
            X: Node features (num_nodes, in_dim).
            H: Incidence matrix (num_nodes, num_visits)
            V: Node indices (COO format).
            E: Hyperedge indices in COO format.

        Returns:
            Phenotype incidence matrix (num_codes, num_visits).
        """
        # Step 1: Add potentially missing connections
        enriched_H = self._add_false_negatives(X, H, V, E)

        # Step 2: Score each (code, visit) for this phenotype
        probs = self._score_pairs(X, V, E, H.shape[1])

        # Step 3: Sample binary mask
        mask = self._gumbel_sample(probs)

        return enriched_H * mask


class PhenotypeAggregator(nn.Module):
    """
    Put a phenotype sub-hypergraph into a vector.

    Args:
        emb_dim: Input code embedding dimension.
        hidden_dim: GRU hidden state dimension.
    """

    def __init__(self, emb_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim, batch_first=False)
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Args:
            X: Code embeddings (num_codes, emb_dim).
            H: Phenotype incidence matrix (num_codes, num_visits).

        Returns:
            Phenotype embedding vector (hidden_dim,).
        """
        # Weighted sum of code embeddings per visit
        visit_emb = H.T.float() @ X  # (num_visits, emb_dim)

        # GRU captures time patterns across visits
        states, _ = self.gru(visit_emb)  # (num_visits, hidden_dim)

        # Attention pooling: weight each visit's importance
        weights = F.softmax(self.attn(states).squeeze(-1), dim=0)
        return (weights.unsqueeze(-1) * states).sum(dim=0)  # (hidden_dim,)


class Decoder(nn.Module):
    """
    Reconstructs the original incidence matrix from phenotype embeddings.

    Args:
        hidden_dim: Phenotype embedding dimension.
        num_tp: Number of temporal phenotypes.
        emb_dim: Code embedding dimension.
        num_codes: Vocabulary size.
    """

    def __init__(self, hidden_dim: int, num_tp: int, emb_dim: int, num_codes: int):
        super().__init__()
        self.W_context = nn.Linear(hidden_dim * num_tp, hidden_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim)
        self.W_output = nn.Linear(hidden_dim, num_codes)
        self.num_codes = num_codes

    def forward(
        self,
        phenotype_embs: torch.Tensor,
        num_visits: int,
        H: torch.Tensor,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            phenotype_embs: Concatenated phenotype vectors (num_tp, hidden_dim).
            num_visits: Number of visits to reconstruct.
            H: Original incidence matrix (num_codes, num_visits).
            X: Code embedding table (num_codes, emb_dim).

        Returns:
            Reconstructed incidence matrix (num_codes, num_visits).
        """
        hidden = self.W_context(phenotype_embs.reshape(-1))
        hidden = hidden.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)

        result = []
        prev_codes = torch.zeros(self.num_codes, device=X.device)

        for t in range(num_visits):
            embed_input = F.relu(prev_codes @ X).unsqueeze(0).unsqueeze(0)
            out, hidden = self.gru(embed_input, hidden)
            pred = torch.sigmoid(self.W_output(out.squeeze(0).squeeze(0)))
            result.append(pred)
            # Teacher forcing during training; use own predictions at eval
            if self.training:
                prev_codes = H.T[t]
            else:
                prev_codes = pred.detach()

        return torch.stack(result, dim=1)  # (num_codes, num_visits)


class Classifier(nn.Module):
    """
    Predicts diagnoses from K phenotype embeddings.

    Args:
        hidden_dim: Phenotype embedding dimension.
        num_codes: Number of diagnosis codes to predict.
        num_tp: Number of temporal phenotypes.
        num_heads: Self-attention heads.
    """

    def __init__(
        self, hidden_dim: int, num_codes: int, num_tp: int, num_heads: int = 4
    ):
        super().__init__()
        self.num_tp = num_tp
        if num_tp > 1:
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads)
            self.W_importance = nn.Linear(hidden_dim, 1, bias=False)
        self.predict = nn.Linear(hidden_dim, num_codes)

    def forward(
        self, phenotype_embs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args:
            phenotype_embs: (batch, num_tp, hidden_dim) or (batch, hidden_dim).

        Returns:
            pred: Predicted probabilities (batch, num_codes).
            logit: Pre-sigmoid logits (batch, num_codes).
            alpha: Phenotype importance weights (batch, num_tp).
        """
        if self.num_tp > 1:
            # nn.MultiheadAttention expects (seq_len, batch, dim)
            x = phenotype_embs.transpose(0, 1)  # (num_tp, batch, dim)
            attended, _ = self.self_attn(x, x, x)
            attended = attended.transpose(0, 1)  # (batch, num_tp, dim)

            # Each phenotype's importance weight
            alpha = F.softmax(self.W_importance(attended).squeeze(-1), dim=-1)

            # Weighted combination of attended phenotype embeddings
            combined = (attended * alpha.unsqueeze(-1)).sum(dim=-2)  # (batch, dim)
            logit = self.predict(combined)
            pred = torch.sigmoid(logit)
            return pred, logit, alpha
        else:
            logit = self.predict(phenotype_embs.squeeze(1))
            pred = torch.sigmoid(logit)
            alpha = torch.ones(phenotype_embs.shape[0], 1, device=pred.device)
            return pred, logit, alpha


class SHy(BaseModel):
    """
    SHy: Self-Explaining Hypergraph Neural Network.

    Pipeline:
        1. Embed diagnosis codes
        2. HGNN message passing personalizes embeddings per patient
        3. Extract K phenotype sub-hypergraphs
        4. GRU + attention put phenotype to a vector
        5. Decoder reconstructs original hypergraph
        6. Classifier predicts next-visit diagnoses

    Note:
        This implementation processes samples sequentially in forward.
        For large batches or datasets, consider batching via
        block-diagonal hypergraph construction.

    Args:
        dataset: PyHealth SampleDataset.
        embedding_dim: Code embedding dimension. Default 32.
        hgnn_dim: HGNN output dimension. Default 64.
        hgnn_layers: Number of HGNN layers. Default 2.
        num_tp: Number of temporal phenotypes K. Default 5.
        hidden_dim: GRU/aggregator hidden dimension. Default 64.
        temperature: Gumbel-Softmax temperature. Default 1.0.
        add_ratio: False-negative addition ratio. Default 0.1.
        num_heads: Self-attention heads in classifier. Default 4.
        dropout: Dropout probability. Default 0.1.
        fidelity_weight: Weight for reconstruction loss. Default 0.1.
        distinct_weight: Weight for phenotype overlap penalty. Default 0.01.
        alpha_weight: Weight for attention diversity. Default 0.01.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "diagnoses_hist": [["d1", "d2"], ["d3", "d6"]],
        ...         "diagnoses": ["d1", "d2", "d3"],
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "diagnoses_hist": [["d1", "d3"], ["d5"]],
        ...         "diagnoses": ["d2", "d3"],
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"diagnoses_hist": "nested_sequence"},
        ...     output_schema={"diagnoses": "multilabel"},
        ...     dataset_name="test",
        ... )
        >>> model = SHy(dataset=dataset)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> out = model(**batch)
        >>> out["loss"].shape
        torch.Size([])
        >>> out["y_prob"].shape[1] == model.output_size
        True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 32,
        hgnn_dim: int = 64,
        hgnn_layers: int = 2,
        num_tp: int = 5,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        add_ratio: float = 0.1,
        num_heads: int = 4,
        dropout: float = 0.1,
        fidelity_weight: float = 0.1,
        distinct_weight: float = 0.01,
        alpha_weight: float = 0.01,
    ):
        super(SHy, self).__init__(dataset=dataset)

        if len(self.label_keys) != 1:
            raise ValueError("SHy supports exactly one label key (multilabel)")
        if len(self.feature_keys) != 1:
            raise ValueError("SHy expects exactly one feature key (nested_sequence)")

        self.label_key = self.label_keys[0]
        self.feature_key = self.feature_keys[0]

        self.num_tp = num_tp
        self.hidden_dim = hidden_dim
        self.fidelity_weight = fidelity_weight
        self.distinct_weight = distinct_weight
        self.alpha_weight = alpha_weight

        processor = dataset.input_processors[self.feature_key]
        self.vocab_size = processor.vocab_size()
        self.output_size = self.get_output_size()

        # 1. Code embedding
        self.code_embedding = nn.Embedding(self.vocab_size, embedding_dim)

        # 2. Message Passing
        self.hgnn_layers_n = hgnn_layers
        if hgnn_layers > 0:
            layers = []
            dims = [embedding_dim] + [hgnn_dim] * hgnn_layers
            for i in range(hgnn_layers):
                layers.append(UniGINConv(dims[i], dims[i + 1]))
            self.hgnn_convs = nn.ModuleList(layers)
            self.hgnn_out = nn.Linear(hgnn_dim, hgnn_dim)
        else:
            self.hgnn_out = nn.Linear(embedding_dim, hgnn_dim)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

        # 3. K phenotype extractors
        self.extractors = nn.ModuleList(
            [
                PhenotypeExtractor(hgnn_dim, temperature, add_ratio)
                for _ in range(num_tp)
            ]
        )

        # 4. Aggregator: sub-hypergraph -> vector
        self.aggregator = PhenotypeAggregator(hgnn_dim, hidden_dim)

        # 5. Decoder: reconstruction
        self.decoder = Decoder(hidden_dim, num_tp, embedding_dim, self.vocab_size)

        # 6. Classifier: phenotype embeddings -> diagnosis prediction
        self.classifier = Classifier(hidden_dim, self.output_size, num_tp, num_heads)

    def _build_incidence_matrix(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert padded code tensor to binary incidence matrix.

        Args:
            codes: (num_visits, max_codes_per_visit), 0 = padding.

        Returns:
            H: (vocab_size, num_visits), binary.
        """
        num_visits = codes.shape[0]
        H = torch.zeros(self.vocab_size, num_visits, device=codes.device)
        visit_idx = torch.arange(num_visits, device=codes.device)
        visit_idx = visit_idx.unsqueeze(1).expand_as(codes)
        mask = codes > 0
        rows = codes[mask].clamp(max=self.vocab_size - 1).long()
        cols = visit_idx[mask]
        H[rows, cols] = 1.0
        return H

    def _run_hgnn(
        self, X: torch.Tensor, V: torch.Tensor, E: torch.Tensor
    ) -> torch.Tensor:
        """Run HGNN message passing to personalize code embeddings."""
        if self.hgnn_layers_n > 0:
            for conv in self.hgnn_convs:
                X = self.dropout(self.act(conv(X, V, E)))
        return self.act(self.hgnn_out(X))

    def _encode_patient(self, X: torch.Tensor, H: torch.Tensor):
        """
        Encode one patient: HGNN -> extract K phenotypes.

        Returns:
            phenotype_matrices: list of K incidence matrices (C, V).
            phenotype_embs: (K, hidden_dim) or (hidden_dim,) if K=1.
        """
        # COO indices from incidence matrix
        nz = torch.nonzero(H)
        V, E = nz[:, 0], nz[:, 1]

        # Personalize embeddings via HGNN
        X_personal = self._run_hgnn(X, V, E)

        # Extract K phenotypes
        tp_matrices = [ext(X_personal, H, V, E) for ext in self.extractors]
        tp_embs = [self.aggregator(X_personal, tp) for tp in tp_matrices]

        if self.num_tp > 1:
            return tp_matrices, torch.stack(tp_embs)
        else:
            return tp_matrices, tp_embs[0]

    def _compute_loss(self, pred, y_true, tp_list, recon_list, H_list, alphas):
        """
        SHy loss.
        L = L_pred + eps*L_fidelity + eta*L_distinct - omega*L_alpha
        """
        # 1. Prediction loss: weighted BCE (upweight rare positives)
        num_pos = y_true.sum(dim=1, keepdim=True).clamp(min=1)
        num_neg = (y_true.shape[1] - num_pos).clamp(min=1)
        pos_weight = (num_neg / num_pos).expand_as(y_true)
        weight = torch.where(y_true > 0.5, pos_weight, torch.ones_like(y_true))
        loss = F.binary_cross_entropy(
            pred.clamp(1e-9, 1 - 1e-9), y_true.float(), weight=weight
        )

        # 2. Fidelity loss: reconstruction (reweighted like prediction loss)
        if recon_list:
            fid_losses = []
            for r, h in zip(recon_list, H_list):
                h_f = h.float()
                n_pos = h_f.sum().clamp(min=1.0)
                n_neg = (h_f.numel() - n_pos).clamp(min=1.0)
                pos_w = n_neg / n_pos
                w = torch.where(h_f > 0.5, pos_w, torch.ones_like(h_f))
                fid_losses.append(
                    F.binary_cross_entropy(
                        r.clamp(1e-9, 1 - 1e-9), h_f, weight=w
                    )
                )
            fidelity = sum(fid_losses) / len(fid_losses)
            loss = loss + self.fidelity_weight * fidelity

        # 3. Distinctness loss
        if self.num_tp > 1 and tp_list:
            eye = torch.eye(self.num_tp, device=pred.device)
            distinct = torch.tensor(0.0, device=pred.device)
            for tps in tp_list:
                if tps[0].dim() >= 2:
                    stacked = torch.stack(tps, dim=-1)  # (C, V, K)
                    for v in range(stacked.shape[1]):
                        col = stacked[:, v, :]  # (C, K)
                        distinct = distinct + torch.norm(eye - col.T @ col)
                    distinct = distinct / stacked.shape[1]
            loss = loss + self.distinct_weight * distinct / max(len(tp_list), 1)

            # 4. Alpha diversity: phenotype balance
            alpha_div = torch.mean(torch.sqrt(torch.var(alphas, dim=1).clamp(min=1e-9)))
            loss = loss - self.alpha_weight * alpha_div

        return loss

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with keys: loss, y_prob, y_true, logit.
        """
        feature_data = kwargs[self.feature_key]
        codes_batch = (
            feature_data[0] if isinstance(feature_data, tuple) else feature_data
        )
        batch_size = codes_batch.shape[0]

        X = self.code_embedding(torch.arange(self.vocab_size, device=self.device))

        tp_list, recon_list, H_list, latent_list = [], [], [], []
        valid_mask = []

        for i in range(batch_size):
            H = self._build_incidence_matrix(codes_batch[i]).to(self.device)

            if H.sum() == 0:
                zero = (
                    torch.zeros(self.num_tp, self.hidden_dim, device=self.device)
                    if self.num_tp > 1
                    else torch.zeros(self.hidden_dim, device=self.device)
                )
                latent_list.append(zero)
                valid_mask.append(False)
                continue

            tp_mats, tp_embs = self._encode_patient(X, H)
            tp_list.append(tp_mats)
            latent_list.append(tp_embs)
            H_list.append(H)
            recon_list.append(self.decoder(tp_embs, H.shape[1], H, X))
            valid_mask.append(True)

        # Classify: phenotype embeddings -> diagnosis prediction
        stacked = torch.stack(latent_list)  # (batch, K, hidden) / (batch, hidden)
        if self.num_tp > 1 and stacked.dim() == 2:
            stacked = stacked.unsqueeze(1)
        pred, logit, alphas = self.classifier(stacked)

        # Labels
        y_true = kwargs[self.label_key].to(self.device).float()

        # Exclude empty-history samples from loss computation
        valid = torch.tensor(valid_mask, device=self.device)
        if valid.any():
            loss = self._compute_loss(
                pred[valid], y_true[valid],
                tp_list, recon_list, H_list, alphas[valid],
            )
        else:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return {"loss": loss, "y_prob": pred, "y_true": y_true, "logit": logit}
