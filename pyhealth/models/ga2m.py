"""GA2M (Generalized Additive Model with Interactions) for PyHealth.

Implements the GA2M architecture described in:
    Hegselmann et al., "An Evaluation of the Doctor-Interpretability of
    Generalized Additive Models with Interactions", MLHC 2020.
    https://proceedings.mlr.press/v126/hegselmann20a.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleDataset


# Sentinel value for missing values (Following paper implementation)
UNKNOWN_SENTINEL = -1.0


class GA2M(BaseModel):
    """
        GA2M model with binned shape functions and pairwise interactions. 
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_bins: int = 256,                  # Sets model hyperparameters
        top_k_interactions: int = 34,       # Builds embeddings for main effects
        use_interactions: bool = True,      # Prepares interaction structures
    ) -> None:
        super().__init__(dataset)

        assert len(self.feature_keys) == 1, (
            "expects exactly one input feature key"
        )
        assert len(self.label_keys) == 1, (
            "expects exactly one binary label key"
        )

        # Store dataset keys for forward pass consistency
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.n_bins = n_bins
        self.top_k_interactions = top_k_interactions
        self.use_interactions = use_interactions

        # Infer input dimensionality from dataset
        sample = dataset[0]
        self.input_dim: int = sample[self.feature_key].shape[0]

        # Bin boundaries - not learned
        self.register_buffer(
            "bin_edges",
            torch.zeros(self.input_dim, n_bins - 1),
        )
        self._bins_fitted: bool = False

        # Bias term - global intercept
        self.bias = nn.Parameter(torch.zeros(1))

        # Main effect shape functions - each feature has (n_bins + 1) embeddings (1 for unknown values)
        self.main_effects = nn.ModuleList([
            nn.Embedding(n_bins + 1, 1)
            for _ in range(self.input_dim)
        ])

        # Initialise all risk scores to zero
        for emb in self.main_effects:
            nn.init.zeros_(emb.weight)

        # Interaction componenets for Stage 2
        self.interaction_pairs: List[Tuple[int, int]] = []
        self.interactions = nn.ModuleDict()

        # Track current stage for error messaging
        self._interactions_selected: bool = False

    # --- Bin step - preprocessing stage before training ---

    def fit_bins(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
            Computes quantile-based discretization bins
            Includes proper preprocessing pipeline step 
            Ensures interpretability which is necessary from the paper
        """
        all_features: List[torch.Tensor] = []

        # Collet dataset for global quantile computation
        for batch in data_loader:
            x = batch[self.feature_key]  # (B, D)
            all_features.append(x.cpu())
        X = torch.cat(all_features, dim=0)  # (N, D)

        edges = torch.zeros(self.input_dim, self.n_bins - 1)
        for d in range(self.input_dim):
            # Extract feature column
            col = x[:, d].contiguous()

            # Remove missing values 
            valid = col[col != UNKNOWN_SENTINEL]
            if valid.numel() == 0:
                continue
            quantiles = torch.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            edges[d] = torch.quantile(valid.float(), quantiles)

        self.bin_edges.copy_(edges)
        self._bins_fitted = True

    # --- bin assignment ---

    def _assign_bins(self, x: torch.Tensor) -> torch.LongTensor:
        """
            Map continuous feature values to bin indices
        """
        batch_size = x.size(0)
        bin_idx = torch.zeros(
            batch_size, self.input_dim,
            dtype=torch.long, device=x.device,
        )

        for d in range(self.input_dim):
            col = x[:, d].contiguous()
            unknown_mask = (col == UNKNOWN_SENTINEL)

            edges = self.bin_edges[d].contiguous()
            idx = torch.bucketize(col, edges)  # (B,) in [0, n_bins-1]

            # Assign unkown values to special bins
            idx[unknown_mask] = self.n_bins
            bin_idx[:, d] = idx

        return bin_idx

    # --- Stage 1 - main effect training --- 

    def fit_main_effects(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 1e-2,
    ) -> None:
        """
            Stage 1: train only main effect shape functions
            Implementing training loop 
        """
        if not self._bins_fitted:
            raise RuntimeError("need to call fit_bins(train_loader) before fit_main_effects()")

        params = list(self.main_effects.parameters()) + [self.bias]
        optimiser = torch.optim.Adam(params, lr=lr)

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in data_loader:
                optimiser.zero_grad()
                out = self._forward_main_effects_only(batch)
                out["loss"].backward()
                optimiser.step()
                total_loss += out["loss"].item()
            print(
                f"[GA2M Stage 1] Epoch {epoch + 1}/{epochs}  "
                f"loss={total_loss / len(data_loader):.4f}"
            )

    # Forward pass used in Stage 1 only 
    def _forward_main_effects_only(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
            Forward pass using only main effects
            logit = bias + sum(main_effects(feature bins))
        """
        x = batch[self.feature_key].to(self.device)
        y_true = batch[self.label_key].float().to(self.device)

        bin_idx = self._assign_bins(x)  # (B, D)
        batch_size = x.size(0)

        logits = self.bias.expand(batch_size, 1).clone()

        # Add per feature contributions
        for d in range(self.input_dim):
            logits = logits + self.main_effects[d](bin_idx[:, d]).squeeze(-1).unsqueeze(-1)

        loss = F.binary_cross_entropy_with_logits(logits, y_true)
        return {"loss": loss, "logits": logits}

    # --- Stage 1 -> Stage 2 - interaction selection --- 

    def select_top_interactions(self) -> List[Tuple[int, int]]:
        """
            Select the top-K feature interactions
        """
        # per-feature variance of learned risk scores
        variances = []
        for d in range(self.input_dim):
            weight = self.main_effects[d].weight.data  # (n_bins+1, 1)
            variances.append(weight.var().item())

        # Rank all pairs by score
        pair_scores: List[Tuple[float, int, int]] = []
        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                score = variances[i] * variances[j]
                pair_scores.append((score, i, j))

        pair_scores.sort(key=lambda t: t[0], reverse=True)
        top_pairs = [
            (i, j)
            for _, i, j in pair_scores[: self.top_k_interactions]
        ]

        self.interaction_pairs = top_pairs

        grid_size = (self.n_bins + 1) ** 2  # flattened 2D bin grid
        self.interactions = nn.ModuleDict({
            f"{i}_{j}": nn.Embedding(grid_size, 1)
            for i, j in top_pairs
        })
        for emb in self.interactions.values():
            nn.init.zeros_(emb.weight)

        self._interactions_selected = True
        print(
            f"[GA2M] Selected {len(top_pairs)} interaction pairs "
            f"(top_k={self.top_k_interactions})."
        )
        return top_pairs

    # --- full model forward pass ---

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
            Compute using main effects and interactions 
            logit = bias + sum_d  f_d(bin(x_d)) + sum_{i,j} f_{ij}(bin(x_i), bin(x_j))
        """
        if not self._bins_fitted:
            raise RuntimeError("run fit_bins(train_loader) before running forward()")
        if self.use_interactions and not self._interactions_selected:
            raise RuntimeError("Call select_top_interactions() after Stage 1 training before running the full model forward pass")

        x = kwargs[self.feature_key]
        y_true = kwargs[self.label_key].float()

        bin_idx = self._assign_bins(x)  # (B, D)
        batch_size = x.size(0)

        # main effects
        logits = self.bias.expand(batch_size, 1).clone()
        for d in range(self.input_dim):
            logits = logits + self.main_effects[d](bin_idx[:, d]).squeeze(-1).unsqueeze(-1)

        # interaction effects 
        if self.use_interactions and self._interactions_selected:
            n_bins_plus1 = self.n_bins + 1
            for i, j in self.interaction_pairs:
                key = f"{i}_{j}"
                flat_idx = (
                    bin_idx[:, i] * n_bins_plus1 + bin_idx[:, j]
                )  # (B,)
                logits = logits + self.interactions[key](flat_idx).squeeze(-1).unsqueeze(-1)

        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        return {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logits": logits,
        }

    # --- helpers for interpretability --- 

    def get_shape_function(self, feature_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
            Extract a single feature from teh learned 1D risk shape function
        """
        edges = self.bin_edges[feature_idx].detach().cpu().numpy()
        lower = np.concatenate([[-np.inf], edges])
        upper = np.concatenate([edges, [np.inf]])
        # find midpoints
        midpoints = np.where(
            np.isfinite(lower) & np.isfinite(upper),
            (lower + upper) / 2,
            np.where(np.isfinite(lower), lower + 1.0, upper - 1.0),
        )
        risk_scores = (
            self.main_effects[feature_idx]
            .weight.data[: self.n_bins]  # exclude unknown bin
            .squeeze(1)
            .detach()
            .cpu()
            .numpy()
        )
        return midpoints, risk_scores

    def get_interaction_shape(
        self, feature_i: int, feature_j: int
    ) -> Optional[np.ndarray]:
        """
            Extract feature pair from the learned 2D interaction risk grid
        """
        key = f"{feature_i}_{feature_j}"
        if key not in self.interactions:
            return None
        grid = (
            self.interactions[key]
            .weight.data.reshape(self.n_bins + 1, self.n_bins + 1)
            .detach()
            .cpu()
            .numpy()
        )
        return grid[: self.n_bins, : self.n_bins]  # exclude unknown row/col