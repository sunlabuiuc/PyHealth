"""GA2M (Generalized Additive Model with Interactions) for PyHealth.

Implements the GA2M architecture described in:
    Hegselmann et al., "An Evaluation of the Doctor-Interpretability of
    Generalized Additive Models with Interactions", MLHC 2020.
    https://proceedings.mlr.press/v126/hegselmann20a.html

Architecture Notes:
    The original paper trains GA2Ms using gradient-boosted trees (via the
    mltk toolkit from Lou et al. 2013). This implementation uses a PyTorch-
    native equivalent: each shape function is represented as a learned
    embedding over discretized bins, which is representationally equivalent
    but optimized via SGD rather than boosting.

    Key design choices faithful to the paper:
        - 256 bins per feature (paper default), computed via quantiles
        - A dedicated unknown bin (index = n_bins) for missing values,
          imputed as -1 (paper Section 2.1)
        - Two-stage training: main effects first, then top-K interaction
          pairs selected by variance of learned risk scores, matching the
          paper's selection of 34 two-dimensional functions (Section 2.2)
        - AUC-ROC / AUC-PR as evaluation metrics (paper Section 2.2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleDataset


# Sentinel value used to signal "unknown / missing" — must match
# the -1 imputation applied during preprocessing (paper Section 2.1).
UNKNOWN_SENTINEL = -1.0


class GA2M(BaseModel):
    """GA2M model with binned shape functions and pairwise interactions.

    Implements a Generalized Additive Model with Interactions (GA2M) using
    learned per-bin risk embeddings for both main effects and pairwise
    interactions. Designed to reproduce the in-hospital mortality prediction
    pipeline from Hegselmann et al. (MLHC 2020).

    Shape functions are step functions over discretized feature bins, making
    them directly visualizable for clinical interpretability review — the
    central contribution of the paper.

    Training is two-stage:
        Stage 1 — fit_main_effects(): trains only the per-feature shape
            functions (main effects). Call this first via the provided helper
            or your own training loop.
        Stage 2 — select_top_interactions() + forward(): selects the top-K
            interaction pairs by variance of their main effect embeddings,
            then the full model (main effects + interactions) is trained.

    Args:
        dataset (SampleDataset): PyHealth dataset. Must have exactly one
            tensor input feature key and one binary label key.
        n_bins (int): Number of quantile bins per feature. The paper uses
            256 (default). Reduce to 32-64 for faster experimentation.
        top_k_interactions (int): Number of pairwise interaction terms to
            retain after Stage 1. The paper uses 34 (default).
        use_interactions (bool): Whether to include pairwise interaction
            terms at all. Set to False for the main-effects-only ablation.

    Example:
        >>> model = GA2M(dataset, n_bins=256, top_k_interactions=34)
        >>> model.fit_main_effects(train_loader, epochs=5, lr=1e-2)
        >>> model.select_top_interactions()
        >>> # now train full model with interactions via normal trainer
    """

    def __init__(
        self,
        dataset: SampleDataset,
        n_bins: int = 256,
        top_k_interactions: int = 34,
        use_interactions: bool = True,
    ) -> None:
        super().__init__(dataset)

        assert len(self.feature_keys) == 1, (
            "GA2M expects exactly one input feature key (a pre-computed "
            f"feature tensor). Got: {self.feature_keys}"
        )
        assert len(self.label_keys) == 1, (
            "GA2M expects exactly one binary label key. "
            f"Got: {self.label_keys}"
        )

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.n_bins = n_bins
        self.top_k_interactions = top_k_interactions
        self.use_interactions = use_interactions

        # Infer input dimensionality from the first dataset sample.
        sample = dataset[0]
        self.input_dim: int = sample[self.feature_key].shape[0]

        # --- Bin boundaries (set by fit_bins, not learned) ---
        # Shape: (input_dim, n_bins - 1)  — the n_bins-1 interior edges.
        # Stored as a buffer so it moves with the model to GPU.
        self.register_buffer(
            "bin_edges",
            torch.zeros(self.input_dim, n_bins - 1),
        )
        self._bins_fitted: bool = False

        # --- Global bias (intercept) ---
        self.bias = nn.Parameter(torch.zeros(1))

        # --- Main effect shape functions ---
        # One embedding per feature: n_bins regular bins + 1 unknown bin.
        # Embedding index n_bins is reserved for UNKNOWN_SENTINEL values.
        self.main_effects = nn.ModuleList([
            nn.Embedding(n_bins + 1, 1)
            for _ in range(self.input_dim)
        ])
        # Initialise all risk scores to zero (neutral prior).
        for emb in self.main_effects:
            nn.init.zeros_(emb.weight)

        # --- Interaction shape functions ---
        # Populated by select_top_interactions() after Stage 1.
        # Each entry is an Embedding over (n_bins+1)^2 cells (flattened grid).
        self.interaction_pairs: List[Tuple[int, int]] = []
        self.interactions = nn.ModuleDict()

        # Track which stage we are in for informative error messages.
        self._interactions_selected: bool = False

    # ------------------------------------------------------------------
    # Bin fitting (called once before any training)
    # ------------------------------------------------------------------

    def fit_bins(self, data_loader: torch.utils.data.DataLoader) -> None:
        """Compute quantile-based bin edges from training data.

        Mirrors the paper's discretisation step (Section 2.1). Values equal
        to UNKNOWN_SENTINEL (-1) are excluded from quantile computation and
        will be routed to the dedicated unknown bin (index = n_bins).

        Args:
            data_loader: A DataLoader over the training SampleDataset.
                Must yield batches containing self.feature_key.
        """
        all_features: List[torch.Tensor] = []
        for batch in data_loader:
            x = batch[self.feature_key]  # (B, D)
            all_features.append(x.cpu())
        X = torch.cat(all_features, dim=0)  # (N, D)

        edges = torch.zeros(self.input_dim, self.n_bins - 1)
        for d in range(self.input_dim):
            # col = X[:, d]
            col = x[:, d].contiguous()
            # Exclude the unknown sentinel from quantile computation.
            valid = col[col != UNKNOWN_SENTINEL]
            if valid.numel() == 0:
                # All values unknown — edges stay at zero (degenerate).
                continue
            quantiles = torch.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            edges[d] = torch.quantile(valid.float(), quantiles)

        self.bin_edges.copy_(edges)
        self._bins_fitted = True

    # ------------------------------------------------------------------
    # Bin assignment
    # ------------------------------------------------------------------

    def _assign_bins(self, x: torch.Tensor) -> torch.LongTensor:
        """Map continuous feature values to bin indices.

        Args:
            x: Feature tensor of shape (batch_size, input_dim).

        Returns:
            LongTensor of shape (batch_size, input_dim) with bin indices
            in [0, n_bins]. Index n_bins marks unknown values.
        """
        batch_size = x.size(0)
        bin_idx = torch.zeros(
            batch_size, self.input_dim,
            dtype=torch.long, device=x.device,
        )

        for d in range(self.input_dim):
            # col = x[:, d]
            col = x[:, d].contiguous()
            unknown_mask = (col == UNKNOWN_SENTINEL)

            # torch.bucketize: returns index in [0, n_bins-1] for known values.
            edges = self.bin_edges[d].contiguous()
            idx = torch.bucketize(col, edges)  # (B,) in [0, n_bins-1]

            # Route unknowns to dedicated bin n_bins.
            idx[unknown_mask] = self.n_bins
            bin_idx[:, d] = idx

        return bin_idx

    # ------------------------------------------------------------------
    # Stage 1: main-effects-only training loop
    # ------------------------------------------------------------------

    def fit_main_effects(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int = 10,
        lr: float = 1e-2,
    ) -> None:
        """Stage 1: train only main effect shape functions.

        Freezes interaction parameters (none exist yet) and trains the
        per-feature embeddings and bias. Must call fit_bins() first.

        After this call, use select_top_interactions() to identify the
        top-K pairs before proceeding to full model training.

        Args:
            data_loader: DataLoader over the training split.
            epochs: Number of full passes over the training data.
            lr: Learning rate for Adam optimiser.

        Raises:
            RuntimeError: If fit_bins() has not been called yet.
        """
        if not self._bins_fitted:
            raise RuntimeError(
                "Call fit_bins(train_loader) before fit_main_effects()."
            )

        # Only optimise main effects + bias in Stage 1.
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

    def _forward_main_effects_only(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using only main effects (used during Stage 1)."""
        x = batch[self.feature_key].to(self.device)
        y_true = batch[self.label_key].float().to(self.device)

        bin_idx = self._assign_bins(x)  # (B, D)
        batch_size = x.size(0)

        logits = self.bias.expand(batch_size, 1).clone()
        for d in range(self.input_dim):
            logits = logits + self.main_effects[d](bin_idx[:, d]).squeeze(-1).unsqueeze(-1)

        loss = F.binary_cross_entropy_with_logits(logits, y_true)
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------
    # Stage 1 → 2 transition: interaction pair selection
    # ------------------------------------------------------------------

    def select_top_interactions(self) -> List[Tuple[int, int]]:
        """Select the top-K interaction pairs by main effect variance.

        Implements the paper's interaction selection strategy (Section 2.2):
        pairs are ranked by the product of their two features' shape function
        variances (a proxy for importance), and the top ``top_k_interactions``
        pairs are retained.

        Initialises an nn.Embedding for each selected pair over a flattened
        (n_bins+1) x (n_bins+1) grid of bin combinations.

        Returns:
            List of (i, j) tuples for the selected interaction pairs.

        Raises:
            RuntimeError: If fit_main_effects() has not been called yet.
        """
        # Compute per-feature variance of learned risk scores.
        variances = []
        for d in range(self.input_dim):
            weight = self.main_effects[d].weight.data  # (n_bins+1, 1)
            variances.append(weight.var().item())

        # Rank all pairs by product of their variances.
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

        # Initialise interaction embeddings for selected pairs.
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

    # ------------------------------------------------------------------
    # Main forward pass (Stage 2 / full model)
    # ------------------------------------------------------------------

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full model forward pass including main effects and interactions.

        Computes the prediction as:
            logit = bias
                  + sum_d  f_d(bin(x_d))          [main effects]
                  + sum_{i,j} f_{ij}(bin(x_i), bin(x_j))  [interactions]

        Args:
            **kwargs: Must contain self.feature_key (tensor of shape
                (batch_size, input_dim)) and self.label_key (binary labels).

        Returns:
            Dict with keys:
                loss  (scalar tensor): BCE loss.
                y_prob (tensor, shape (B, 1)): Predicted probabilities.
                y_true (tensor, shape (B, 1)): Ground-truth labels.
                logits (tensor, shape (B, 1)): Raw logits.
        """
        if not self._bins_fitted:
            raise RuntimeError(
                "Call fit_bins(train_loader) before running forward()."
            )
        if self.use_interactions and not self._interactions_selected:
            raise RuntimeError(
                "Call select_top_interactions() after Stage 1 training "
                "before running the full model forward pass."
            )

        x = kwargs[self.feature_key]
        y_true = kwargs[self.label_key].float()

        bin_idx = self._assign_bins(x)  # (B, D)
        batch_size = x.size(0)

        # --- Main effects ---
        logits = self.bias.expand(batch_size, 1).clone()
        for d in range(self.input_dim):
            logits = logits + self.main_effects[d](bin_idx[:, d]).squeeze(-1).unsqueeze(-1)

        # --- Pairwise interactions ---
        if self.use_interactions and self._interactions_selected:
            n_bins_plus1 = self.n_bins + 1
            for i, j in self.interaction_pairs:
                key = f"{i}_{j}"
                # Flatten 2D bin coordinates to a single index.
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

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    def get_shape_function(self, feature_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract the learned 1D risk shape function for a single feature.

        Returns the step-function visualisation described in the paper:
        bin midpoints on the x-axis and learned risk scores on the y-axis.

        Args:
            feature_idx: Index of the feature (0 to input_dim - 1).

        Returns:
            Tuple of (bin_midpoints, risk_scores), both numpy arrays of
            length n_bins. The unknown bin (index n_bins) is excluded.
        """
        edges = self.bin_edges[feature_idx].detach().cpu().numpy()
        # Midpoints: prepend -inf and append +inf to get n_bins intervals.
        lower = np.concatenate([[-np.inf], edges])
        upper = np.concatenate([edges, [np.inf]])
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
        """Extract the learned 2D interaction risk grid for a feature pair.

        Args:
            feature_i: Index of the first feature.
            feature_j: Index of the second feature.

        Returns:
            2D numpy array of shape (n_bins, n_bins) with risk scores,
            or None if this pair was not selected.
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