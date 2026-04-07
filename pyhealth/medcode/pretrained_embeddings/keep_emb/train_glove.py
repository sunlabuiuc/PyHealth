"""Train regularized GloVe embeddings (KEEP Stage 2).

Implements KEEP Stage 2 as described in Algorithm 1 and Appendix A.3
of the paper. This is the core of KEEP: a GloVe-style objective that
learns from patient co-occurrence data, regularized to preserve the
ontological structure captured by Node2Vec in Stage 1.

The total objective (Equation 4 in the paper):

    J(W) = L_GloVe + L_reg

    L_GloVe = sum f(X_ij) * (w_i^T * w~_j + b_i + b~_j - log X_ij)^2

    L_reg = lambda * sum ||w_i - w_i^n2v||^2

Where:
    - X_ij = co-occurrence count of codes i and j across patients
    - f(X_ij) = weighting: (X/x_max)^alpha if X < x_max, else 1
    - w_i, w~_j = two sets of learned GloVe embeddings (context/target)
    - b_i, b~_j = bias terms
    - w_i^n2v = frozen Node2Vec embedding from Stage 1
    - lambda = regularization strength

The regularization term is what makes KEEP more than just GloVe on
medical codes. It prevents the co-occurrence objective from destroying
the ontological structure learned by Node2Vec. The balance is controlled
by lambda: higher lambda = more ontology, lower lambda = more co-occurrence.

Hyperparameters from KEEP paper (Appendix A.3, Table 6):
    embedding_dim=100, lr=0.05, epochs=300, batch_size=1024,
    x_max=75th percentile, alpha=0.75, lambda=1e-3.

    Note: The paper's code uses Adagrad optimizer, but Table 6 does not
    explicitly state the optimizer. The paper's Algorithm 1 says "AdamW"
    but the code uses Adagrad. We default to Adagrad to match the code.

    The KEEP paper's code also uses cosine distance for regularization
    rather than L2 distance. We implement cosine distance to match the
    code, with L2 as a configurable alternative.

Authors: Colton Loew, Desmond Fung, Lookman Olowo, Christiana Beard
Paper: Elhussein et al., "KEEP: Integrating Medical Ontologies with Clinical
       Data for Robust Code Embeddings", CHIL 2025.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CooccurrenceDataset(Dataset):
    """PyTorch Dataset of (i, j, count) triples from a co-occurrence matrix.

    Enumerates all non-zero entries in the co-occurrence matrix as
    training examples for the GloVe objective. Each example is a
    (row_idx, col_idx, count) triple.

    Args:
        matrix: Co-occurrence matrix of shape ``(N, N)``, symmetric,
            from ``build_cooccurrence_matrix``.

    Example:
        >>> dataset = CooccurrenceDataset(matrix)
        >>> i, j, count = dataset[0]
    """

    def __init__(self, matrix: np.ndarray):
        # Extract non-zero entries (upper triangle to avoid duplicates,
        # but GloVe uses both directions so we take all non-zero)
        rows, cols = np.nonzero(matrix)
        self.rows = torch.tensor(rows, dtype=torch.long)
        self.cols = torch.tensor(cols, dtype=torch.long)
        self.counts = torch.tensor(
            matrix[rows, cols], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.counts[idx]


class KeepGloVe(nn.Module):
    """KEEP's regularized GloVe model.

    Standard GloVe architecture (two embedding matrices + biases) with
    an additional regularization term that penalizes drift from the
    Node2Vec initialization.

    The model learns two embedding matrices (U and V, following GloVe
    convention of "context" and "target" embeddings). The final output
    is the average: ``(U + V) / 2``.

    Args:
        vocab_size: Number of unique codes.
        embedding_dim: Dimensionality of embeddings. Default: 100.
        init_embeddings: Node2Vec embeddings to initialize from and
            regularize toward. Shape ``(vocab_size, embedding_dim)``.
            If None, uses random initialization and no regularization.
        lambd: Regularization strength (lambda in the paper).
            Default: 1e-3 (KEEP paper Table 6).
        use_cosine_reg: If True, use cosine distance for regularization
            (matches KEEP code). If False, use L2 distance (matches
            paper Algorithm 1). Default: True.

    Example:
        >>> model = KeepGloVe(5000, 100, init_embeddings=n2v_embs)
        >>> loss = model(row_idx, col_idx, counts)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        init_embeddings: Optional[np.ndarray] = None,
        lambd: float = 1e-3,
        use_cosine_reg: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lambd = lambd
        self.use_cosine_reg = use_cosine_reg

        # Two embedding matrices (GloVe uses separate context/target)
        self.emb_u = nn.Embedding(vocab_size, embedding_dim)
        self.emb_v = nn.Embedding(vocab_size, embedding_dim)
        self.bias_u = nn.Embedding(vocab_size, 1)
        self.bias_v = nn.Embedding(vocab_size, 1)

        # Initialize
        if init_embeddings is not None:
            init_tensor = torch.tensor(init_embeddings, dtype=torch.float32)
            self.emb_u.weight.data.copy_(init_tensor)
            self.emb_v.weight.data.copy_(init_tensor)
            # Store frozen copy for regularization
            self.register_buffer(
                "init_emb", init_tensor.clone()
            )
        else:
            nn.init.uniform_(self.emb_u.weight, -0.5, 0.5)
            nn.init.uniform_(self.emb_v.weight, -0.5, 0.5)
            self.init_emb = None

        nn.init.zeros_(self.bias_u.weight)
        nn.init.zeros_(self.bias_v.weight)

    def forward(
        self,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        counts: torch.Tensor,
        x_max: float = 50.0,
        alpha: float = 0.75,
    ) -> torch.Tensor:
        """Compute the KEEP loss (GloVe + regularization).

        Args:
            row_idx: Batch of row indices, shape ``(batch,)``.
            col_idx: Batch of column indices, shape ``(batch,)``.
            counts: Batch of co-occurrence counts, shape ``(batch,)``.
            x_max: GloVe weighting threshold. Co-occurrences above this
                get weight 1.0; below get ``(count/x_max)^alpha``.
                Default: 50.0 (dynamically adjusted in training loop).
            alpha: GloVe weighting exponent. Default: 0.75
                (KEEP paper Table 6).

        Returns:
            Scalar loss tensor (GloVe loss + regularization loss).
        """
        # Look up embeddings
        u = self.emb_u(row_idx)       # (batch, dim)
        v = self.emb_v(col_idx)       # (batch, dim)
        bu = self.bias_u(row_idx).squeeze(1)  # (batch,)
        bv = self.bias_v(col_idx).squeeze(1)  # (batch,)

        # GloVe weighting function: f(x) = (x/x_max)^alpha if x < x_max
        weights = torch.where(
            counts < x_max,
            (counts / x_max).pow(alpha),
            torch.ones_like(counts),
        )

        # GloVe loss: f(X_ij) * (w_i . w~_j + b_i + b~_j - log X_ij)^2
        dot = (u * v).sum(dim=1)  # (batch,)
        log_counts = torch.log(counts.clamp(min=1.0))
        diff = dot + bu + bv - log_counts
        glove_loss = (weights * diff.pow(2)).mean()

        # Regularization loss
        reg_loss = torch.tensor(0.0, device=glove_loss.device)
        if self.init_emb is not None and self.lambd > 0:
            # Average of U and V (the final embedding)
            avg_u = self.emb_u(row_idx)
            avg_v = self.emb_v(row_idx)
            avg_emb = (avg_u + avg_v) / 2.0
            init = self.init_emb[row_idx]

            if self.use_cosine_reg:
                # Cosine distance: 1 - cosine_similarity
                cos_sim = nn.functional.cosine_similarity(
                    avg_emb, init, dim=1
                )
                reg_loss = self.lambd * (1.0 - cos_sim).mean()
            else:
                # L2 distance (paper Algorithm 1)
                reg_loss = self.lambd * (avg_emb - init).pow(2).mean()

        return glove_loss + reg_loss

    def get_embeddings(self) -> np.ndarray:
        """Return the final KEEP embeddings as a numpy array.

        The final embedding is the average of the two GloVe matrices
        (U + V) / 2, following the standard GloVe convention.

        Returns:
            np.ndarray of shape ``(vocab_size, embedding_dim)``.
        """
        with torch.no_grad():
            emb = (
                self.emb_u.weight.data + self.emb_v.weight.data
            ) / 2.0
        return emb.cpu().numpy()


def train_keep(
    cooc_matrix: np.ndarray,
    init_embeddings: Optional[np.ndarray] = None,
    embedding_dim: int = 100,
    epochs: int = 300,
    batch_size: int = 1024,
    lr: float = 0.05,
    alpha: float = 0.75,
    lambd: float = 1e-3,
    use_cosine_reg: bool = True,
    device: str = "cpu",
    seed: int = 42,
    log_every: int = 50,
) -> np.ndarray:
    """Train KEEP embeddings (regularized GloVe) end-to-end.

    This is the main training function for KEEP Stage 2. It takes a
    co-occurrence matrix and optional Node2Vec initialization, and
    produces the final KEEP embeddings.

    Default hyperparameters are from the KEEP paper (Appendix A.3,
    Table 6).

    Args:
        cooc_matrix: Co-occurrence matrix of shape ``(N, N)`` from
            ``build_cooccurrence_matrix``.
        init_embeddings: Node2Vec embeddings from Stage 1, shape
            ``(N, embedding_dim)``. If None, uses random init and
            no regularization (equivalent to plain GloVe).
        embedding_dim: Dimensionality of embeddings.
            Default: 100 (KEEP paper Table 6).
        epochs: Number of training epochs.
            Default: 300 (KEEP paper Table 6).
        batch_size: Batch size for training.
            Default: 1024 (KEEP paper Table 6).
        lr: Learning rate for Adagrad optimizer.
            Default: 0.05 (KEEP paper Table 6).
        alpha: GloVe weighting exponent.
            Default: 0.75 (KEEP paper Table 6).
        lambd: Regularization strength (lambda).
            Default: 1e-3 (KEEP paper Table 6).
        use_cosine_reg: If True, use cosine distance for regularization
            (matches KEEP code). If False, use L2 (matches paper text).
            Default: True.
        device: Device to train on ("cpu" or "cuda"). Default: "cpu".
        seed: Random seed. Default: 42.
        log_every: Print loss every N epochs. Default: 50.

    Returns:
        np.ndarray: Final KEEP embeddings, shape ``(N, embedding_dim)``.

    Example:
        >>> embeddings = train_keep(
        ...     cooc_matrix,
        ...     init_embeddings=n2v_embeddings,
        ...     epochs=300,
        ...     lambd=1e-3,
        ... )
        >>> embeddings.shape
        (5686, 100)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    vocab_size = cooc_matrix.shape[0]
    if vocab_size == 0:
        return np.zeros((0, embedding_dim), dtype=np.float32)

    # Dynamic x_max: max(50, 75th percentile) per KEEP code
    nonzero_vals = cooc_matrix[cooc_matrix > 0]
    if len(nonzero_vals) > 0:
        x_max = max(50.0, float(np.percentile(nonzero_vals, 75)))
    else:
        x_max = 50.0

    logger.info(
        "Training KEEP GloVe: vocab=%d, dim=%d, epochs=%d, lr=%.4f, "
        "lambda=%.1e, x_max=%.1f, alpha=%.2f, cosine_reg=%s",
        vocab_size,
        embedding_dim,
        epochs,
        lr,
        lambd,
        x_max,
        alpha,
        use_cosine_reg,
    )

    # Build dataset and dataloader
    dataset = CooccurrenceDataset(cooc_matrix)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    # Build model
    model = KeepGloVe(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        init_embeddings=init_embeddings,
        lambd=lambd,
        use_cosine_reg=use_cosine_reg,
    ).to(device)

    # Adagrad optimizer (matches KEEP code)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_batches = 0

        for row_idx, col_idx, counts in dataloader:
            row_idx = row_idx.to(device)
            col_idx = col_idx.to(device)
            counts = counts.to(device)

            loss = model(row_idx, col_idx, counts, x_max=x_max, alpha=alpha)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        if epoch % log_every == 0 or epoch == 1:
            avg_loss = total_loss / max(num_batches, 1)
            logger.info("Epoch %d/%d, loss=%.6f", epoch, epochs, avg_loss)

    embeddings = model.get_embeddings()
    logger.info("KEEP training complete: %s embeddings", embeddings.shape)
    return embeddings
