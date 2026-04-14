"""
KEEP-lite: A minimal co-occurrence + embedding model for
readmission prediction.

Extended with:
- Optional Frequency-Aware Regularization
- Lightweight GloVe-style embedding pretraining
- Mean pooled supervised readmission prediction
"""

import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Any

import torch
import torch.nn as nn

from pyhealth.models import BaseModel


class KEEP(BaseModel):
    """
    KEEP-lite implementation.

    This model learns medical code embeddings using a lightweight
    co-occurrence objective and supports optional frequency-aware
    regularization. The learned embeddings are used for supervised
    readmission prediction via mean pooling.

    Args:
        dataset: PyHealth dataset instance. Must contain the
            "conditions" feature in `input_processors`.
        embedding_dim: Dimension of embedding vectors.
        lambda_base: Base regularization strength.
        use_frequency_regularization: Whether to apply
            frequency-aware regularization during pretraining.

    Example:
        >>> model = KEEP(dataset, embedding_dim=128)
        >>> output = model(conditions=batch["conditions"],
        ...                label=batch["label"])
    """

    def __init__(
        self,
        dataset,
        embedding_dim: int = 128,
        lambda_base: float = 0.1,
        use_frequency_regularization: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)

        if "conditions" not in dataset.input_processors:
            raise ValueError(
                "KEEP requires 'conditions' feature in dataset."
            )

        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.lambda_base = lambda_base
        self.use_frequency_regularization = use_frequency_regularization

        processor = dataset.input_processors["conditions"]

        # ----------------------------------------------------------
        # Robust vocabulary size detection (handles:
        #  - attribute
        #  - method
        #  - vocab object
        #  - legacy code_vocab_size
        # ----------------------------------------------------------
        if hasattr(processor, "code_vocab_size"):
            vocab_size = processor.code_vocab_size

        elif hasattr(processor, "vocab_size"):
            attr = processor.vocab_size
            vocab_size = attr() if callable(attr) else attr

        elif hasattr(processor, "get_vocab_size"):
            vocab_size = processor.get_vocab_size()

        elif hasattr(processor, "vocab"):
            vocab_size = len(processor.vocab)

        else:
            raise AttributeError(
                "Cannot determine vocabulary size from processor."
            )

        vocab_size = int(vocab_size)
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0,
        )

        self.classifier = nn.Linear(embedding_dim, 1)
        self.loss_fn = nn.BCELoss()

        # Frequency-aware components
        self.code_frequencies: torch.Tensor | None = None
        self.lambda_vector: torch.Tensor | None = None

    # ==========================================================
    # PART A — Sparse Co-occurrence Builder
    # ==========================================================
    def build_cooccurrence(
        self, samples: List[Dict[str, Any]]
    ) -> Dict:
        """Build sparse visit-level co-occurrence counts.

        Args:
            samples: List of dataset samples containing "conditions".

        Returns:
            Dictionary mapping (code_i, code_j) to co-occurrence count.
        """
        cooccur = defaultdict(int)

        for sample in samples:
            codes = sample["conditions"]

            if isinstance(codes, torch.Tensor):
                codes = codes.detach().cpu().tolist()

            codes = [c for c in codes if c != 0]

            if len(codes) < 2:
                continue

            unique_codes = list(set(codes))
            if len(unique_codes) < 2:
                continue

            for i, j in combinations(sorted(unique_codes), 2):
                cooccur[(int(i), int(j))] += 1

        return cooccur

    # ==========================================================
    # PART B — Frequency Computation
    # ==========================================================
    def compute_code_frequencies(
        self, samples: List[Dict[str, Any]]
    ) -> None:
        """Compute per-code frequency across samples.

        Args:
            samples: List of dataset samples.

        Sets:
            self.code_frequencies
            self.lambda_vector
        """
        freq = torch.zeros(self.vocab_size)

        for sample in samples:
            codes = sample["conditions"]

            if isinstance(codes, torch.Tensor):
                codes = codes.detach().cpu().tolist()

            codes = list(set([c for c in codes if c != 0]))

            for c in codes:
                freq[int(c)] += 1

        self.code_frequencies = freq
        self.lambda_vector = self.lambda_base / torch.sqrt(
            freq + 1.0
        )

    # ==========================================================
    # PART C — Lightweight GloVe-style Pretraining
    # ==========================================================
    def pretrain_embeddings(
        self,
        samples: List[Dict[str, Any]],
        epochs: int = 3,
        lr: float = 1e-3,
    ) -> None:
        """Pretrain embeddings using co-occurrence objective.

        Minimizes:
            ( dot(w_i, w_j) - log(count + 1) )^2

        If frequency regularization is enabled:
            + lambda_i ||w_i||^2 + lambda_j ||w_j||^2

        Args:
            samples: Training samples.
            epochs: Number of pretraining epochs.
            lr: Learning rate.
        """
        print("Building co-occurrence matrix...")
        cooccur = self.build_cooccurrence(samples)

        if len(cooccur) == 0:
            print("No co-occurring condition pairs found.")
            return

        if self.use_frequency_regularization:
            print("Computing code frequencies...")
            self.compute_code_frequencies(samples)

        optimizer = torch.optim.Adam(
            self.embedding.parameters(), lr=lr
        )

        self.embedding.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for (i, j), count in cooccur.items():
                wi = self.embedding.weight[i]
                wj = self.embedding.weight[j]

                dot = torch.dot(wi, wj)
                target = dot.new_tensor(math.log(count + 1.0))
                glove_loss = (dot - target) ** 2

                if self.use_frequency_regularization:
                    lambda_i = self.lambda_vector[i]
                    lambda_j = self.lambda_vector[j]

                    reg_loss = (
                        lambda_i * torch.norm(wi, p=2) ** 2
                        + lambda_j * torch.norm(wj, p=2) ** 2
                    )

                    loss = glove_loss + reg_loss
                else:
                    loss = glove_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(
                f"[Pretrain] Epoch {epoch+1}/{epochs} "
                f"| Loss: {total_loss:.4f}"
            )

        print("Pretraining complete.\n")

    # ==========================================================
    # PART D — Supervised Forward
    # ==========================================================
    def forward(
        self,
        conditions: torch.Tensor,
        x: torch.Tensor = None,
        label: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for readmission prediction."""

        if conditions is None and x is not None:
            conditions = x

        # ----------------------------------------------------------
        # Resolve label key dynamically (PyHealth compatibility)
        # ----------------------------------------------------------
        if label is None:
            if y is not None:
                label = y
            elif "readmission" in kwargs:
                label = kwargs["readmission"]
            elif "label" in kwargs:
                label = kwargs["label"]

        x = self.embedding(conditions)

        mask = (conditions != 0).unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts

        logits = self.classifier(pooled).squeeze(-1)
        y_prob = torch.sigmoid(logits)

        if label is not None:
            label = label.view(-1)  # flatten to [batch_size]
            loss = self.loss_fn(y_prob, label.float())
            return {
                "loss": loss,
                "y_prob": y_prob,
                "y_true": label,
            }

        return {"y_prob": y_prob}