"""CMA Prior Encoder for PyHealth.

Trainable MLP that learns a prior distribution over treatment effects
from untrusted trial data. Used as the first layer of conformal
meta-analysis in Kaul & Gordon (2024).

The encoder maps trial features X to a predicted effect M(X) = w^T phi(X)
via a ReLU deep network. Unlike the main CMA algorithm, this component
is trained via standard gradient descent on observed effects from
"untrusted" trials (observational data, non-placebo RCTs, etc.).

After training, the encoder produces prior mean values M for each
trial in the trusted set, which feed into the conformal meta-analysis
model.

Reference:
    Kaul, S.; and Gordon, G. J. 2024. Meta-Analysis with Untrusted Data.
    In Proceedings of Machine Learning Research, volume 259, 563-593.

Example:
    >>> from pyhealth.datasets import AmiodaroneTrialDataset
    >>> from pyhealth.tasks import ConformalMetaAnalysisTask
    >>> from pyhealth.models import CMAPriorEncoder
    >>>
    >>> dataset = AmiodaroneTrialDataset(root="./data/amiodarone")
    >>> task = ConformalMetaAnalysisTask(
    ...     target_column="log_relative_risk",
    ...     split_column="split",
    ...     split_value="untrusted",
    ... )
    >>> samples = dataset.set_task(task)
    >>> encoder = CMAPriorEncoder(
    ...     dataset=samples,
    ...     hidden_dims=[64, 32],
    ...     embed_dim=16,
    ... )
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.models.base_model import BaseModel


class CMAPriorEncoder(BaseModel):
    """ReLU deep network for learning a prior over treatment effects.

    Maps trial features to a predicted effect via an embedding:
        phi(x) = ReLU-MLP(x)        # learned embedding
        mu(x) = w^T phi(x) + b      # linear head (predicted M)

    Standard regression training with MSE loss. The learned embedding
    also defines an implicit kernel kappa(x, x') = phi(x)^T phi(x'),
    used by the CMA model.

    Args:
        dataset: SampleDataset produced by
            ``ConformalMetaAnalysisTask``.
        hidden_dims: Sizes of the hidden layers. Defaults to
            ``[64, 32]``.
        embed_dim: Dimension of the learned embedding phi(x).
            Defaults to 16.
        dropout: Dropout probability between layers. Defaults to 0.0.
        feature_key: Name of the input feature in the batch.
            Defaults to "features".
        target_key: Name of the target in the batch. Defaults to
            "true_effect" to match ConformalMetaAnalysisTask's
            output_schema. For amiodarone, the task should map
            ``log_relative_risk`` to ``true_effect`` in its output.

    Attributes:
        hidden_dims: Hidden layer sizes.
        embed_dim: Embedding dimension.
        encoder: The ReLU-MLP producing phi(x).
        head: Linear head producing mu(x) from phi(x).
        mode: Always "regression".
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 16,
        dropout: float = 0.0,
        feature_key: str = "features",
        target_key: str = "true_effect",
    ) -> None:
        super().__init__(dataset=dataset)

        if hidden_dims is None:
            hidden_dims = [64, 32]
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.hidden_dims = list(hidden_dims)
        self.embed_dim = embed_dim
        self.dropout_p = dropout
        self.feature_key = feature_key
        self.target_key = target_key
        self.mode = "regression"

        # Infer input dimension from the first sample
        input_dim = self._infer_input_dim(dataset)
        self.input_dim = input_dim

        self.encoder = self._build_encoder(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            embed_dim=embed_dim,
            dropout=dropout,
        )
        self.head = nn.Linear(embed_dim, 1)

    @staticmethod
    def _infer_input_dim(dataset: SampleDataset) -> int:
        """Peek at the first sample to determine the feature size."""
        try:
            first = dataset[0]
        except (IndexError, TypeError):
            raise ValueError(
                "Cannot infer feature dimension from an empty dataset."
            )
        features = first.get("features")
        if features is None:
            raise KeyError(
                "Sample is missing a 'features' key. "
                "Check the task's input_schema."
            )
        if isinstance(features, torch.Tensor):
            return int(features.numel())
        return int(len(features))

    @staticmethod
    def _build_encoder(
        input_dim: int,
        hidden_dims: List[int],
        embed_dim: int,
        dropout: float,
    ) -> nn.Sequential:
        """Construct the ReLU-MLP."""
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, embed_dim))
        return nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # PyHealth forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        features: torch.Tensor,
        true_effect: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the encoder and prediction head.

        Args:
            features: Input feature tensor of shape (batch_size, d).
            true_effect: Optional target tensor of shape
                (batch_size, 1) for computing MSE loss.

        Returns:
            Dict with keys:
                - ``y_pred``: predicted effect, shape (batch_size, 1)
                - ``embedding``: learned embedding phi(x), shape
                  (batch_size, embed_dim)
                - ``loss``: MSE loss (only if ``true_effect`` is
                  provided)
                - ``y_true``: the labels, if provided
        """
        if features.dim() == 1:
            features = features.unsqueeze(0)

        embedding = self.encoder(features.float())
        prediction = self.head(embedding)

        out: Dict[str, torch.Tensor] = {
            "y_pred": prediction,
            "embedding": embedding,
        }

        if true_effect is not None:
            target = true_effect.float()
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            loss = nn.functional.mse_loss(prediction, target)
            out["loss"] = loss
            out["y_true"] = target

        return out

    def forward_from_embedding(
        self, embedding: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Run only the head on a pre-computed embedding.

        Useful for interpretability methods that operate on the
        intermediate representation.

        Args:
            embedding: Embedding tensor, shape (batch_size, embed_dim).

        Returns:
            Dict with ``y_pred`` and ``embedding``.
        """
        return {
            "y_pred": self.head(embedding),
            "embedding": embedding,
        }

    # ------------------------------------------------------------------
    # Convenience methods for CMA integration
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_prior_mean(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Produce prior mean M(x) for a batch of trials.

        Args:
            features: Feature tensor of shape (batch_size, d) or (d,).

        Returns:
            Tensor of predicted effects, shape (batch_size,).
        """
        self.eval()
        if features.dim() == 1:
            features = features.unsqueeze(0)
        out = self.forward(features=features)
        return out["y_pred"].squeeze(-1)

    @torch.no_grad()
    def predict_kernel_matrix(
        self, X1: torch.Tensor, X2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute the implicit kernel matrix phi(X1)^T phi(X2).

        This can be used as the kappa input to the CMA model, giving
        a feature-learned kernel instead of a fixed RBF.

        Args:
            X1: First feature batch, shape (n1, d).
            X2: Second feature batch, shape (n2, d). If None, uses X1.

        Returns:
            Kernel matrix of shape (n1, n2).
        """
        self.eval()
        if X2 is None:
            X2 = X1
        phi1 = self.encoder(X1.float())
        phi2 = self.encoder(X2.float())
        return phi1 @ phi2.T
