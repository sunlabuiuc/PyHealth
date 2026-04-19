from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from typing import Any
from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class WordBasisLinearModel(BaseModel):
    """Linear classifier on frozen embeddings with word-basis explanations.

    This model reproduces the core two-step idea from
    "Representing visual classification as a linear combination of words":

    1. Learn a linear classifier over precomputed frozen embeddings.
    2. Approximate the learned classifier weight vector as a linear
       combination of fixed word embeddings.

    Args:
        dataset: PyHealth SampleDataset.
        input_dim: Dimension of the precomputed embedding vector.
        feature_key: Optional feature field name. If None, the model expects
            exactly one input feature in the dataset schema and uses it.
        ridge_lambda: Default ridge penalty used when solving for word-basis
            coefficients.

        Example:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import WordBasisLinearModel
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "embedding": [0.1] * 8,
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "embedding": [0.0] * 8,
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"embedding": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="word_basis_linear_model_example",
        ... )
        >>> model = WordBasisLinearModel(
        ...     dataset=dataset,
        ...     input_dim=8,
        ...     feature_key="embedding",
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        input_dim: int,
        feature_key: Optional[str] = None,
        ridge_lambda: float = 0.0,
    ) -> None:
        super().__init__(dataset=dataset)
        self.mode = "binary"
        if len(self.label_keys) != 1:
            raise ValueError(
                "WordBasisLinearModel currently supports exactly one label key."
            )
        self.label_key = self.label_keys[0]

        if feature_key is None:
            if len(self.feature_keys) != 1:
                raise ValueError(
                    "feature_key was not provided, but the dataset has "
                    f"{len(self.feature_keys)} feature keys. Please pass feature_key "
                    "explicitly."
                )
            self.feature_key = self.feature_keys[0]
        else:
            if feature_key not in self.feature_keys:
                raise ValueError(
                    f"feature_key '{feature_key}' not found in dataset feature keys: "
                    f"{self.feature_keys}"
                )
            self.feature_key = feature_key

        if self.get_output_size() != 1:
            raise ValueError(
                "WordBasisLinearModel currently supports binary classification only."
            )

        self.input_dim = input_dim
        self.ridge_lambda = ridge_lambda
        self.classifier = nn.Linear(input_dim, 1, bias=False)

    def _get_input_tensor(self, feature: Any) -> torch.Tensor:
        """Extracts the dense tensor from a PyHealth feature payload."""
        if isinstance(feature, torch.Tensor):
            x = feature
        elif isinstance(feature, (tuple, list)):
            if len(feature) == 0 or not isinstance(feature[0], torch.Tensor):
                raise TypeError(
                    "Expected feature payload to contain a tensor as the first item."
                )
            x = feature[0]
        else:
            raise TypeError(
                f"Unsupported feature type for {self.feature_key}: {type(feature)}"
            )

        x = x.to(self.device).float()

        if x.ndim != 2:
            raise ValueError(
                f"Expected feature tensor to have shape (batch_size, input_dim), "
                f"but got shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, but got tensor with shape "
                f"{tuple(x.shape)}."
            )
        return x

    def _prepare_labels(self, y_true: torch.Tensor) -> torch.Tensor:
        """Normalizes labels to shape (batch_size, 1)."""
        y_true = y_true.to(self.device).float()
        if y_true.ndim == 1:
            y_true = y_true.unsqueeze(1)
        elif y_true.ndim == 2 and y_true.shape[1] == 1:
            pass
        else:
            raise ValueError(
                f"Expected labels of shape (batch_size,) or (batch_size, 1), "
                f"but got {tuple(y_true.shape)}."
            )
        return y_true

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass for binary classification on dense embeddings."""
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing required feature key: {self.feature_key}")

        x = self._get_input_tensor(kwargs[self.feature_key])
        logit = self.classifier(x)
        y_prob = self.prepare_y_prob(logit)

        result = {
            "logit": logit,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = self._prepare_labels(kwargs[self.label_key])
            loss = self.get_loss_function()(logit, y_true)
            result["loss"] = loss
            result["y_true"] = y_true

        return result

    def forward_from_embedding(
        self,
        feature_embeddings: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass that bypasses feature processing.

        This is useful for interpretability-style workflows where the input is
        already in embedding space.
        """
        kwargs = {self.feature_key: feature_embeddings}
        if y is not None:
            kwargs[self.label_key] = y
        return self.forward(**kwargs)

    def get_classifier_weight(self) -> torch.Tensor:
        """Returns the learned classifier weight vector of shape (input_dim,)."""
        return self.classifier.weight.squeeze(0)

    def fit_word_basis(
        self,
        word_embeddings: torch.Tensor,
        ridge_lambda: Optional[float] = None,
    ) -> torch.Tensor:
        """Solves for word coefficients that reconstruct the classifier weight.

        Args:
            word_embeddings: Tensor of shape (num_words, input_dim).
            ridge_lambda: Optional ridge penalty. If None, uses self.ridge_lambda.

        Returns:
            Tensor of shape (num_words,) containing the word coefficients.
        """
        if ridge_lambda is None:
            ridge_lambda = self.ridge_lambda

        word_embeddings = word_embeddings.to(self.device).float()

        if word_embeddings.ndim != 2:
            raise ValueError(
                "word_embeddings must have shape (num_words, input_dim)."
            )
        if word_embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected word_embeddings.shape[1] == {self.input_dim}, but got "
                f"{word_embeddings.shape[1]}."
            )

        beta = self.get_classifier_weight()  # (input_dim,)
        num_words = word_embeddings.shape[0]

        if ridge_lambda > 0:
            gram = word_embeddings @ word_embeddings.T
            rhs = word_embeddings @ beta
            eye = torch.eye(num_words, device=self.device, dtype=word_embeddings.dtype)
            coeffs = torch.linalg.solve(gram + ridge_lambda * eye, rhs)
        else:
            # Solve W^T c ≈ beta in least-squares sense.
            coeffs = torch.linalg.lstsq(word_embeddings.T, beta).solution

        return coeffs

    def reconstruct_from_word_basis(
        self,
        word_embeddings: torch.Tensor,
        word_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstructs classifier weights from word embeddings and coefficients."""
        word_embeddings = word_embeddings.to(self.device).float()
        word_coeffs = word_coeffs.to(self.device).float()

        if word_embeddings.ndim != 2:
            raise ValueError(
                "word_embeddings must have shape (num_words, input_dim)."
            )
        if word_coeffs.ndim != 1:
            raise ValueError("word_coeffs must have shape (num_words,).")
        if word_embeddings.shape[0] != word_coeffs.shape[0]:
            raise ValueError(
                "word_embeddings and word_coeffs must agree on num_words."
            )

        return word_coeffs @ word_embeddings

    def compute_word_basis_cosine_similarity(
        self,
        word_embeddings: torch.Tensor,
        word_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """Computes cosine similarity between true and reconstructed weights."""
        beta = self.get_classifier_weight()
        beta_hat = self.reconstruct_from_word_basis(word_embeddings, word_coeffs)

        cosine = nn.CosineSimilarity(dim=0)
        return cosine(beta, beta_hat)

    def explain_words(
        self,
        word_embeddings: torch.Tensor,
        word_list: Sequence[str],
        ridge_lambda: Optional[float] = None,
        sort_by_abs: bool = True,
    ) -> List[Tuple[str, float]]:
        """Returns (word, coefficient) pairs for interpretation."""
        coeffs = self.fit_word_basis(
            word_embeddings=word_embeddings,
            ridge_lambda=ridge_lambda,
        )

        if len(word_list) != coeffs.shape[0]:
            raise ValueError(
                f"word_list has length {len(word_list)}, but coeffs has length "
                f"{coeffs.shape[0]}."
            )

        pairs = list(zip(word_list, coeffs.detach().cpu().tolist()))
        if sort_by_abs:
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs