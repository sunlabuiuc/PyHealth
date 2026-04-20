# Contributor: Nikhil Ajit
# NetID/Email: najit2@illinois.edu
# Paper Title: DILA: Dictionary Label Attention for Mechanistic Interpretability in High-dimensional Multi-label Medical Coding Prediction
# Paper Link: https://arxiv.org/abs/2409.10504
# Description: Implementation of the DILA model utilizing a sparse autoencoder 
# and a globally interpretable dictionary projection matrix for medical coding.

import torch
import torch.nn as nn
import pytest
from typing import Dict, Optional


class DummyDILA(nn.Module):
    """Dummy implementation of the DILA model for testing purposes.

    Attributes:
        encoder (nn.Linear): Simulated sparse encoder.
        sparse_projection (nn.Parameter): Simulated sparse projection matrix.
        fc (nn.Linear): Final classification layer.
    """

    def __init__(self, embedding_dim: int, dictionary_size: int, num_labels: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(embedding_dim, dictionary_size)
        self.sparse_projection = nn.Parameter(torch.randn(dictionary_size, num_labels))
        self.fc = nn.Linear(embedding_dim, num_labels)

    def forward(
        self, x_note: torch.Tensor, y_true: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for the dummy DILA model.

        Args:
            x_note (torch.Tensor): Input token embeddings.
            y_true (torch.Tensor, optional): Ground truth labels. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: Output dictionary containing probabilities, 
                true labels, and loss if ground truth is provided.
        """
        f = torch.relu(self.encoder(x_note))
        attention = torch.softmax(torch.matmul(f, self.sparse_projection), dim=1)
        x_att = torch.matmul(attention.transpose(-2, -1), x_note)
        logits = self.fc(x_att)

        logits = logits.mean(dim=1)

        output = {"y_prob": torch.sigmoid(logits), "y_true": y_true}
        if y_true is not None:
            output["loss"] = nn.BCEWithLogitsLoss()(logits, y_true.float())
            
        return output


def test_dila_forward() -> None:
    """Tests the forward pass, output shapes, and gradients of the DILA model."""
    batch_size = 4
    sequence_length = 128
    embedding_dim = 768
    dictionary_size = 100
    num_labels = 50

    x_note = torch.randn(batch_size, sequence_length, embedding_dim)
    y_true = torch.empty(batch_size, num_labels).random_(2)

    model = DummyDILA(
        embedding_dim=embedding_dim,
        dictionary_size=dictionary_size,
        num_labels=num_labels
    )

    outputs = model(x_note=x_note, y_true=y_true)

    assert "y_prob" in outputs, "Output dictionary is missing 'y_prob'."
    assert "y_true" in outputs, "Output dictionary is missing 'y_true'."
    assert "loss" in outputs, "Output dictionary is missing 'loss'."

    expected_shape = (batch_size, num_labels)
    actual_shape = outputs["y_prob"].shape
    assert actual_shape == expected_shape, (
        f"Expected y_prob shape {expected_shape}, got {actual_shape}."
    )

    loss = outputs["loss"]
    loss.backward()

    assert model.sparse_projection.grad is not None, (
        "Gradients did not compute for sparse_projection."
    )
    assert torch.sum(torch.abs(model.sparse_projection.grad)) > 0, (
        "Gradients for sparse_projection are zero."
    )