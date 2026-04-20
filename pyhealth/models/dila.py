# Contributor: Nikhil Ajit
# NetID/Email: najit2@illinois.edu
# Paper Title: DILA: Dictionary Label Attention for Mechanistic Interpretability in High-dimensional Multi-label Medical Coding Prediction
# Paper Link: https://arxiv.org/abs/2409.10504
# Description: Implementation of the DILA model utilizing a sparse autoencoder 
# and a globally interpretable dictionary projection matrix for medical coding.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from pyhealth.models.base_model import BaseModel
from pyhealth.datasets import SampleEHRDataset


class DILA(BaseModel):
    """Dictionary Label Attention (DILA) Model for Medical Coding.

    This model implements the Dictionary Label Attention mechanism to predict
    medical codes from clinical sequences. It uses a sparse autoencoder to 
    disentangle dense embeddings into sparse, interpretable dictionary features,
    which are then projected to the label space.

    Attributes:
        feature_keys (List[str]): Keys to access input features in the dataset.
        label_key (str): Key to access the ground truth labels.
        mode (str): Mode of the task, e.g., "multilabel".
        embedding_dim (int): Dimension of the token embeddings.
        dictionary_size (int): Number of sparse dictionary features (m).
        sparsity_penalty (float): Penalty weight for the L1/L2 regularization.
        simulated_plm (nn.Embedding): Embedding layer simulating PLM token features.
        encoder_weight (nn.Linear): Linear projection for the sparse encoder.
        decoder_weight (nn.Linear): Linear projection for the sparse decoder.
        decoder_bias (nn.Parameter): Bias term for the sparse autoencoder.
        sparse_projection (nn.Parameter): Globally interpretable projection matrix.
        fc (nn.Linear): Final linear decision layer for predictions.
        
    Example:
        >>> from pyhealth.models import DILA
        >>> model = DILA(
        ...     dataset=dataset,
        ...     feature_keys=["conditions"],
        ...     label_key="label",
        ...     mode="multilabel"
        ... )
        >>> # kwargs must include the features and labels
        >>> outputs = model(conditions=torch.randn(4, 128, 768), label=torch.empty(4, 50).random_(2))
        >>> loss = outputs["loss"]
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        feature_keys: Optional[List[str]] = None,
        label_key: Optional[str] = None,
        mode: Optional[str] = None,
        embedding_dim: int = 768,
        dictionary_size: int = 6088,
        sparsity_penalty: float = 1e-6,
        **kwargs: Any
    ) -> None:
        super(DILA, self).__init__(dataset=dataset, **kwargs)

        self.feature_keys = feature_keys or ["conditions"]
        self.label_key = label_key or "label"
        self.mode = mode or "multilabel"

        self.embedding_dim = embedding_dim
        self.dictionary_size = dictionary_size
        self.sparsity_penalty = sparsity_penalty

        self.simulated_plm = nn.Embedding(5000, embedding_dim, padding_idx=0)

        self.encoder_weight = nn.Linear(embedding_dim, dictionary_size)
        self.decoder_weight = nn.Linear(dictionary_size, embedding_dim)
        self.decoder_bias = nn.Parameter(torch.zeros(embedding_dim))

        num_labels = self.get_output_size()
        self.sparse_projection = nn.Parameter(torch.randn(dictionary_size, num_labels))

        self.fc = nn.Linear(embedding_dim, num_labels)

    def sparse_autoencoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Disentangles dense embeddings into sparse features.

        Args:
            x (torch.Tensor): Dense token embeddings of shape (batch, seq_len, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the sparse 
                dictionary features and the reconstructed dense embeddings.
        """
        x_bar = x - self.decoder_bias
        f = torch.relu(self.encoder_weight(x_bar))
        x_hat = self.decoder_weight(f) + self.decoder_bias
        return f, x_hat

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """Forward pass for the DILA model.

        Args:
            **kwargs: Keyword arguments containing the input features and labels.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the total loss, 
                predicted probabilities, and true labels.
        """
        input_sequence = kwargs[self.feature_keys[0]]

        if input_sequence.dim() == 3:
            input_sequence = input_sequence.squeeze(-1)

        x = self.simulated_plm(input_sequence)
        f, x_hat = self.sparse_autoencoder(x)

        attention_scores = torch.matmul(f, self.sparse_projection)
        a_laat = torch.softmax(attention_scores, dim=1)

        x_att = torch.matmul(a_laat.transpose(-2, -1), x)
        x_att_pooled = x_att.mean(dim=1)
        logits = self.fc(x_att_pooled)

        mse_loss = nn.MSELoss()(x_hat, x)
        l1_loss = torch.norm(f, p=1)
        l2_loss = torch.norm(f, p=2) ** 2
        sae_loss = mse_loss + self.sparsity_penalty * (l1_loss + l2_loss)

        y_true = kwargs[self.label_key].float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, y_true)

        total_loss = bce_loss + sae_loss

        return {
            "loss": total_loss,
            "y_prob": torch.sigmoid(logits),
            "y_true": y_true
        }