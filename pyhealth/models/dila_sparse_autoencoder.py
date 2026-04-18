"""Sparse Autoencoder module for DILA.

Implements equations 1-4 from:
    DILA: Dictionary Label Attention for Interpretable ICD Coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder with elastic-net regularization for learning dictionary features.

    Maps dense PLM token embeddings into a larger sparse feature space.
    The decoder bias serves as the centering term so that the autoencoder
    learns to represent residuals around the data mean.

    Architecture (Eq. 1-4):
        x̄ = x - b_d                         (center input by decoder bias)
        f  = ReLU(W_e x̄ + b_e)              (sparse feature activations)
        x̂  = W_d f + b_d                     (reconstruct centered + bias)

        L_saenc = mean(||x - x̂||²₂)
                  + lambda_l1 * mean(||f||₁)
                  + lambda_l2 * mean(||f||²₂)

    Args:
            input_dim: The dimensionality of the input PLM embeddings.
            dict_size: The number of features in the sparse dictionary.
            lambda_l1: L1 sparsity penalty coefficient. Default: 1e-4.
            lambda_l2: L2 weight-decay coefficient. Default: 1e-5.
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        lambda_l1: float = 1e-4,
        lambda_l2: float = 1e-5,
    ):
        """Initializes the SparseAutoencoder module.

        Args:
            input_dim: The dimensionality of the input PLM embeddings.
            dict_size: The number of features in the sparse dictionary.
        """
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, dict_size)
        self.decoder = nn.Linear(dict_size, input_dim)
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.dict_size = dict_size
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor):
        """Performs a forward pass to generate sparse features and reconstruction.

        Args:
            x: Input dense embeddings of shape [batch, seq_len, input_dim].

        Returns:
            A tuple containing:
                - features: Sparse dictionary features [batch, seq_len, dict_size].
                - reconstructed: Reconstructed embeddings [batch, seq_len, input_dim].
                - loss_dict: Dictionary containing 'loss_saenc', 'loss_recon', and 'loss_l1'.
        """
        # f = ReLU(W_e * x + b_e)
        features = F.relu(self.encoder(x))
        reconstructed = self.decoder(features)

        # Add the loss math here
        loss_recon = F.mse_loss(reconstructed, x)
        loss_l1 = features.abs().mean()
        loss_l2 = (features**2).mean()
        loss_saenc = loss_recon + self.lambda_l1 * loss_l1 + self.lambda_l2 * loss_l2

        loss_dict = {
            "loss_saenc": loss_saenc,
            "loss_recon": loss_recon,
            "loss_l1": loss_l1,
        }
        return features, reconstructed, loss_dict

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input embeddings directly into sparse features without reconstructing.

        Args:
            x: Input dense embeddings of shape [batch, seq_len, input_dim].

        Returns:
            Sparse dictionary features of shape [batch, seq_len, dict_size].
        """
        return F.relu(self.encoder(x))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Normalizes the decoder weights to have unit column norms.

        This prevents the autoencoder from cheating the sparsity penalty by
        scaling down feature activations and scaling up decoder weights.
        """
        norms = torch.norm(self.decoder.weight, dim=0, keepdim=True)
        self.decoder.weight.copy_(self.decoder.weight / torch.clamp(norms, min=1e-8))
