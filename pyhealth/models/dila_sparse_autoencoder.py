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
        input_dim: Dimensionality of the input embeddings (e.g. 768 for RoBERTa).
        dict_size: Number of dictionary features (m). Typically alpha * input_dim.
        lambda_l1: L1 sparsity coefficient. Default: 1e-4.
        lambda_l2: L2 weight-decay coefficient. Default: 1e-5.

    Examples:
        >>> sae = SparseAutoencoder(input_dim=768, dict_size=4096)
        >>> x = torch.randn(32, 768)
        >>> f, x_hat, losses = sae(x)
        >>> f.shape
        torch.Size([32, 4096])
        >>> x_hat.shape
        torch.Size([32, 768])
        >>> set(losses.keys())
        {'loss_saenc', 'loss_recon', 'loss_l1'}
    """

    def __init__(
        self,
        input_dim: int,
        dict_size: int,
        lambda_l1: float = 1e-4,
        lambda_l2: float = 1e-5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dict_size = dict_size
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2

        self.encoder = nn.Linear(input_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, input_dim, bias=True)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs to sparse non-negative feature activations.

        Centers the input by the decoder bias before encoding, so the encoder
        learns to represent deviations from the mean activation pattern.

        Args:
            x: Input tensor of shape (..., input_dim).

        Returns:
            Sparse feature tensor of shape (..., dict_size). All values >= 0.
        """
        x_centered = x - self.decoder.bias
        return F.relu(self.encoder(x_centered))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """Forward pass computing sparse features, reconstruction, and losses.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Tuple of:
                f (Tensor): Sparse feature activations of shape (batch, dict_size).
                    All values are non-negative (ReLU output).
                x_hat (Tensor): Reconstructed input of shape (batch, input_dim).
                loss_dict (dict): Loss components with keys:
                    - "loss_saenc": Combined SAE loss (scalar).
                    - "loss_recon": Reconstruction loss (scalar).
                    - "loss_l1": L1 sparsity loss before lambda scaling (scalar).
        """
        f = self.encode(x)
        x_hat = self.decoder(f)

        # Mean of per-sample squared L2 reconstruction error
        loss_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        # Mean of per-sample L1 and squared L2 feature norms
        loss_l1 = f.abs().sum(dim=-1).mean()
        loss_l2 = f.pow(2).sum(dim=-1).mean()

        loss_saenc = loss_recon + self.lambda_l1 * loss_l1 + self.lambda_l2 * loss_l2

        return f, x_hat, {
            "loss_saenc": loss_saenc,
            "loss_recon": loss_recon,
            "loss_l1": loss_l1,
        }

    def normalize_decoder(self) -> None:
        """Normalize decoder weight columns to unit L2 norm.

        Should be called after each optimizer step. Prevents the decoder
        from absorbing feature scale into its column norms, keeping feature
        activations and decoder directions cleanly separated.
        """
        with torch.no_grad():
            col_norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(col_norms)
