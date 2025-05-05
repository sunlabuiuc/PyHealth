# ==============================================================================
# Author(s): Sharim Khan, Gabriel Lee
# NetID(s): sharimk2, gjlee4
# Paper title:
#           Explaining A Machine Learning Decision to Physicians via Counterfactuals
# Paper link: https://arxiv.org/abs/2306.06325
# Description: This file defines the Counterfactual Variational Autoencoder (CFVAE)
#              model, which reconstructs input data while generating counterfactual
#              examples that flip the prediction of a frozen classifier.
# ==============================================================================

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


class CFVAE(BaseModel):
    """Counterfactual Variational Autoencoder (CFVAE) for binary prediction tasks.

    This is a parametrized version of the CFVAE model described by Nagesh et al.

    The CFVAE learns to reconstruct inputs while generating counterfactual samples
    that flip the output of a fixed, externally trained binary classifier. It combines
    VAE reconstruction and KL divergence losses with a classifier-based loss.

    NOTE: A binary classifier MUST be passed as an argument.
    NOTE: The sparsity constraint should be implemented in the training loop.

    Attributes:
        feature_keys: Feature keys used as inputs.
        label_keys: A list containing the label key.
        mode: Task mode (must be 'binary').
        latent_dim: Latent dimensionality of the VAE.
        external_classifier: Frozen external classifier for guiding counterfactuals.
        enc1: First encoder layer.
        enc2: Layer projecting to latent mean and log-variance.
        dec1: First decoder layer.
        dec2: Layer projecting to reconstructed input space.

    Example:
        cfvae = CFVAE(
            dataset=samples,
            feature_keys=["labs"],
            label_key="mortality",
            mode="binary",
            feat_dim=27,
            latent_dim=32,
            hidden_dim=64,
            external_classifier=frozen_classifier
        )
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        feat_dim: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        external_classifier: nn.Module = None,
    ):
        """
        Initializes the CFVAE model and freezes the external classifier.

        Args:
            dataset: PyHealth-compatible dataset object.
            feature_keys: List of input feature keys.
            label_key: Output label key (must be binary).
            mode: Task mode ('binary' only supported).
            feat_dim: Input feature dimensionality.
            latent_dim: Latent space dimensionality.
            hidden_dim: Hidden layer size in encoder/decoder.
            external_classifier: Frozen binary classifier to guide counterfactuals.
        """
        super().__init__(dataset)
        self.feature_keys = feature_keys
        self.label_keys = [label_key]
        self.mode = mode

        assert mode == "binary", "Only binary classification is supported."
        assert external_classifier is not None, "external_classifier must be provided."

        self.latent_dim = latent_dim
        self.external_classifier = external_classifier.eval()
        for param in self.external_classifier.parameters():
            param.requires_grad = False

        self.enc1 = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.enc2 = nn.Linear(hidden_dim, 2 * latent_dim)

        self.dec1 = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.dec2 = nn.Linear(hidden_dim, feat_dim)

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample z from Gaussian N.

        Args:
            mu: Mean of the latent distribution, shape (B, latent_dim).
            log_var: Log variance of the latent distribution, shape (B, latent_dim).

        Returns:
            z: Sampled latent variable, shape (B, latent_dim).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass for CFVAE: encodes input, reparameterizes, decodes with flipped
        labels, and computes reconstruction, KL, and classifier-based losses.

        Args:
            kwargs: Dict of inputs including:
                - feature_keys[0]: Input tensor (B, feat_dim)
                - label_keys[0]: Ground truth label tensor (B,)

        Returns:
            Dictionary containing:
                - loss: Total training loss (recon + KL + classifier disagreement).
                - y_prob: Classifier output probabilities for reconstructed inputs.
                - y_true: Ground truth labels.
        """
        x = kwargs[self.feature_keys[0]].to(self.device)
        y = kwargs[self.label_keys[0]].to(self.device)

        # Encode inputs
        h = self.enc1(x)
        h = self.enc2(h).view(-1, 2, self.latent_dim)
        mu, log_var = h[:, 0, :], h[:, 1, :]
        z = self.reparameterize(mu, log_var)

        # Flip labels to condition decoder on opposite class (counterfactual)
        y_cf = 1 - y
        y_cf_onehot = F.one_hot(y_cf.view(-1).long(), num_classes=2).float()
        z_cond = torch.cat([z, y_cf_onehot], dim=1)

        h_dec = self.dec1(z_cond)
        x_recon = torch.sigmoid(self.dec2(h_dec))

        # Evaluate external classifier on counterfactual
        with torch.no_grad():
            logits = self.external_classifier(x_recon)

        # Compute losses
        clf_loss = self.get_loss_function()(logits, y)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kld_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )
        total_loss = recon_loss + kld_loss + clf_loss

        return {
            "loss": total_loss,
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y,
        }

