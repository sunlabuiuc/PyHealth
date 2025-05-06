from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.models import BaseModel  # Adjust this import path as needed


features = 32


class CFVAE(BaseModel):
    """
    Counterfactual Variational Autoencoder (CFVAE) model for generating 
    counterfactual explanations in healthcare datasets.

    This model combines a Variational Autoencoder (VAE) branch for reconstructing
    input features and a Multilayer Perceptron (MLP) classification branch to
    predict outcomes. It learns a latent representation of patient data and uses
    it to generate both predictions and feature reconstructions, enabling
    counterfactual analysis.

    Args:
        dataset: A PyHealth dataset instance that includes patient records.
        feat_dim (int): Dimensionality of the input feature vector.
        emb_dim1 (int): Dimensionality of the intermediate VAE embedding layer.
        _mlp_dim1 (int): (Unused) First MLP dimension (kept for compatibility).
        _mlp_dim2 (int): (Unused) Second MLP dimension (kept for compatibility).
        _mlp_dim3 (int): (Unused) Third MLP dimension (kept for compatibility).
        mlp_inpemb (int): Dimensionality of the input embedding to the MLP.
        f_dim1 (int): First hidden layer size in MLP.
        f_dim2 (int): Second hidden layer size in MLP.

    Attributes:
        enc1, enc2: Linear layers forming the encoder of the VAE.
        dec1, dec2: Linear layers forming the decoder of the VAE.
        word_embeddings: Initial embedding layer for the MLP.
        fc1, fc2: Fully connected layers in the MLP.
        ln1, ln2: Layer normalization layers for stabilizing training.
        scorelayer: Projects final MLP output to a scalar.
        pred: Final prediction layer using the output size from the base model.

    """
    def __init__(
        self,
        dataset,
        feat_dim: int,
        emb_dim1: int,
        _mlp_dim1: int,
        _mlp_dim2: int,
        _mlp_dim3: int,
        mlp_inpemb: int,
        f_dim1: int,
        f_dim2: int,
    ):
        super(CFVAE, self).__init__(dataset=dataset)

        # VAE branch
        self.enc1 = nn.Linear(in_features=feat_dim, out_features=emb_dim1)
        self.enc2 = nn.Linear(in_features=emb_dim1, out_features=features * 2)

        self.dec1 = nn.Linear(in_features=features, out_features=emb_dim1)
        self.dec2 = nn.Linear(in_features=emb_dim1, out_features=feat_dim)

        # MLP branch
        self.word_embeddings = nn.Linear(feat_dim, mlp_inpemb)
        self.ln1 = nn.LayerNorm(mlp_inpemb)

        self.fc1 = nn.Linear(mlp_inpemb, f_dim1)
        self.ln2 = nn.LayerNorm(f_dim1)

        self.fc2 = nn.Linear(f_dim1, f_dim2)

        self.scorelayer = nn.Linear(f_dim2, 1)
        self.pred = nn.Linear(1, self.get_output_size())  # From BaseModel

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick to sample from a Gaussian distribution
        with mean `mu` and log-variance `log_var`.

        This is commonly used in variational autoencoders (VAEs) to allow gradients
        to propagate through stochastic nodes during training.

        Args:
            mu (torch.Tensor): The mean of the latent Gaussian distribution.
            log_var (torch.Tensor): The log-variance of the latent Gaussian distribution.

        Returns:
            torch.Tensor: A sampled tensor from the Gaussian distribution using
                        the reparameterization trick: z = mu + std * eps,
                        where eps ~ N(0, I) and std = exp(0.5 * log_var).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Args:
            samples (List[Dict]): A batch of samples from the dataset

        Returns:
            Dict[str, torch.Tensor]: Includes 'logits' and optionally 'reconstruction', 'mu', 'log_var'
        """
        # Combine input tensors from feature keys
        x = torch.stack([
            torch.cat([sample[k] for k in self.feature_keys], dim=-1)
            for sample in samples
        ])

        # VAE
        enc = F.relu(self.enc1(x))
        enc = self.enc2(enc).view(-1, 2, features)

        mu = enc[:, 0, :]
        log_var = enc[:, 1, :]
        z = self.reparameterize(mu, log_var)

        dec = F.relu(self.dec1(z))
        reconstruction = self.dec2(dec)

        # MLP
        embeds = self.word_embeddings(reconstruction)
        embeds = self.ln1(embeds)

        out1 = F.relu(self.fc1(embeds))
        out1 = self.ln2(out1)
        out2 = F.relu(self.fc2(out1))

        out3 = self.scorelayer(out2)
        logits = self.pred(out3)

        return {
            "logits": logits,
            "reconstruction": reconstruction,
            "mu": mu,
            "log_var": log_var
        }