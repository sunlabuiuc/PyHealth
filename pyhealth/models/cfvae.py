from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.models import BaseModel  # Adjust this import path as needed


features = 32


class CFVAE(BaseModel):
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