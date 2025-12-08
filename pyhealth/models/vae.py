import functools
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import BaseSignalDataset
from pyhealth.models import BaseModel, ResBlock2D, EmbeddingModel


class VAE(BaseModel):
    """VAE model for images or time-series data.

    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."

    Supports both image generation/reconstruction and time-series modeling.
    Images mode take 128x128 or 64x64 or 32x32 images.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        input_type: 'image' for CNN-based VAE on images, 'timeseries' for RNN-based on sequences. Default 'image'.
        input_channel: number of input channels (for images). Required if input_type='image'.
        input_size: input image size (for images, e.g. 128). Required if input_type='image'.
        hidden_dim: the latent dimension. Default is 128.
        conditional_feature_keys: list of feature keys to use as conditions for generation (optional).
        **kwargs: other parameters.
    """

    def __init__(
        self,
        dataset,
        feature_keys: List[str],
        label_key: str,
        mode: str,
        input_type: str = "image",
        input_channel: Optional[int] = None,
        input_size: Optional[int] = None,
        hidden_dim: int = 128,
        conditional_feature_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        super(VAE, self).__init__(dataset=dataset)
        self.input_type = input_type
        self.hidden_dim = hidden_dim
        self.conditional_feature_keys = conditional_feature_keys
        self.mode = mode
        self.feature_keys = feature_keys
        self.label_key = label_key

        # These will be lazily initialized when we see actual tensor sizes
        self.cond_proj: Optional[nn.Linear] = None   # for conditional metadata → latent
        self.ts_proj: Optional[nn.Linear] = None     # for timeseries concatenated features → hidden_dim

        if input_type == "image":
            assert input_channel is not None and input_size is not None, \
                "For image mode, input_channel and input_size must be provided"

            # Embedding model for conditional features only (if used)
            if conditional_feature_keys:
                self.embedding_model = EmbeddingModel(dataset, embedding_dim=hidden_dim)

            # ----- Encoder -----
            if input_size == 128:
                self.encoder1 = nn.Sequential(
                    ResBlock2D(input_channel, 16, 2, True, True),
                    ResBlock2D(16, 64, 2, True, True),
                    ResBlock2D(64, 256, 2, True, True),
                )
                self.mu = nn.Linear(256 * 2 * 2, self.hidden_dim)
                self.log_std2 = nn.Linear(256 * 2 * 2, self.hidden_dim)

                self.decoder1 = nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dim, 256, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                    nn.Sigmoid(),
                )

            elif input_size == 64:
                self.encoder1 = nn.Sequential(
                    ResBlock2D(input_channel, 16, 2, True, True),
                    ResBlock2D(16, 64, 2, True, True),
                    ResBlock2D(64, 256, 2, True, True),
                )
                self.mu = nn.Linear(256, self.hidden_dim)
                self.log_std2 = nn.Linear(256, self.hidden_dim)

                self.decoder1 = nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dim, 128, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                    nn.Sigmoid(),
                )

            elif input_size == 32:
                self.encoder1 = nn.Sequential(
                    ResBlock2D(input_channel, 16, 2, True, True),
                    ResBlock2D(16, 64, 2, True, True),
                    # ResBlock2D(64, 256, 2, True, True),
                )
                self.mu = nn.Linear(64 * 2 * 2, self.hidden_dim)
                self.log_std2 = nn.Linear(64 * 2 * 2, self.hidden_dim)

                self.decoder1 = nn.Sequential(
                    nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                    nn.ReLU(),
                    nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                    nn.Sigmoid(),
                )
            else:
                raise ValueError("Unsupported input_size for image mode")

        elif input_type == "timeseries":
            # Embedding model for sequence features
            self.embedding_model = EmbeddingModel(dataset, embedding_dim=hidden_dim)
            self.encoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.mu = nn.Linear(hidden_dim, hidden_dim)
            self.log_std2 = nn.Linear(hidden_dim, hidden_dim)
            self.decoder_linear = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError("input_type must be 'image' or 'timeseries'")

    # -------------------------------------------------------------
    # ENCODER
    # -------------------------------------------------------------
    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.input_type == "image":
            h = self.encoder1(x)
            batch_size = h.shape[0]
            h = h.view(batch_size, -1)
            mu = self.mu(h)
            std = torch.sqrt(torch.exp(self.log_std2(h)))

        elif self.input_type == "timeseries":
            # x is dict of embedded features from embedding_model
            embedded_list = []
            for key, emb in x.items():
                if emb.dim() == 3:  # (batch, seq, emb)
                    _, h_seq = self.encoder_rnn(emb)   # h_seq: (1, batch, hidden_dim)
                    h = h_seq.squeeze(0)               # (batch, hidden_dim)
                else:
                    h = emb  # (batch, emb_dim)
                embedded_list.append(h)

            h = torch.cat(embedded_list, dim=-1) if len(embedded_list) > 1 else embedded_list[0]

            # Project to hidden_dim once, with a learnable layer
            if h.shape[-1] != self.hidden_dim:
                if self.ts_proj is None:
                    self.ts_proj = nn.Linear(h.shape[-1], self.hidden_dim).to(h.device)
                h = self.ts_proj(h)

            mu = self.mu(h)
            std = torch.sqrt(torch.exp(self.log_std2(h)))

        return mu, std

    # -------------------------------------------------------------
    # SAMPLING
    # -------------------------------------------------------------
    def sampling(self, mu, std) -> torch.Tensor:
        eps = torch.randn_like(std)
        return mu + eps * std

    # -------------------------------------------------------------
    # DECODER
    # -------------------------------------------------------------
    def decoder(self, z) -> torch.Tensor:
        if self.input_type == "image":
            x_hat = self.decoder1(z)
        elif self.input_type == "timeseries":
            x_hat = self.decoder_linear(z)  # (batch, hidden_dim)
        return x_hat

    # -------------------------------------------------------------
    # LOSS
    # -------------------------------------------------------------
    def loss_function(self, y, x, mu, std):
        if self.input_type == "image":
            ERR = F.binary_cross_entropy(y, x, reduction='sum')
        elif self.input_type == "timeseries":
            ERR = F.mse_loss(y, x, reduction='sum')

        # KL divergence term
        KLD = -0.5 * torch.sum(1 + torch.log(std ** 2) - mu ** 2 - std ** 2)
        return ERR + KLD

    # -------------------------------------------------------------
    # FORWARD
    # -------------------------------------------------------------
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:

        if self.input_type == "image":
            # x: image tensor
            x = kwargs[self.feature_keys[0]].to(self.device)  # (batch, C, H, W)

            mu, std = self.encoder(x)
            z = self.sampling(mu, std)  # (batch, hidden_dim)

            # Conditional embeddings (if any)
            if self.conditional_feature_keys:
                cond_raw = {k: kwargs[k] for k in self.conditional_feature_keys}
                cond_emb = self.embedding_model(cond_raw)  # dict: key -> tensor

                # Pool temporal dims if needed and concat
                cond_list = [
                    emb.mean(dim=1) if emb.dim() == 3 else emb
                    for emb in cond_emb.values()
                ]
                cond_vec = torch.cat(cond_list, dim=-1) if len(cond_list) > 1 else cond_list[0]

                # Project cond_vec to latent dim with a learnable layer
                if cond_vec.shape[-1] != self.hidden_dim:
                    if self.cond_proj is None:
                        self.cond_proj = nn.Linear(cond_vec.shape[-1], self.hidden_dim).to(cond_vec.device)
                    cond_vec = self.cond_proj(cond_vec)

                # Simple conditioning: shift latent by conditional vector
                z = z + cond_vec

            # Prepare for ConvTranspose decoder
            z = z.unsqueeze(2).unsqueeze(3)  # (batch, hidden_dim, 1, 1)
            x_rec = self.decoder(z)

        elif self.input_type == "timeseries":
            # Embed all feature_keys first
            embedded = self.embedding_model({k: kwargs[k] for k in self.feature_keys})
            mu, std = self.encoder(embedded)
            z = self.sampling(mu, std)
            x_rec = self.decoder(z)

            # For reconstruction target x: re-aggregate exactly as in encoder
            embedded_list = []
            for key, emb in embedded.items():
                if emb.dim() == 3:  # (batch, seq, emb)
                    _, h_seq = self.encoder_rnn(emb)
                    h = h_seq.squeeze(0)  # (batch, hidden_dim)
                else:
                    h = emb
                embedded_list.append(h)

            x = torch.cat(embedded_list, dim=-1) if len(embedded_list) > 1 else embedded_list[0]
            if x.shape[-1] != self.hidden_dim:
                if self.ts_proj is None:
                    self.ts_proj = nn.Linear(x.shape[-1], self.hidden_dim).to(x.device)
                x = self.ts_proj(x)

        loss = self.loss_function(x_rec, x, mu, std)
        results = {
            "loss": loss,
            "y_prob": x_rec,
            "y_true": x,
        }
        return results



if __name__ == "__main__":
    from pyhealth.datasets import SampleSignalDataset, get_dataloader
    from pyhealth.datasets import COVID19CXRDataset
    from torchvision import transforms
    
    root = "/srv/local/data/COVID-19_Radiography_Dataset"
    base_dataset = COVID19CXRDataset(root, dev=True, refresh_cache=False)
    
    sample_dataset = base_dataset.set_task()

    # the transformation automatically normalize the pixel intensity into [0, 1]
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x if x.shape[0] == 3 else x.repeat(3, 1, 1)), # only use the first channel
        transforms.Resize((128, 128)),
    ])

    def encode(sample):
        sample["path"] = transform(sample["path"])
        return sample

    sample_dataset.set_transform(encode)

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(sample_dataset, batch_size=2, shuffle=True)

    # model
    model = VAE(
        dataset=sample_dataset,
        feature_keys=["path"],
        label_key="path",
        mode="regression",
        input_type="image",
        input_channel=3,
        input_size=128,
        hidden_dim=256,
    ).to("cuda")

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()