import functools
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import BaseSignalDataset
from pyhealth.models import BaseModel, ResBlock2D


class VAE(BaseModel):
    """VAE model (take 128x128 or 64x64 or 32x32 images)

    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."

    Note:
        We use CNN models as the encoder and decoder layers for now.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        feature_keys:  list of keys in samples to use as features,
            e.g. ["conditions", "procedures"].
        label_key: key in samples to use as label (e.g., "drugs").
        mode: one of "binary", "multiclass", or "multilabel".
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        **kwargs: other parameters for the Deepr layer.

    Examples:
    """

    def __init__(
        self,
        dataset: BaseSignalDataset,
        feature_keys: List[str],
        label_key: str,
        input_channel: int,
        input_size: int,
        mode: str,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(VAE, self).__init__(
            dataset=dataset,
            feature_keys=feature_keys,
            label_key=label_key,
            mode=mode,
        )
        self.hidden_dim = hidden_dim

        # encoder part
        if input_size == 128:
            self.encoder1 = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
            )
            self.mu = nn.Linear(256 * 2 * 2, self.hidden_dim) # for mu 
            self.log_std2 = nn.Linear(256 * 2 * 2, self.hidden_dim) # for log (sigma^2)
            
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
            self.mu = nn.Linear(256, self.hidden_dim) # for mu 
            self.log_std2 = nn.Linear(256, self.hidden_dim) # for log (sigma^2)
            
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
            self.mu = nn.Linear(64 * 2 * 2, self.hidden_dim) # for mu 
            self.log_std2 = nn.Linear(64 * 2 * 2, self.hidden_dim) # for log (sigma^2)
            
            self.decoder1 = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
            
    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder1(x)
        batch_size = h.shape[0]
        h = h.view(batch_size, -1)
        return self.mu(h), torch.sqrt(torch.exp(self.log_std2(h)))
    
    def sampling(self, mu, std) -> torch.Tensor: # reparameterization trick
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z) -> torch.Tensor:
        x_hat = self.decoder1(z)
        return x_hat
    
    @staticmethod
    def loss_function(y, x, mu, std): 
        ERR = F.binary_cross_entropy(y, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2)
        return ERR + KLD
    
    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        
        # concat the info within one batch (batch, channel, height, width)
        # if the input is a list of numpy array, we need to convert it to tensor
        if isinstance(kwargs[self.feature_keys[0]][0], np.ndarray):
            x = torch.tensor(
                np.array(kwargs[self.feature_keys[0]]).astype("float16"), device=self.device
            ).float()
        else:
            x = torch.stack(kwargs[self.feature_keys[0]], dim=0).to(self.device)
    
        mu, std = self.encoder(x)
        z = self.sampling(mu, std)
        z = z.unsqueeze(2).unsqueeze(3)
        x_rec = self.decoder(z)
        
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
        input_channel=3,
        input_size=128,
        feature_keys=["path"],
        label_key="path",
        mode="regression",
        hidden_dim = 256,
    ).to("cuda")

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()