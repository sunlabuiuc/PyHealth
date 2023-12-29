import functools
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import BaseSignalDataset
from pyhealth.models import BaseModel


class VAE(BaseModel):
    """VAE model

    Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes."

    Note:
        We use two-layer DNN models as the encoder and decoder layers for now.

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
        input_flatten_size: int,
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
        self.encoder1 = nn.Linear(input_flatten_size, self.hidden_dim)
        self.mu = nn.Linear(self.hidden_dim, self.hidden_dim) # for mu 
        self.log_std2 = nn.Linear(self.hidden_dim, self.hidden_dim) # for log (sigma^2)
        
        # decoder part
        self.decoder1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.decoder2 = nn.Linear(self.hidden_dim, input_flatten_size)

    def encoder(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.encoder1(x))
        return self.mu(h), torch.sqrt(torch.exp(self.log_std2(h)))
    
    def sampling(self, mu, std) -> torch.Tensor: # reparameterization trick
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z) -> torch.Tensor:
        h = torch.tanh(self.decoder1(z))
        return torch.sigmoid(self.decoder2(h)) 
    
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
        
        batch_size = x.size(0)
        x_flatten = x.view(batch_size, -1)
        mu, std = self.encoder(x_flatten)
        z = self.sampling(mu, std)
        x_rec = self.decoder(z)
        
        loss = self.loss_function(x_rec, x_flatten, mu, std)
        results = {
            "loss": loss,
            "y_prob": x_rec,
            "y_true": x_flatten,
        }
        return results


if __name__ == "__main__":
    from pyhealth.datasets import SampleSignalDataset, get_dataloader
    from pyhealth.datasets import SleepEDFDataset
    from pyhealth.tasks import sleep_staging_sleepedf_fn

    dataset = SleepEDFDataset(
        root="/srv/local/data/SLEEPEDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        dev=True,
        refresh_cache=False,
    )
    
    dataset = dataset.set_task(sleep_staging_sleepedf_fn)
    # print (pickle.load(open(dataset.samples[0]["epoch_path"], "rb"))["signal"].shape) # (7, 3000)

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = VAE(
        dataset=dataset,
        input_flatten_size=7*3000,
        feature_keys=["signal"],
        label_key="signal",
        mode="regression",
    ).to("cuda:0")

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()