import functools
from typing import Dict, List, Optional, Tuple
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import BaseSignalDataset
from pyhealth.models import BaseModel, ResBlock2D

class Flatten(nn.Module):
    def forward(self, input):
        '''
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        '''
        batch_size = input.size(0)
        out = input.view(batch_size,-1)
        return out # (batch_size, *size)
    
class GAN(nn.Module):
    """GAN model (take 128x128 or 64x64 or 32x32 images)

    Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,
    Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets.

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
        input_channel: int,
        input_size: int,
        hidden_dim: int = 128,
        **kwargs,
    ):
        super(GAN, self).__init__()
        self.hidden_dim = hidden_dim

        # encoder part
        if input_size == 128:
            self.discriminator = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
                Flatten(),
                nn.Linear(256 * 2 * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )
            
            self.generator = nn.Sequential(
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
            self.discriminator = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
                Flatten(),
                nn.Linear(256, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )

            self.generator = nn.Sequential(
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
            self.discriminator = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                Flatten(),
                nn.Linear(64 * 2 * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
                nn.Sigmoid(),
            )
            
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
            
    def discriminate(self, x) -> torch.Tensor:
        y = self.discriminator(x)
        return y
    
    def sampling(self, n_samples, device) -> torch.Tensor:
        eps = torch.randn(n_samples, self.hidden_dim, 1, 1).to(device)
        return eps

    def generate_fake(self, n_samples, device) -> torch.Tensor:
        eps = self.sampling(n_samples, device)
        fake_images = self.generator(eps)
        return fake_images
    

if __name__ == "__main__":
    
    """ simple test """
    model = GAN(
        input_channel=3,
        input_size=128,
        hidden_dim = 256,
    )
    
    # test generation 
    n_samples = 10
    device = "cpu"
    fake_images = model.generate_fake(n_samples, device)
    print (fake_images.shape)
    
    # test discriminate
    y = model.discriminate(fake_images)
    print (y.shape)
    
    
    