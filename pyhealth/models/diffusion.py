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
    
class Diffuser(nn.Module):
    """Diffuser model (take 128x128 or 64x64 or 32x32 images)

    Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." 
    Advances in neural information processing systems 33 (2020): 6840-6851.

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
        diff_steps: the diffusion steps, e.g., 25.
        n_labels: the label space of the dataset.
        sigma: the variance of the generated image.
        **kwargs: other parameters for the Deepr layer.

    Examples:
    """

    def __init__(
        self,
        input_channel: int,
        input_size: int,
        hidden_dim: int = 128,
        diff_steps: int = 25,
        n_labels: int = 2,
        sigma: float = 0.35,
        **kwargs,
    ):
        super(Diffuser, self).__init__()
        self.hidden_dim = hidden_dim
        self.diff_steps = diff_steps
        self.n_labels = n_labels
        self.sigma = sigma

        # encoder part
        if input_size == 128:
            self.encoder = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
                Flatten(),
                nn.Linear(256 * 2 * 2, self.hidden_dim),
                nn.ReLU(),
            )
            self.infuser = nn.Sequential(
                nn.Linear(self.hidden_dim + diff_steps + n_labels, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
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
            self.encoder = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                ResBlock2D(64, 256, 2, True, True),
                Flatten(),
                nn.Linear(256, self.hidden_dim),
                nn.ReLU(),
            )

            self.infuser = nn.Sequential(
                nn.Linear(self.hidden_dim + diff_steps + n_labels, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
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
            self.encoder = nn.Sequential(
                ResBlock2D(input_channel, 16, 2, True, True),
                ResBlock2D(16, 64, 2, True, True),
                Flatten(),
                nn.Linear(64 * 2 * 2, self.hidden_dim),
                nn.ReLU(),
            )
            
            self.infuser = nn.Sequential(
                nn.Linear(self.hidden_dim + diff_steps + n_labels, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            
            self.generator = nn.Sequential(
                nn.ConvTranspose2d(self.hidden_dim, 64, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(32, input_channel, kernel_size=6, stride=2),
                nn.Sigmoid(),
            )
            
    def forward(self, x, step, labels) -> torch.Tensor:
        # one hot encoding of step
        step_encoding = F.one_hot(
            torch.tensor([step] * x.shape[0]), num_classes = self.diff_steps
        ).to(x.device)
        
        # one hot encoding of labels
        labels_encoding = F.one_hot(labels, num_classes = self.n_labels).to(x.device)
        
        # image encodinig
        image_encoding = self.encoder(x)
        
        # concatenate three and infuse the information
        infused_info = torch.cat([image_encoding, step_encoding, labels_encoding], 1)
        infused_info = self.infuser(infused_info).unsqueeze(2).unsqueeze(3)
        
        # generate denoised image
        out = self.generator(infused_info)
        
        # normalize the image pixel into N(0, sigma)
        MEAN = out.mean((0,1), keepdim=True)
        STD = out.std((0,1), keepdim=True) + 1e-8
        out = (out - MEAN) / STD * self.sigma
        return out
    

if __name__ == "__main__":
    
    # we have 25 diffusion steps
    diff_steps = 25
    n_labels = 2
    
    """ simple test """
    model = Diffuser(
        input_channel=3,
        input_size=128,
        hidden_dim = 256,
        diff_steps = diff_steps,
        n_labels = n_labels,
    )

    # random images
    images = torch.randn((1, 3, 128, 128))
    
    # mimic the difussion process at step=24, label=1
    label = torch.LongTensor([1])
    step = 24
    denoised_image = model(images, step, label)
    print (denoised_image.shape)
    
    """
    torch.Size([1, 3, 128, 128])
    """
    
    