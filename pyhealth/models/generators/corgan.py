import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pyhealth.models import BaseModel


class CorGANDataset(Dataset):
    """Dataset wrapper for CorGAN training"""

    def __init__(self, data, transform=None):
        self.transform = transform
        self.data = data.astype(np.float32)
        self.sampleSize = data.shape[0]
        self.featureSize = data.shape[1]

    def return_data(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]
        sample = np.clip(sample, 0, 1)

        if self.transform:
            pass

        return torch.from_numpy(sample)


class CorGANAutoencoder(nn.Module):
    """Autoencoder for CorGAN - uses 1D convolutions to capture correlations"""

    def __init__(self, feature_size: int, latent_dim: int = 128, use_adaptive_pooling: bool = False):
        super(CorGANAutoencoder, self).__init__()
        self.feature_size = feature_size
        self.latent_dim = latent_dim
        self.use_adaptive_pooling = use_adaptive_pooling
        n_channels_base = 4

        # calculate the size after convolutions
        # input: (batch, 1, feature_size)
        # conv1: kernel=5, stride=2 -> (batch, 4, (feature_size-4)//2)
        # conv2: kernel=5, stride=2 -> (batch, 8, ((feature_size-4)//2-4)//2)
        # conv3: kernel=5, stride=3 -> (batch, 16, (((feature_size-4)//2-4)//2-4)//3)
        # conv4: kernel=5, stride=3 -> (batch, 32, ((((feature_size-4)//2-4)//2-4)//3-4)//3)
        # conv5: kernel=5, stride=3 -> (batch, 64, (((((feature_size-4)//2-4)//2-4)//3-4)//3-4)//3)
        # conv6: kernel=8, stride=1 -> (batch, 128, ((((((feature_size-4)//2-4)//2-4)//3-4)//3-4)//3-7))

        # rough estimate for latent size
        latent_size = max(1, feature_size // 100)  # ensure at least 1

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_channels_base, kernel_size=5, stride=2, padding=0, dilation=1,
                      groups=1, bias=True, padding_mode='zeros'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=n_channels_base, out_channels=2 * n_channels_base, kernel_size=5, stride=2, padding=0,
                      dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=2 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=4 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=8 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=3,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(16 * n_channels_base),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=16 * n_channels_base, out_channels=32 * n_channels_base, kernel_size=8, stride=1,
                      padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.Tanh(),
        )

        # decoder - exact match to synthEHRella (wgancnnmimic.py lines 200-228)
        # Kernel sizes: [5, 5, 7, 7, 7, 3]
        # Strides: [1, 4, 4, 3, 2, 2]
        # Activations: ReLU (not LeakyReLU)
        # Note: First layer has NO BatchNorm
        decoder_layers = [
            nn.ConvTranspose1d(in_channels=32 * n_channels_base, out_channels=16 * n_channels_base, kernel_size=5, stride=1,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16 * n_channels_base, out_channels=8 * n_channels_base, kernel_size=5, stride=4,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(8 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=8 * n_channels_base, out_channels=4 * n_channels_base, kernel_size=7, stride=4,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(4 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=4 * n_channels_base, out_channels=2 * n_channels_base, kernel_size=7, stride=3,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(2 * n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2 * n_channels_base, out_channels=n_channels_base, kernel_size=7, stride=2,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
            nn.BatchNorm1d(n_channels_base),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=n_channels_base, out_channels=1, kernel_size=3, stride=2,
                               padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'),
        ]

        # Add adaptive pooling if enabled (for variable vocabulary sizes)
        if self.use_adaptive_pooling:
            decoder_layers.append(nn.AdaptiveAvgPool1d(output_size=feature_size))

        decoder_layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        # add channel dimension if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Squeeze only the channel dimension (dim=1), not the batch dimension
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded

    def decode(self, x):
        # x shape: (batch, 128) from generator - unsqueeze for CNN decoder
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (batch, 128, 1)
        decoded = self.decoder(x)  # (batch, 1, output_len)
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)  # (batch, output_len)
        return decoded


class CorGAN8LayerAutoencoder(nn.Module):
    """
    8-Layer CNN Autoencoder for CorGAN - designed for 6,955 codes.

    Extends the original 6-layer architecture to support larger vocabularies
    without adaptive pooling. The encoder compresses 6,955 codes down to a
    latent space of size (128, 1), then the decoder reconstructs exactly 6,955.

    This is an experimental architecture designed to test whether native
    dimension matching (no adaptive pooling) produces better synthetic data
    quality compared to the 6-layer + adaptive pooling approach.

    Args:
        feature_size: Must be 6955 (architecture is hardcoded for this size)
        latent_dim: Latent dimension (default: 128)
    """

    def __init__(self, feature_size: int = 6955, latent_dim: int = 128):
        super(CorGAN8LayerAutoencoder, self).__init__()
        assert feature_size == 6955, "8-layer architecture only supports 6955 codes"

        self.feature_size = feature_size
        self.latent_dim = latent_dim

        # Encoder: 6955 -> 1 (8 layers)
        self.encoder = nn.Sequential(
            # Layer 1: 6955 -> 3476
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2: 3476 -> 1736
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3: 1736 -> 578
            nn.Conv1d(8, 16, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 4: 578 -> 192
            nn.Conv1d(16, 32, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 5: 192 -> 63
            nn.Conv1d(32, 64, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 6: 63 -> 20 [NEW]
            nn.Conv1d(64, 96, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(96),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 7: 20 -> 4 [NEW]
            nn.Conv1d(96, 112, kernel_size=5, stride=4, padding=0),
            nn.BatchNorm1d(112),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 8: 4 -> 1 [NEW]
            nn.Conv1d(112, 128, kernel_size=4, stride=1, padding=0),
            nn.Tanh(),
        )

        # Decoder: 1 -> 6955 (8 layers)
        self.decoder = nn.Sequential(
            # Layer 1: 1 -> 4 (NO BatchNorm on first layer)
            nn.ConvTranspose1d(128, 112, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),

            # Layer 2: 4 -> 20
            nn.ConvTranspose1d(112, 96, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm1d(96),
            nn.ReLU(),

            # Layer 3: 20 -> 63
            nn.ConvTranspose1d(96, 64, kernel_size=6, stride=3, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            # Layer 4: 63 -> 192
            nn.ConvTranspose1d(64, 32, kernel_size=6, stride=3, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            # Layer 5: 192 -> 578
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # Layer 6: 578 -> 1736
            nn.ConvTranspose1d(16, 8, kernel_size=5, stride=3, padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(),

            # Layer 7: 1736 -> 3476
            nn.ConvTranspose1d(8, 4, kernel_size=6, stride=2, padding=0),
            nn.BatchNorm1d(4),
            nn.ReLU(),

            # Layer 8: 3476 -> 6955
            nn.ConvTranspose1d(4, 1, kernel_size=5, stride=2, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # (batch, 1, features)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        # Squeeze only channel dimension
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded

    def decode(self, x):
        """Decode latent representation from generator."""
        if x.dim() == 2:
            x = x.unsqueeze(2)  # (batch, 128, 1)
        decoded = self.decoder(x)
        if decoded.dim() == 3 and decoded.shape[1] == 1:
            decoded = decoded.squeeze(1)
        return decoded


class CorGANLinearAutoencoder(nn.Module):
    """
    Linear autoencoder for CorGAN - simpler than CNN, appropriate for unordered codes.

    This variant replaces the CNN autoencoder with a simple linear architecture,
    which is more appropriate for unordered medical codes (ICD-9) where spatial
    locality doesn't exist. Based on:
    - SynthEHRella's commented linear decoder alternative (line 229 in wgancnnmimic.py)
    - MedGAN's proven linear architecture (achieves 10.66 codes/patient)
    - Simpler gradient flow to address mode collapse

    The core CorGAN components are preserved:
    - WGAN training with Wasserstein loss
    - Generator with residual connections
    - Discriminator with minibatch averaging

    This architecture is referred to as "CorGAN-Linear" to distinguish it from
    the original CNN-based CorGAN while maintaining the core WGAN design.
    """

    def __init__(self, feature_size: int, latent_dim: int = 128):
        super(CorGANLinearAutoencoder, self).__init__()
        self.feature_size = feature_size
        self.latent_dim = latent_dim

        # Encoder: feature_size -> latent_dim
        # Use ReLU+BatchNorm (V11 achieved 4.49 codes, best linear result)
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )

        # Decoder: latent_dim -> feature_size
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, feature_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass for autoencoder training.

        Args:
            x: Input tensor of shape (batch, feature_size)

        Returns:
            Decoded tensor of shape (batch, feature_size)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def decode(self, x):
        """
        Decode latent representation from generator.

        Args:
            x: Latent tensor from generator of shape (batch, latent_dim)

        Returns:
            Decoded tensor of shape (batch, feature_size)
        """
        return self.decoder(x)


class CorGANGenerator(nn.Module):
    """
    Generator for CorGAN - MLP with residual connections

    Architecture matches synthEHRella exactly (wgancnnmimic.py lines 242-263)
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128):
        super(CorGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Layer 1
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation1 = nn.ReLU()

        # Layer 2
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim, eps=0.001, momentum=0.01)
        self.activation2 = nn.Tanh()

    def forward(self, x):
        # Layer 1 with residual connection
        residual = x
        temp = self.activation1(self.bn1(self.linear1(x)))
        out1 = temp + residual

        # Layer 2 with residual connection
        residual = out1
        temp = self.activation2(self.bn2(self.linear2(out1)))
        out2 = temp + residual

        return out2


class CorGANDiscriminator(nn.Module):
    """
    Discriminator for CorGAN - MLP with minibatch averaging

    Architecture matches synthEHRella exactly (wgancnnmimic.py lines 265-296):
    - 4 linear layers: input -> 256 -> 256 -> 256 -> 1
    - ReLU activations
    - No sigmoid (WGAN uses unbounded critic outputs)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, minibatch_averaging: bool = True):
        super(CorGANDiscriminator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.minibatch_averaging = minibatch_averaging

        # adjust input dimension for minibatch averaging
        ma_coef = 1
        if minibatch_averaging:
            ma_coef = ma_coef * 2
        model_input_dim = ma_coef * input_dim

        # 4-layer architecture matching synthEHRella exactly
        self.model = nn.Sequential(
            nn.Linear(model_input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, int(self.hidden_dim)),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, int(self.hidden_dim)),
            nn.ReLU(True),
            nn.Linear(int(self.hidden_dim), 1)
            # No sigmoid - WGAN uses unbounded critic outputs
        )

    def forward(self, x):
        if self.minibatch_averaging:
            # minibatch averaging: concatenate batch mean to each sample
            x_mean = torch.mean(x, dim=0).repeat(x.shape[0], 1)
            x = torch.cat((x, x_mean), dim=1)

        output = self.model(x)
        return output


def weights_init(m):
    """
    Custom weight initialization (synthEHRella implementation)

    Reference: synthEHRella wgancnnmimic.py lines 363-377
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def autoencoder_loss(x_output, y_target):
    """
    Autoencoder reconstruction loss (synthEHRella implementation)

    This implementation is equivalent to torch.nn.BCELoss(reduction='sum') / batch_size
    As our matrix is too sparse, first we will take a sum over the features and then
    do the mean over the batch.

    WARNING: This is NOT equivalent to torch.nn.BCELoss(reduction='mean') as the latter
    means over both features and batches.

    Reference: synthEHRella wgancnnmimic.py lines 312-323
    """
    epsilon = 1e-12
    term = y_target * torch.log(x_output + epsilon) + (1. - y_target) * torch.log(1. - x_output + epsilon)
    loss = torch.mean(-torch.sum(term, 1), 0)
    return loss


def discriminator_accuracy(predicted, y_true):
    """Calculate discriminator accuracy"""
    predicted = (predicted >= 0.5).float()
    accuracy = (predicted == y_true).float().mean()
    return accuracy.item()


class CorGAN(BaseModel):
    """
    CorGAN: Correlation-capturing Generative Adversarial Network for synthetic EHR generation.

    Uses CNNs to capture correlations between adjacent medical features by combining
    Convolutional GANs with Convolutional Autoencoders.

    Reference:
        Baowaly et al., "Synthesizing Electronic Health Records Using Improved
        Generative Adversarial Networks", JAMIA 2019.

    Args:
        dataset (SampleDataset): A fitted SampleDataset with ``input_schema = {"visits": "multi_hot"}``.
        latent_dim: Dimensionality of the generator latent space. Default: 128.
        hidden_dim: Hidden dimension for the generator MLP. Default: 128.
        batch_size: Training batch size. Default: 512.
        epochs: Total GAN training epochs. Default: 1000.
        n_epochs_pretrain: Autoencoder pre-training epochs. Default: 1.
        lr: Learning rate for all optimizers. Default: 0.001.
        weight_decay: Weight decay for Adam optimizers. Default: 0.0001.
        b1: Beta1 for Adam optimizers. Default: 0.9.
        b2: Beta2 for Adam optimizers. Default: 0.999.
        n_iter_D: Discriminator update steps per generator step. Default: 5.
        clamp_lower: Lower weight-clipping bound for WGAN critic. Default: -0.01.
        clamp_upper: Upper weight-clipping bound for WGAN critic. Default: 0.01.
        autoencoder_type: One of ``"cnn"`` (default), ``"cnn8layer"``, or ``"linear"``.
        use_adaptive_pooling: If True, add adaptive average pooling to the CNN
            autoencoder decoder so it matches any vocabulary size. Ignored when
            ``autoencoder_type`` is not ``"cnn"``. Default: True.
        minibatch_averaging: Whether to use minibatch averaging in the discriminator.
            Default: True.
        save_dir: Directory for saving checkpoints. Default: ``"./corgan_checkpoints"``.
        **kwargs: Additional arguments passed to ``BaseModel``.

    Examples:
        >>> from pyhealth.datasets.sample_dataset import InMemorySampleDataset
        >>> dataset = InMemorySampleDataset(
        ...     samples=[
        ...         {"patient_id": "p1", "visits": ["A", "B", "C"]},
        ...         {"patient_id": "p2", "visits": ["A", "C", "D"]},
        ...     ],
        ...     input_schema={"visits": "multi_hot"},
        ...     output_schema={},
        ... )
        >>> model = CorGAN(dataset, latent_dim=32, hidden_dim=32)
        >>> isinstance(model, CorGAN)
        True
    """

    def __init__(
        self,
        dataset,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        batch_size: int = 512,
        epochs: int = 1000,
        n_epochs_pretrain: int = 1,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        b1: float = 0.9,
        b2: float = 0.999,
        n_iter_D: int = 5,
        clamp_lower: float = -0.01,
        clamp_upper: float = 0.01,
        autoencoder_type: str = "cnn",
        use_adaptive_pooling: bool = True,
        minibatch_averaging: bool = True,
        save_dir: str = "./corgan_checkpoints",
        **kwargs
    ):
        super(CorGAN, self).__init__(dataset=dataset)

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.n_epochs = epochs
        self.n_epochs_pretrain = n_epochs_pretrain
        self.lr = lr
        self.weight_decay = weight_decay
        self.b1 = b1
        self.b2 = b2
        self.n_iter_D = n_iter_D
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        self.minibatch_averaging = minibatch_averaging
        self.save_dir = save_dir

        # vocabulary from the dataset's fitted processor
        processor = dataset.input_processors["visits"]
        self.input_dim = processor.size()
        # build reverse-lookup: integer index -> code string
        self._idx_to_code: List[Optional[str]] = [None] * self.input_dim
        for code, idx in processor.label_vocab.items():
            self._idx_to_code[idx] = code

        # initialize components
        # CNN autoencoder requires a minimum input size to survive its convolution chain
        # (6 layers with kernels 5,5,5,5,5,8 and strides 2,2,3,3,3,1 need at least ~500
        # features). Fall back to the linear autoencoder for small vocabularies.
        _effective_type = autoencoder_type
        if autoencoder_type not in ("linear", "cnn8layer") and self.input_dim < 500:
            _effective_type = "linear"

        if _effective_type == "cnn8layer":
            self.autoencoder = CorGAN8LayerAutoencoder(
                feature_size=self.input_dim,
                latent_dim=latent_dim,
            )
        elif _effective_type == "linear":
            self.autoencoder = CorGANLinearAutoencoder(
                feature_size=self.input_dim,
                latent_dim=latent_dim,
            )
        else:
            self.autoencoder = CorGANAutoencoder(
                feature_size=self.input_dim,
                latent_dim=latent_dim,
                use_adaptive_pooling=use_adaptive_pooling,
            )

        self.autoencoder_decoder = self.autoencoder.decoder  # separate decoder for generator

        self.generator = CorGANGenerator(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        self.discriminator = CorGANDiscriminator(
            input_dim=self.input_dim,
            hidden_dim=256,  # Match synthEHRella exactly (not hidden_dim * 2)
            minibatch_averaging=minibatch_averaging,
        )

        # apply custom weight initialization
        self._init_weights()

        # move to device (uses BaseModel's device property)
        self.to(self.device)

        # setup optimizers
        g_params = [
            {'params': self.generator.parameters()},
            {'params': self.autoencoder_decoder.parameters(), 'lr': 1e-4}
        ]
        self.optimizer_G = torch.optim.Adam(g_params, lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)
        self.optimizer_A = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, betas=(b1, b2), weight_decay=weight_decay)

        # setup tensors for training
        self.one = torch.tensor(1.0, device=self.device)
        self.mone = torch.tensor(-1.0, device=self.device)

    def _init_weights(self):
        """Initialize network weights"""
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.autoencoder.apply(weights_init)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Not used in GAN context."""
        raise NotImplementedError("Forward pass not implemented for GAN models.")

    def train_model(self, train_dataset, val_dataset=None):
        """Train the CorGAN model on a SampleDataset.

        Builds multi-hot encodings from ``train_dataset``, pre-trains the
        autoencoder, then runs WGAN adversarial training.

        Args:
            train_dataset: A fitted SampleDataset with
                ``input_schema = {"visits": "multi_hot"}``.
            val_dataset: Unused. Accepted for API compatibility.

        Returns:
            dict: Loss history with keys:
                - ``"autoencoder_loss"``: list of float, one per pretrain epoch.
                - ``"discriminator_loss"``: list of float, one per adversarial epoch.
                - ``"generator_loss"``: list of float, one per adversarial epoch.
        """
        print("Starting CorGAN training...")

        history = {
            "autoencoder_loss": [],
            "discriminator_loss": [],
            "generator_loss": [],
        }

        # build multi-hot matrix by stacking the pre-encoded tensors from MultiHotProcessor
        tensors = [train_dataset[i]["visits"] for i in range(len(train_dataset))]
        data_matrix = torch.stack(tensors).numpy()  # shape (n_patients, vocab_size)

        corgan_ds = CorGANDataset(data=data_matrix)
        sampler = torch.utils.data.sampler.RandomSampler(
            data_source=corgan_ds, replacement=True
        )
        train_dataloader = DataLoader(
            corgan_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=True,
            sampler=sampler,
        )

        # pretrain autoencoder
        print(f"Pretraining autoencoder for {self.n_epochs_pretrain} epochs...")
        for epoch_pre in range(self.n_epochs_pretrain):
            for i, samples in enumerate(train_dataloader):
                # configure input
                real_samples = samples.to(self.device)

                # generate a batch of images
                recons_samples = self.autoencoder(real_samples)

                # loss measures autoencoder's ability to reconstruct
                a_loss = autoencoder_loss(recons_samples, real_samples)

                # reset gradients
                self.optimizer_A.zero_grad()
                a_loss.backward()
                self.optimizer_A.step()

                if i % 100 == 0:
                    print(f"[Epoch {epoch_pre + 1}/{self.n_epochs_pretrain}] [Batch {i}/{len(train_dataloader)}] [A loss: {a_loss.item():.3f}]")
            history["autoencoder_loss"].append(a_loss.item())

        # adversarial training
        print(f"Starting adversarial training for {self.n_epochs} epochs...")
        gen_iterations = 0

        for epoch in range(self.n_epochs):
            epoch_start = time.time()

            for i, samples in enumerate(train_dataloader):
                # configure input
                real_samples = samples.to(self.device)

                # sample noise as generator input
                z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                for p in self.discriminator.parameters():
                    p.requires_grad = True

                # train the discriminator n_iter_D times
                if gen_iterations < 25 or gen_iterations % 500 == 0:
                    n_iter_D = 100
                else:
                    n_iter_D = self.n_iter_D

                j = 0
                while j < n_iter_D:
                    j += 1

                    # clamp parameters to a cube
                    for p in self.discriminator.parameters():
                        p.data.clamp_(self.clamp_lower, self.clamp_upper)

                    # reset gradients of discriminator
                    self.optimizer_D.zero_grad()

                    errD_real = torch.mean(self.discriminator(real_samples)).squeeze()
                    errD_real.backward(self.one)

                    # sample noise as generator input
                    z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)

                    # generate a batch of images
                    fake_samples = self.generator(z)
                    fake_samples = self.autoencoder.decode(fake_samples)

                    errD_fake = torch.mean(self.discriminator(fake_samples.detach())).squeeze()
                    errD_fake.backward(self.mone)
                    errD = errD_real - errD_fake

                    # optimizer step
                    self.optimizer_D.step()

                # -----------------
                #  Train Generator
                # -----------------

                for p in self.discriminator.parameters():
                    p.requires_grad = False

                # zero grads
                self.optimizer_G.zero_grad()

                # sample noise as generator input
                z = torch.randn(samples.shape[0], self.latent_dim, device=self.device)

                # generate a batch of images
                fake_samples = self.generator(z)
                fake_samples = self.autoencoder.decode(fake_samples)

                # loss measures generator's ability to fool the discriminator
                errG = torch.mean(self.discriminator(fake_samples)).squeeze()
                errG.backward(self.one)

                # optimizer step
                self.optimizer_G.step()
                gen_iterations += 1

            # end of epoch
            epoch_end = time.time()
            print(f"[Epoch {epoch + 1}/{self.n_epochs}] [Batch {i}/{len(train_dataloader)}] "
                  f"Loss_D: {errD.item():.3f} Loss_G: {errG.item():.3f} "
                  f"Loss_D_real: {errD_real.item():.3f} Loss_D_fake: {errD_fake.item():.3f}")
            print(f"Epoch time: {epoch_end - epoch_start:.2f} seconds")
            history["discriminator_loss"].append(errD.item())
            history["generator_loss"].append(errG.item())

        print("Training completed!")

        # save final checkpoint if save_dir is configured
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.save_dir, "corgan_final.pt")
            self.save_model(checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        return history

    def synthesize_dataset(self, num_samples: int, random_sampling: bool = True) -> List[Dict]:
        """Generate synthetic patient records.

        Each synthetic patient is represented as a flat list of codes decoded
        from the generated binary vector. This mirrors the ``multi_hot`` input
        schema used during training.

        Args:
            num_samples: Number of synthetic patients to generate.
            random_sampling: Unused; accepted for API compatibility.

        Returns:
            list of dict: Synthetic patient records. Each dict has:
                ``"patient_id"`` (str): e.g. ``"synthetic_0"``.
                ``"visits"`` (list of str): flat list of decoded ICD code strings.
                    May be empty if the generated vector has all values below the 0.5 threshold.
        """
        # set models to eval mode
        self.generator.eval()
        self.autoencoder_decoder.eval()

        device = self.device
        gen_samples = np.zeros((num_samples, self.input_dim), dtype=np.float32)
        n_batches = num_samples // self.batch_size

        with torch.no_grad():
            for i in range(n_batches):
                z = torch.randn(self.batch_size, self.latent_dim, device=device)
                gen_samples_tensor = self.generator(z)
                gen_samples_decoded = self.autoencoder.decode(gen_samples_tensor)
                gen_samples[i * self.batch_size:(i + 1) * self.batch_size, :] = (
                    gen_samples_decoded.cpu().data.numpy()
                )

            # handle remaining samples
            remaining = num_samples % self.batch_size
            if remaining > 0:
                z = torch.randn(remaining, self.latent_dim, device=device)
                gen_samples_tensor = self.generator(z)
                gen_samples_decoded = self.autoencoder.decode(gen_samples_tensor)
                gen_samples[n_batches * self.batch_size:, :] = (
                    gen_samples_decoded.cpu().data.numpy()
                )

        # binarize at threshold 0.5
        gen_samples[gen_samples >= 0.5] = 1.0
        gen_samples[gen_samples < 0.5] = 0.0

        # decode binary vectors to code strings
        results: List[Dict] = []
        for i in range(num_samples):
            row = gen_samples[i]
            codes = [
                self._idx_to_code[idx]
                for idx in np.where(row == 1.0)[0]
                if self._idx_to_code[idx] not in (None, "<pad>", "<unk>")
            ]
            results.append({
                "patient_id": f"synthetic_{i}",
                "visits": codes,
            })

        return results

    def save_model(self, path: str):
        """Save model weights and vocabulary to a checkpoint file.

        Args:
            path (str): File path to write the checkpoint (.pt file).

        Returns:
            None
        """
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'autoencoder_state_dict': self.autoencoder.state_dict(),
            'autoencoder_decoder_state_dict': self.autoencoder_decoder.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_A_state_dict': self.optimizer_A.state_dict(),
            'idx_to_code': self._idx_to_code,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
        }, path)

    def load_model(self, path: str):
        """Load model weights and vocabulary from a checkpoint file.

        Args:
            path (str): File path to read the checkpoint (.pt file).

        Returns:
            None
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        self.autoencoder_decoder.load_state_dict(checkpoint['autoencoder_decoder_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.optimizer_A.load_state_dict(checkpoint['optimizer_A_state_dict'])

        self._idx_to_code = checkpoint['idx_to_code']
        self.input_dim = checkpoint['input_dim']
        self.latent_dim = checkpoint['latent_dim']
