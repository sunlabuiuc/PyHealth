"""Variational autoencoder for images.

Author: Sameh Aissa (port to the PyHealth 2.0 API)
Paper: Auto-Encoding Variational Bayes
Paper Link: https://arxiv.org/abs/1312.6114

Description:
    A convolutional VAE that learns to reconstruct images from a small latent
    vector and to generate new ones by decoding random latent vectors. The
    image feature, its channel count and its size are all read from the
    dataset's ``ImageProcessor``, so the model adapts to whatever image task
    it is given, for example :class:`~pyhealth.tasks.COVID19CXRImageGeneration`.

    The model reports ``mode == "regression"``, with the input image as the
    target and the reconstruction as the prediction, so it trains with the
    standard :class:`~pyhealth.trainer.Trainer` and its validation metrics are
    reconstruction errors such as ``"mse"`` and ``"mae"``.
"""

import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.contrawr import ResBlock2D
from pyhealth.processors import ImageProcessor

# Encoder widths. Each block quarters the image side, and blocks are added
# until the feature map is smaller than 8 pixels or the widths run out.
ENCODER_WIDTHS = (16, 64, 256)
MIN_ENCODED_SIDE = 8


class VAE(BaseModel):
    """Convolutional variational autoencoder for a single image feature.

    The encoder is a stack of ``ResBlock2D`` blocks that maps an image to the
    mean and log-variance of a Gaussian over a ``hidden_dim``-dimensional
    latent space. The decoder maps a latent vector back to an image of the
    same shape with transposed convolutions.

    The loss is the negative evidence lower bound, averaged over the batch: a
    reconstruction term plus the KL divergence of the latent posterior from a
    standard normal prior. Images scaled to ``[0, 1]`` use binary cross-entropy
    and a sigmoid output. Images normalized by the processor (``normalize=True``)
    can fall outside that range, so they use a squared-error term and an
    unbounded output instead.

    Only image inputs are supported. Labels in the dataset's ``output_schema``
    are ignored, so a classification task's dataset can be used as is.

    Args:
        dataset: The dataset to train the model. Its ``input_schema`` must
            contain exactly one feature, processed by an ``ImageProcessor``
            with ``to_tensor=True``.
        hidden_dim: Size of the latent space. Default is 128.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import VAE
        >>> root = Path(tempfile.mkdtemp())
        >>> samples = []
        >>> for i in range(4):
        ...     path = root / f"{i}.png"
        ...     Image.new("L", (32, 32), color=40 * i).save(path)
        ...     samples.append({"patient_id": f"p{i}", "image": str(path)})
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
        ...     output_schema={},
        ... )
        >>> model = VAE(dataset=dataset, hidden_dim=8)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> output = model(**batch)
        >>> output["y_prob"].shape
        torch.Size([2, 1, 32, 32])
        >>> model.generate(num_samples=3).shape
        torch.Size([3, 1, 32, 32])
    """

    def __init__(self, dataset: SampleDataset, hidden_dim: int = 128) -> None:
        super().__init__(dataset=dataset)
        self.hidden_dim = hidden_dim

        # Step 1: find the image feature. The modality comes from the
        # processor rather than from an argument.
        if len(self.feature_keys) != 1:
            raise ValueError(
                "VAE takes exactly one input feature, an image, but the dataset "
                f"has {len(self.feature_keys)}: {self.feature_keys}"
            )
        self.feature_key = self.feature_keys[0]
        processor = dataset.input_processors[self.feature_key]
        if not isinstance(processor, ImageProcessor):
            raise TypeError(
                f"VAE only supports image inputs, but '{self.feature_key}' is "
                f"processed by {type(processor).__name__}"
            )
        if not processor.to_tensor:
            raise ValueError("VAE needs the ImageProcessor to use to_tensor=True")
        self.normalized = bool(processor.normalize)

        # Step 2: read the image shape from a processed sample, so the
        # processor's mode and image_size decide channels and size.
        image = dataset[0][self.feature_key]
        if image.dim() != 3:
            raise ValueError(
                f"Expected images of shape (channels, height, width), got "
                f"{tuple(image.shape)}"
            )
        self.image_shape: tuple[int, int, int] = tuple(image.shape)
        channels, height, width = self.image_shape
        if min(height, width) < MIN_ENCODED_SIDE:
            raise ValueError(
                f"Images must be at least {MIN_ENCODED_SIDE} pixels on each side, "
                f"got {height}x{width}"
            )

        # Step 3: build the encoder one block at a time, probing the output
        # shape, until the feature map is small enough to flatten.
        blocks: list[nn.Module] = []
        probe = torch.zeros(2, channels, height, width)
        in_channels = channels
        with torch.no_grad():
            for out_channels in ENCODER_WIDTHS:
                block = ResBlock2D(in_channels, out_channels, 2, True, True)
                probe = block.eval()(probe)
                blocks.append(block)
                in_channels = out_channels
                if min(probe.shape[-2:]) < MIN_ENCODED_SIDE:
                    break
        self.encoder = nn.Sequential(*blocks)
        self.encoded_shape: tuple[int, int, int] = tuple(probe.shape[1:])
        encoded_size = probe[0].numel()
        self.mu = nn.Linear(encoded_size, hidden_dim)
        self.logvar = nn.Linear(encoded_size, hidden_dim)

        # Step 4: build the decoder. Each transposed convolution doubles the
        # side until it reaches the image size, and decode() resizes the
        # result to the exact size when the doubling overshoots.
        self.decoder_input = nn.Linear(hidden_dim, encoded_size)
        layers: list[nn.Module] = []
        side_h, side_w = self.encoded_shape[1:]
        width_now = self.encoded_shape[0]
        while side_h < height or side_w < width:
            out_channels = max(width_now // 2, 16)
            layers.append(
                nn.ConvTranspose2d(
                    width_now, out_channels, kernel_size=4, stride=2, padding=1
                )
            )
            layers.append(nn.ReLU())
            width_now = out_channels
            side_h, side_w = side_h * 2, side_w * 2
        layers.append(nn.Conv2d(width_now, channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)
        self.output_activation = nn.Identity() if self.normalized else nn.Sigmoid()

        # The target is the input image, so the Trainer evaluates the
        # reconstruction with regression metrics.
        self.mode = "regression"

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Maps images to the mean and log-variance of the latent posterior.

        Args:
            x: Images of shape (batch_size, channels, height, width).

        Returns:
            A tuple ``(mu, logvar)``, each of shape (batch_size, hidden_dim).
        """
        h = self.encoder(x).flatten(start_dim=1)
        return self.mu(h), self.logvar(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Maps latent vectors to images.

        Args:
            z: Latent vectors of shape (batch_size, hidden_dim).

        Returns:
            Images of shape (batch_size, channels, height, width), in ``[0, 1]``
            unless the processor normalizes its images.
        """
        h = self.decoder_input(z).view(-1, *self.encoded_shape)
        x = self.decoder(h)
        if tuple(x.shape[-2:]) != self.image_shape[1:]:
            x = F.interpolate(
                x, size=self.image_shape[1:], mode="bilinear", align_corners=False
            )
        return self.output_activation(x)

    def forward(
        self, **kwargs: torch.Tensor | list[torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Encodes, samples and decodes a batch of images.

        In training mode the latent vector is sampled with the
        reparameterization trick. In eval mode the posterior mean is used, so
        reconstructions and validation metrics are deterministic.

        Args:
            **kwargs: The batch, holding the image feature as a tensor of shape
                (batch_size, channels, height, width) or a list of image tensors.

        Returns:
            A dictionary with:
                loss: the negative evidence lower bound, averaged over the batch.
                y_prob: the reconstructed images.
                y_true: the input images.
        """
        x = kwargs[self.feature_key]
        if isinstance(x, (list, tuple)):
            x = torch.stack(list(x))
        x = x.to(self.device).float()

        mu, logvar = self.encode(x)
        if self.training:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        else:
            z = mu
        x_rec = self.decode(z)

        batch_size = x.shape[0]
        if self.normalized:
            reconstruction = F.mse_loss(x_rec, x, reduction="sum")
        else:
            reconstruction = F.binary_cross_entropy(x_rec, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (reconstruction + kl) / batch_size
        return {"loss": loss, "y_prob": x_rec, "y_true": x}

    @torch.no_grad()
    def generate(
        self, num_samples: int, device: str | torch.device | None = None
    ) -> torch.Tensor:
        """Generates new images by decoding samples from the prior.

        Args:
            num_samples: Number of images to generate.
            device: Device to generate on. Defaults to the model's device.

        Returns:
            Images of shape (num_samples, channels, height, width).
        """
        was_training = self.training
        self.eval()
        z = torch.randn(num_samples, self.hidden_dim, device=device or self.device)
        images = self.decode(z)
        self.train(was_training)
        return images
