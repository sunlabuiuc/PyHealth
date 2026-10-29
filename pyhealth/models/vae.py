# Author: Giovanni Dall'Olio
# Description: Conditional image VAE for PyHealth 2.0 SampleDataset inputs.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import ImageProcessor


class VAE(BaseModel):
    """Conditional variational autoencoder for image-valued samples.

    The model reconstructs one image feature and optionally conditions the
    reconstruction on one categorical output label. Its loss is the sum of
    binary cross-entropy reconstruction loss and the latent KL divergence.
    Since the model predicts its input rather than the dataset label, its
    ``mode`` is ``None`` and :class:`~pyhealth.trainer.Trainer` evaluates loss.

    Args:
        dataset: A fitted ``SampleDataset`` with one image input and one
            categorical output label.
        feature_key: Image key in ``dataset.input_schema``. If omitted, the
            dataset must have exactly one image input.
        condition_key: Optional categorical output key used for conditional
            image generation. If omitted and the dataset has one output key,
            that key is used. An empty output schema creates an unconditional
            VAE.
        latent_dim: Size of the latent representation. Defaults to 128.
        hidden_dim: Number of channels in the convolutional encoder and decoder.
            Defaults to 32.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> import numpy as np
        >>> from PIL import Image
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import VAE
        >>> temp_dir = TemporaryDirectory()
        >>> image_path = f"{temp_dir.name}/image.png"
        >>> Image.fromarray(np.zeros((32, 32), dtype=np.uint8)).save(image_path)
        >>> samples = [
        ...     {"image": image_path, "label": "normal"},
        ...     {"image": image_path, "label": "disease"},
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"image": ("image", {"image_size": 32, "mode": "L"})},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="vae-example",
        ... )
        >>> model = VAE(dataset, latent_dim=16, hidden_dim=8)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> result = model(**batch)
        >>> result["reconstruction"].shape
        torch.Size([2, 1, 32, 32])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str | None = None,
        condition_key: str | None = None,
        latent_dim: int = 128,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__(dataset=dataset)
        if latent_dim < 1 or hidden_dim < 1:
            raise ValueError("latent_dim and hidden_dim must be positive integers.")

        image_keys = [
            key
            for key, processor in dataset.input_processors.items()
            if isinstance(processor, ImageProcessor)
        ]
        if feature_key is None:
            if len(image_keys) != 1:
                raise ValueError(
                    "VAE requires exactly one ImageProcessor input; specify "
                    "feature_key when the dataset has multiple image inputs."
                )
            feature_key = image_keys[0]
        elif feature_key not in image_keys:
            raise ValueError(
                f"feature_key {feature_key!r} must use an ImageProcessor."
            )
        self.feature_key = feature_key

        if condition_key is None and len(self.label_keys) == 1:
            condition_key = self.label_keys[0]
        if condition_key is not None:
            if condition_key not in dataset.output_processors:
                raise ValueError(
                    f"condition_key {condition_key!r} must be in output_schema."
                )
            condition_processor = dataset.output_processors[condition_key]
            condition_mode = self._resolve_mode(dataset.output_schema[condition_key])
            if condition_mode not in ("binary", "multiclass"):
                raise ValueError(
                    "condition_key must use a binary or multiclass label processor."
                )
            self.num_conditions = (
                2 if condition_mode == "binary" else int(condition_processor.size())
            )
            if self.num_conditions < 1:
                raise ValueError("condition_key must have at least one category.")
            self.condition_key = condition_key
            self.condition_embedding = nn.Embedding(self.num_conditions, latent_dim)
        else:
            self.num_conditions = 0
            self.condition_key = None
            self.condition_embedding = None

        image_processor = dataset.input_processors[feature_key]
        self.image_size = image_processor.image_size
        self.channels = {"L": 1, "RGB": 3, "RGBA": 4}.get(
            image_processor.mode, 3
        )
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        encoded_size = hidden_dim * 4 * 4 * 4
        self.mu = nn.Linear(encoded_size, latent_dim)
        self.log_var = nn.Linear(encoded_size, latent_dim)

        decoder_input = latent_dim * (2 if self.condition_key else 1)
        self.decoder_input = nn.Linear(decoder_input, hidden_dim * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, self.channels, 4, 2, 1),
            nn.Sigmoid(),
        )
        # Reconstruction is the objective, not a prediction of the dataset's
        # label. Trainer therefore reports loss without classification metrics.
        self._mode = None

    def _condition(
        self, labels: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Convert encoded class labels to condition embeddings."""
        if self.condition_key is None:
            if labels is not None:
                raise ValueError("This VAE was initialized without a condition label.")
            return None
        if labels is None:
            raise ValueError(
                f"Conditional generation requires {self.condition_key!r} labels."
            )
        labels = labels.to(self.device).reshape(batch_size, -1)
        if labels.shape[1] != 1:
            raise ValueError(
                "VAE conditioning supports one categorical label per image."
            )
        indices = labels[:, 0].long()
        if bool(((indices < 0) | (indices >= self.num_conditions)).any()):
            raise ValueError(
                "Condition labels are outside the fitted label vocabulary."
            )
        return self.condition_embedding(indices)

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode and reconstruct a batch of images."""
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing image input {self.feature_key!r}.")
        image = kwargs[self.feature_key].to(self.device, dtype=torch.float32)
        if image.ndim != 4:
            raise ValueError(
                "Image input must have shape (batch, channels, height, width)."
            )
        if image.shape[1] != self.channels:
            raise ValueError(
                f"Expected {self.channels} image channels, got {image.shape[1]}."
            )
        if bool(((image < 0) | (image > 1)).any()):
            raise ValueError(
                "Image values must be in [0, 1]; configure ImageProcessor "
                "with normalize=False."
            )

        encoded = self.encoder(image)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded).clamp(min=-20, max=20)
        latent = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
        condition = self._condition(
            kwargs.get(self.condition_key) if self.condition_key else None,
            image.shape[0],
        )
        if condition is not None:
            latent = torch.cat((latent, condition), dim=1)
        decoded = self.decoder_input(latent).reshape(
            image.shape[0], self.hidden_dim * 4, 4, 4
        )
        reconstruction = self.decoder(decoded)
        reconstruction = F.interpolate(
            reconstruction, size=image.shape[-2:], mode="bilinear", align_corners=False
        )
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, image, reduction="sum"
        )
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp())
        loss = (reconstruction_loss + kl_loss) / image.shape[0]
        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "mu": mu,
            "log_var": log_var,
        }

    def generate(
        self,
        conditions: torch.Tensor | None = None,
        num_samples: int | None = None,
    ) -> torch.Tensor:
        """Generate images from latent noise and optional encoded class labels.

        Args:
            conditions: Encoded label indices, one per generated image.
            num_samples: Number of unconditional samples. Required when
                ``conditions`` is None.

        Returns:
            Generated images shaped ``(batch, channels, height, width)``.

        Examples:
            >>> model.generate(conditions=torch.tensor([0, 1]))  # doctest: +SKIP
        """
        if self.condition_key and conditions is None:
            raise ValueError("Conditional generation requires encoded conditions.")
        if self.condition_key is None and conditions is not None:
            raise ValueError("This VAE was initialized without a condition label.")
        if conditions is None:
            if num_samples is None or num_samples < 1:
                raise ValueError(
                    "num_samples must be positive for unconditional generation."
                )
            batch_size = num_samples
            latent = torch.randn(batch_size, self.latent_dim, device=self.device)
        else:
            conditions = conditions.to(self.device).reshape(-1).long()
            batch_size = conditions.shape[0]
            if batch_size == 0:
                raise ValueError("At least one condition is required.")
            latent = torch.randn(batch_size, self.latent_dim, device=self.device)
            if self.condition_embedding is not None:
                if bool(((conditions < 0) | (conditions >= self.num_conditions)).any()):
                    raise ValueError(
                        "Condition labels are outside the fitted vocabulary."
                    )
                condition = self.condition_embedding(conditions)
                latent = torch.cat((latent, condition), dim=1)
        decoded = self.decoder_input(latent).reshape(
            batch_size, self.hidden_dim * 4, 4, 4
        )
        images = self.decoder(decoded)
        return F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
