# Author: Giovanni Dall'Olio
# Description: Image and categorical-sequence VAEs for PyHealth 2.0.

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel, EmbeddingModel
from pyhealth.processors import ImageProcessor, SequenceProcessor


class VAE(BaseModel):
    """Variational autoencoder for images or categorical EHR sequences.

    The input modality is inferred from its fitted processor. Image inputs use
    a convolutional encoder and decoder with a pixel reconstruction objective.
    ``SequenceProcessor`` inputs use a recurrent encoder and an autoregressive
    decoder trained to reconstruct code tokens. Both paths use the same latent
    sampling and KL-divergence objective. The model reconstructs inputs rather
    than predicting the dataset's output label, so ``Trainer`` reports loss
    without classification metrics.

    Args:
        dataset: A fitted ``SampleDataset`` with one supported input feature.
        feature_key: Image or sequence key. If omitted, the dataset must have
            exactly one supported input; specify it when there are several.
        condition_key: Optional binary or multiclass output label used to
            condition image generation. If omitted, the sole output label is
            used for image datasets. Sequence generation is unconditional.
        latent_dim: Size of the latent representation. Defaults to 128.
        hidden_dim: Number of convolutional channels or token-embedding units.
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
        sequence_keys = [
            key
            for key, processor in dataset.input_processors.items()
            if isinstance(processor, SequenceProcessor)
        ]
        supported_keys = image_keys + sequence_keys
        if feature_key is None:
            if len(supported_keys) != 1:
                raise ValueError(
                    "VAE requires one ImageProcessor or SequenceProcessor input; "
                    "specify feature_key when the dataset has multiple inputs."
                )
            feature_key = supported_keys[0]
        if feature_key in image_keys:
            self.modality = "image"
        elif feature_key in sequence_keys:
            self.modality = "sequence"
        else:
            raise ValueError(
                f"feature_key {feature_key!r} must use ImageProcessor or "
                "SequenceProcessor."
            )

        self.feature_key = feature_key
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.condition_key = None
        self.condition_embedding = None
        self.num_conditions = 0

        if self.modality == "image":
            self._init_image(dataset, condition_key)
        else:
            if condition_key is not None:
                raise ValueError(
                    "condition_key is only supported for image generation."
                )
            self._init_sequence(dataset)

        # This is an input-reconstruction model; dataset labels are optional
        # conditions and are not predictions for Trainer's task metrics.
        self._mode = None

    def _init_image(
        self, dataset: SampleDataset, condition_key: str | None
    ) -> None:
        """Build the image encoder, latent projections, and decoder."""
        if condition_key is None and len(self.label_keys) == 1:
            condition_key = self.label_keys[0]
        if condition_key is not None:
            if condition_key not in dataset.output_processors:
                raise ValueError(
                    f"condition_key {condition_key!r} must be in output_schema."
                )
            processor = dataset.output_processors[condition_key]
            mode = self._resolve_mode(dataset.output_schema[condition_key])
            if mode not in ("binary", "multiclass"):
                raise ValueError(
                    "condition_key must use a binary or multiclass label processor."
                )
            self.num_conditions = 2 if mode == "binary" else int(processor.size())
            if self.num_conditions < 1:
                raise ValueError("condition_key must have at least one category.")
            self.condition_key = condition_key
            self.condition_embedding = nn.Embedding(
                self.num_conditions, self.latent_dim
            )

        image_processor = self.dataset.input_processors[self.feature_key]
        self.image_size = image_processor.image_size
        self.channels = {"L": 1, "RGB": 3, "RGBA": 4}.get(
            image_processor.mode, 3
        )
        channels = self.hidden_dim
        self.image_encoder = nn.Sequential(
            nn.Conv2d(self.channels, channels, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 4, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        encoded_size = channels * 4 * 4 * 4
        self.image_mu = nn.Linear(encoded_size, self.latent_dim)
        self.image_log_var = nn.Linear(encoded_size, self.latent_dim)
        decoder_input = self.latent_dim * (2 if self.condition_key else 1)
        self.image_decoder_input = nn.Linear(decoder_input, encoded_size)
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(channels, self.channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def _init_sequence(self, dataset: SampleDataset) -> None:
        """Build a recurrent VAE for categorical code sequences."""
        self.sequence_processor = dataset.input_processors[self.feature_key]
        self.pad_token = int(self.sequence_processor.code_vocab.get("<pad>", 0))
        self.vocab_size = self.sequence_processor.vocab_size()
        self.embedding_model = EmbeddingModel(dataset, self.hidden_dim)
        self.sequence_encoder = nn.GRU(
            self.hidden_dim, self.hidden_dim, batch_first=True
        )
        self.sequence_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.sequence_log_var = nn.Linear(self.hidden_dim, self.latent_dim)
        self.sequence_start = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.latent_to_hidden = nn.Linear(self.latent_dim, self.hidden_dim)
        self.sequence_decoder = nn.GRU(
            self.hidden_dim, self.hidden_dim, batch_first=True
        )
        self.sequence_output = nn.Linear(self.hidden_dim, self.vocab_size)

    @staticmethod
    def _sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Draw a latent sample using the reparameterization trick."""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def _condition(
        self, labels: torch.Tensor | None, batch_size: int
    ) -> torch.Tensor | None:
        """Convert encoded image-class labels to condition embeddings."""
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

    def _forward_image(
        self, kwargs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct an image batch and calculate its VAE objective."""
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing image input {self.feature_key!r}.")
        image = kwargs[self.feature_key].to(self.device, dtype=torch.float32)
        if image.ndim != 4 or image.shape[1] != self.channels:
            raise ValueError(
                "Image input must have shape (batch, channels, height, width) "
                f"with {self.channels} channels."
            )
        if bool(((image < 0) | (image > 1)).any()):
            raise ValueError(
                "Image values must be in [0, 1]; configure ImageProcessor "
                "with normalize=False."
            )

        encoded = self.image_encoder(image)
        mu = self.image_mu(encoded)
        log_var = self.image_log_var(encoded).clamp(min=-20, max=20)
        latent = self._sample(mu, log_var)
        condition = self._condition(
            kwargs.get(self.condition_key) if self.condition_key else None,
            image.shape[0],
        )
        if condition is not None:
            latent = torch.cat((latent, condition), dim=1)
        decoded = self.image_decoder_input(latent).reshape(
            image.shape[0], self.hidden_dim * 4, 4, 4
        )
        reconstruction = self.image_decoder(decoded)
        reconstruction = F.interpolate(
            reconstruction,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        reconstruction_loss = F.binary_cross_entropy(
            reconstruction, image, reduction="sum"
        ) / image.shape[0]
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp())
        loss = reconstruction_loss + kl_loss / image.shape[0]
        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "mu": mu,
            "log_var": log_var,
        }

    def _forward_sequence(
        self, kwargs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Reconstruct categorical tokens with an autoregressive decoder."""
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing sequence input {self.feature_key!r}.")
        tokens = kwargs[self.feature_key].to(self.device).long()
        if tokens.ndim != 2:
            raise ValueError(
                "Sequence input must have shape (batch, sequence_length)."
            )
        mask = tokens != self.pad_token
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        embeddings = self.embedding_model({self.feature_key: tokens})[
            self.feature_key
        ]
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.sequence_encoder(packed)
        mu = self.sequence_mu(hidden[-1])
        log_var = self.sequence_log_var(hidden[-1]).clamp(min=-20, max=20)
        latent = self._sample(mu, log_var)

        start = self.sequence_start.expand(tokens.shape[0], -1, -1)
        if tokens.shape[1] > 1:
            previous_embeddings = self.embedding_model(
                {self.feature_key: tokens[:, :-1]}
            )[self.feature_key]
            decoder_inputs = torch.cat((start, previous_embeddings), dim=1)
        else:
            decoder_inputs = start
        decoder_hidden = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        decoded, _ = self.sequence_decoder(decoder_inputs, decoder_hidden)
        logits = self.sequence_output(decoded)

        token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.pad_token,
            reduction="sum",
        )
        reconstruction_loss = token_loss / mask.sum().clamp(min=1)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp())
        loss = reconstruction_loss + kl_loss / tokens.shape[0]
        return {
            "loss": loss,
            "reconstruction_logits": logits,
            "mu": mu,
            "log_var": log_var,
            "mask": mask,
        }

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode and reconstruct an image or categorical sequence batch."""
        if self.modality == "image":
            return self._forward_image(kwargs)
        return self._forward_sequence(kwargs)

    def generate(
        self,
        num_samples: int | None = None,
        sequence_length: int | None = None,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate images or categorical sequences from latent noise.

        Args:
            num_samples: Number of samples for unconditional generation.
            sequence_length: Number of tokens to generate for a sequence VAE.
            conditions: Encoded class indices for a conditional image VAE.

        Returns:
            Images shaped ``(batch, channels, height, width)``, or generated
            token indices shaped ``(batch, sequence_length)``.

        Examples:
            >>> model.generate(num_samples=2, sequence_length=5)  # doctest: +SKIP
        """
        if self.modality == "sequence":
            if conditions is not None:
                raise ValueError("Sequence generation does not accept conditions.")
            if num_samples is None or num_samples < 1:
                raise ValueError("num_samples must be a positive integer.")
            if sequence_length is None or sequence_length < 1:
                raise ValueError("sequence_length must be a positive integer.")
            latent = torch.randn(num_samples, self.latent_dim, device=self.device)
            hidden = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
            previous = self.sequence_start.expand(num_samples, -1, -1)
            generated = []
            for _ in range(sequence_length):
                decoded, hidden = self.sequence_decoder(previous, hidden)
                logits = self.sequence_output(decoded[:, 0, :])
                logits[:, self.pad_token] = torch.finfo(logits.dtype).min
                token = logits.argmax(dim=-1)
                generated.append(token)
                previous = self.embedding_model(
                    {self.feature_key: token.unsqueeze(1)}
                )[self.feature_key]
            return torch.stack(generated, dim=1)

        if self.condition_key and conditions is None:
            raise ValueError("Conditional generation requires encoded conditions.")
        if self.condition_key is None and conditions is not None:
            raise ValueError("This VAE was initialized without a condition label.")
        if conditions is not None:
            conditions = conditions.to(self.device).reshape(-1).long()
            num_samples = conditions.shape[0]
            if bool(
                ((conditions < 0) | (conditions >= self.num_conditions)).any()
            ):
                raise ValueError(
                    "Condition labels are outside the fitted label vocabulary."
                )
        if num_samples is None or num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        latent = torch.randn(num_samples, self.latent_dim, device=self.device)
        if conditions is not None:
            latent = torch.cat(
                (latent, self.condition_embedding(conditions)), dim=1
            )
        decoded = self.image_decoder_input(latent).reshape(
            num_samples, self.hidden_dim * 4, 4, 4
        )
        images = self.image_decoder(decoded)
        return F.interpolate(
            images,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
