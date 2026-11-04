# Author: Giovanni Dall'Olio
# Description: Image and categorical-sequence VAEs for PyHealth 2.0.

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.models.contrawr import ResBlock2D
from pyhealth.processors import ImageProcessor, SequenceProcessor


ENCODER_WIDTHS = (16, 64, 256)
MIN_ENCODED_SIDE = 8


class VAE(BaseModel):
    """Variational autoencoder for images or categorical EHR sequences.

    The input modality is inferred from its fitted processor. Image inputs use
    a convolutional encoder and decoder with a pixel reconstruction objective.
    ``SequenceProcessor`` inputs use a recurrent encoder and an autoregressive
    decoder trained to reconstruct code tokens. Both paths use the same latent
    sampling and KL-divergence objective. The model reconstructs inputs rather
    than predicting the dataset's output label. ``Trainer.evaluate()`` reports
    the loss for sequence VAEs; ``Trainer.inference()`` returns each reconstructed
    sequence and its token probabilities as variable-length per-sample arrays.
    Each token probability is a dense vector over the fitted code vocabulary, so
    inference memory grows with the number of tokens times the vocabulary size.

    Args:
        dataset: A fitted ``SampleDataset`` with one supported input feature.
        hidden_dim: Size of the latent representation. Defaults to 128, matching
            the image VAE API before the PyHealth 2.0 port.
        feature_key: Image or sequence key. If omitted, the dataset must have
            exactly one supported input; specify it when there are several.
        condition_key: Optional binary or multiclass output label used to
            condition image reconstruction and generation. Labels are ignored
            unless this key is explicitly provided. Sequence generation is
            unconditional.
        sequence_hidden_dim: Embedding and recurrent hidden size for sequence
            VAEs. Defaults to 32.

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
        >>> model = VAE(dataset, hidden_dim=16)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> result = model(**batch)
        >>> result["reconstruction"].shape
        torch.Size([2, 1, 32, 32])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 128,
        *,
        feature_key: str | None = None,
        condition_key: str | None = None,
        sequence_hidden_dim: int = 32,
    ) -> None:
        super().__init__(dataset=dataset)
        if hidden_dim < 1 or sequence_hidden_dim < 1:
            raise ValueError(
                "hidden_dim and sequence_hidden_dim must be positive integers."
            )

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
        # Preserve hidden_dim as the latent size from the image VAE API that
        # preceded this port. New architecture widths have explicit names.
        self.latent_dim = hidden_dim
        self.hidden_dim = hidden_dim
        self.sequence_hidden_dim = sequence_hidden_dim
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
        self._mode = "regression" if self.modality == "image" else None
        self.inference_output_mode = (
            "categorical_sequence_reconstruction"
            if self.modality == "sequence"
            else None
        )

    def _init_image(
        self, dataset: SampleDataset, condition_key: str | None
    ) -> None:
        """Build the image encoder, latent projections, and decoder."""
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
        if not image_processor.to_tensor:
            raise ValueError("VAE needs the ImageProcessor to use to_tensor=True")
        image = dataset[0][self.feature_key]
        if not torch.is_tensor(image) or image.ndim != 3:
            shape = tuple(image.shape) if hasattr(image, "shape") else type(image)
            raise ValueError(
                "Expected ImageProcessor to return a tensor shaped "
                f"(channels, height, width), got {shape}."
            )
        self.image_shape: tuple[int, int, int] = tuple(image.shape)
        self.channels, height, width = self.image_shape
        if self.channels < 1:
            raise ValueError("Image inputs must have at least one channel.")
        self.normalized = bool(image_processor.normalize)
        self.output_activation = nn.Identity() if self.normalized else nn.Sigmoid()
        if min(height, width) < MIN_ENCODED_SIDE:
            raise ValueError(
                f"Images must be at least {MIN_ENCODED_SIDE} pixels on each side, "
                f"got {height}x{width}."
            )
        # Keep the image module names, shapes, and ResBlock2D architecture from
        # the merged image VAE. This preserves strict state_dict loading.
        blocks: list[nn.Module] = []
        probe = torch.zeros(2, self.channels, height, width)
        in_channels = self.channels
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
        self.mu = nn.Linear(encoded_size, self.latent_dim)
        self.logvar = nn.Linear(encoded_size, self.latent_dim)
        self.decoder_input = nn.Linear(self.latent_dim, encoded_size)
        decoder_layers: list[nn.Module] = []
        side_h, side_w = self.encoded_shape[1:]
        width_now = self.encoded_shape[0]
        while side_h < height or side_w < width:
            out_channels = max(width_now // 2, 16)
            decoder_layers.append(
                nn.ConvTranspose2d(
                    width_now, out_channels, kernel_size=4, stride=2, padding=1
                )
            )
            decoder_layers.append(nn.ReLU())
            width_now = out_channels
            side_h, side_w = side_h * 2, side_w * 2
        decoder_layers.append(
            nn.Conv2d(width_now, self.channels, kernel_size=3, padding=1)
        )
        self.decoder = nn.Sequential(*decoder_layers)

    def _init_sequence(self, dataset: SampleDataset) -> None:
        """Build a recurrent VAE for categorical code sequences."""
        self.sequence_processor = dataset.input_processors[self.feature_key]
        self.pad_token = int(self.sequence_processor.code_vocab.get("<pad>", 0))
        self.unk_token = self.sequence_processor.code_vocab.get("<unk>")
        self.vocab_size = self.sequence_processor.vocab_size()
        self.token_embedding = nn.Embedding(
            self.vocab_size,
            self.sequence_hidden_dim,
            padding_idx=self.pad_token,
        )
        self.sequence_encoder = nn.GRU(
            self.sequence_hidden_dim, self.sequence_hidden_dim, batch_first=True
        )
        self.sequence_mu = nn.Linear(self.sequence_hidden_dim, self.latent_dim)
        self.sequence_log_var = nn.Linear(
            self.sequence_hidden_dim, self.latent_dim
        )
        self.sequence_start = nn.Parameter(
            torch.zeros(1, 1, self.sequence_hidden_dim)
        )
        self.latent_to_hidden = nn.Linear(
            self.latent_dim, self.sequence_hidden_dim
        )
        self.sequence_decoder = nn.GRU(
            self.sequence_hidden_dim, self.sequence_hidden_dim, batch_first=True
        )
        self.sequence_output = nn.Linear(
            self.sequence_hidden_dim, self.vocab_size
        )

    @staticmethod
    def _sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Draw a latent sample using the reparameterization trick."""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images as ``(mu, logvar)``; retained for image API compatibility."""
        if self.modality != "image":
            raise ValueError("encode() is available for image VAEs.")
        encoded = self.encoder(x.to(self.device, dtype=torch.float32)).flatten(1)
        return self.mu(encoded), self.logvar(encoded)

    def decode(
        self, z: torch.Tensor, conditions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decode latent image vectors; retained for image API compatibility."""
        if self.modality != "image":
            raise ValueError("decode() is available for image VAEs.")
        z = z.to(self.device)
        if self.condition_key is not None:
            z = z + self._condition(conditions, z.shape[0])
        elif conditions is not None:
            raise ValueError("This VAE was initialized without a condition label.")
        decoded = self.decoder_input(z).view(-1, *self.encoded_shape)
        image = self.decoder(decoded)
        if tuple(image.shape[-2:]) != self.image_shape[1:]:
            image = F.interpolate(
                image, size=self.image_shape[1:], mode="bilinear", align_corners=False
            )
        return self.output_activation(image)

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
        self, kwargs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Reconstruct an image batch and calculate its VAE objective."""
        if self.feature_key not in kwargs:
            raise KeyError(f"Missing image input {self.feature_key!r}.")
        image = kwargs[self.feature_key]
        if isinstance(image, (list, tuple)):
            image = torch.stack(image, dim=0)
        image = image.to(self.device, dtype=torch.float32)
        if image.ndim != 4 or image.shape[1] != self.channels:
            raise ValueError(
                "Image input must have shape (batch, channels, height, width) "
                f"with {self.channels} channels."
            )
        if not self.normalized and bool(((image < 0) | (image > 1)).any()):
            raise ValueError(
                "Image values must be in [0, 1]; configure ImageProcessor "
                "with normalize=False."
            )

        mu, log_var = self.encode(image)
        log_var = log_var.clamp(min=-20, max=20)
        latent = self._sample(mu, log_var) if self.training else mu
        reconstruction = self.decode(
            latent,
            kwargs.get(self.condition_key) if self.condition_key else None,
        )
        reconstruction_loss = F.mse_loss(
            reconstruction, image, reduction="sum"
        ) if self.normalized else F.binary_cross_entropy(
            reconstruction, image, reduction="sum"
        )
        reconstruction_loss = reconstruction_loss / image.shape[0]
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.square() - log_var.exp())
        loss = reconstruction_loss + kl_loss / image.shape[0]
        return {
            "loss": loss,
            "reconstruction": reconstruction,
            "y_prob": reconstruction,
            "y_true": image,
            "mu": mu,
            "log_var": log_var,
        }

    def _forward_sequence(
        self, kwargs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
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
        embeddings = self.token_embedding(tokens)
        packed = pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.sequence_encoder(packed)
        mu = self.sequence_mu(hidden[-1])
        log_var = self.sequence_log_var(hidden[-1]).clamp(min=-20, max=20)
        latent = self._sample(mu, log_var) if self.training else mu

        start = self.sequence_start.expand(tokens.shape[0], -1, -1)
        if tokens.shape[1] > 1:
            previous_embeddings = self.token_embedding(tokens[:, :-1])
            decoder_inputs = torch.cat((start, previous_embeddings), dim=1)
        else:
            decoder_inputs = start
        decoder_hidden = torch.tanh(self.latent_to_hidden(latent)).unsqueeze(0)
        decoded, _ = self.sequence_decoder(decoder_inputs, decoder_hidden)
        logits = self.sequence_output(decoded)

        # Sum token NLL within each sequence, then average both terms per
        # sequence so reconstruction and KL have matching batch reductions.
        token_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tokens.reshape(-1),
            ignore_index=self.pad_token,
            reduction="none",
        ).reshape_as(tokens)
        reconstruction_loss = token_loss.sum(dim=1).mean()
        kl_per_sequence = -0.5 * torch.sum(
            1 + log_var - mu.square() - log_var.exp(), dim=1
        )
        kl_loss = kl_per_sequence.mean()
        loss = reconstruction_loss + kl_loss
        token_probabilities = F.softmax(logits, dim=-1)
        return {
            "loss": loss,
            "reconstruction_logits": logits,
            "y_true": tokens,
            "y_prob": token_probabilities,
            "mu": mu,
            "log_var": log_var,
            "mask": mask,
        }

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode and reconstruct an image or categorical sequence batch."""
        if self.modality == "image":
            return self._forward_image(kwargs)
        return self._forward_sequence(kwargs)

    @torch.no_grad()
    def generate(
        self,
        num_samples: int | None = None,
        device: str | torch.device | None = None,
        *,
        sequence_length: int | None = None,
        conditions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate images or categorical sequences from latent noise.

        Args:
            num_samples: Number of samples for unconditional generation.
            device: Requested device for generated sequence tokens. Sequence
                latents are sampled on the model's device, then tokens are
                moved here. Image latents are sampled here and decoded on the
                model's device. Defaults to the model's device.
            sequence_length: Number of tokens to generate for a sequence VAE.
            conditions: Encoded class indices for a conditional image VAE.

        Returns:
            Images shaped ``(batch, channels, height, width)``, or generated
            token indices shaped ``(batch, sequence_length)``.

        Examples:
            >>> model.generate(num_samples=2, sequence_length=5)  # doctest: +SKIP
        """
        generation_device = device or self.device
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
                if self.unk_token is not None:
                    logits[:, self.unk_token] = torch.finfo(logits.dtype).min
                token = torch.multinomial(
                    F.softmax(logits, dim=-1), num_samples=1
                ).squeeze(-1)
                generated.append(token)
                previous = self.token_embedding(token.unsqueeze(1))
            return torch.stack(generated, dim=1).to(generation_device)

        if self.condition_key and conditions is None:
            raise ValueError("Conditional generation requires encoded conditions.")
        if self.condition_key is None and conditions is not None:
            raise ValueError("This VAE was initialized without a condition label.")
        if conditions is not None:
            conditions = conditions.to(generation_device).reshape(-1).long()
            num_samples = conditions.shape[0]
            if bool(
                ((conditions < 0) | (conditions >= self.num_conditions)).any()
            ):
                raise ValueError(
                    "Condition labels are outside the fitted label vocabulary."
                )
        if num_samples is None or num_samples < 1:
            raise ValueError("num_samples must be a positive integer.")
        latent = torch.randn(
            num_samples, self.latent_dim, device=generation_device
        )
        return self.decode(latent, conditions)
