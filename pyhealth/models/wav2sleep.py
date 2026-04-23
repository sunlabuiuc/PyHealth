"""
Author(s): Bronze Frazer
NetID(s):  bfrazer2
Paper:     wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification from Physiological Signals
Link:      https://arxiv.org/abs/2411.04644
Desc:      PyHealth Model implementation of wav2sleep for sleep stage classification
"""

from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.functional import F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

# Global hyperparameters (as used in the paper)
FEATURE_DIM = 128
ACTIVATION_FUNCTION = nn.GELU()
DROPOUT_RATE = 0.1


class ResidualBlock(nn.Module):
    """Residual Convolution Block to encode a signal

    Args:
        c_in: Number of input channels
        c_out: Number of output channels
        pool_size: downsampling factor
    """

    def __init__(self, c_in: int, c_out: int, pool_size: int = 2) -> None:
        super().__init__()

        def create_conv_block(
            input_dim: int, output_dim: int, kernel_size: int = 3
        ) -> nn.Sequential:
            """Create a Convolution Block

            Args:
                input_dim: Dimension of the input
                output_dim: Dimension of the output
                kernel_size: Size of the convolutional kernel

            Returns:
                nn.Sequential: A convolutional block with instance normalization
            """
            pad = kernel_size // 2
            return nn.Sequential(
                nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, padding=pad),
                nn.InstanceNorm1d(output_dim),
                ACTIVATION_FUNCTION,
                nn.Dropout(DROPOUT_RATE),
            )

        self.conv1 = create_conv_block(c_in, c_out)
        self.conv2 = create_conv_block(c_out, c_out)
        self.conv3 = create_conv_block(c_out, c_out)
        self.pool = nn.MaxPool1d(pool_size)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
        self.activation = ACTIVATION_FUNCTION
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the residual block

        Args:
            x: Tensor input; shape = (batch_size, c_in, length)

        Returns:
            Tensor: Output tensor; shape = (batch_size, c_out, length//2)
        """
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += residual
        out = self.pool(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class SignalEncoder(nn.Module):
    """Architecture for a Signal Encoder as described in Section 3.1

    Turns a raw input signal into a per-epoch feature vector sequence.

    A Signal Encoder consists of a stack of residual layers.
    Each layer contains three convolutional layers followed by a
    max pooling layer to downsample the signal by a factor of 2.
    Residual layers are followed by a reshape operation and a
    time-distributed dense layer to produce the sequence of feature vectors.

    Args:
        signal_sample_rate: The original sample rate (Hz) used when measuring the signal

    Raises:
        ValueError: If signal_sample_rate is not 1024 or 256.
    """

    def __init__(self, signal_sample_rate: int) -> None:
        super().__init__()

        self.T = 1200  # total epochs after preprocessing
        self.signal_sample_rate = signal_sample_rate

        if signal_sample_rate == 1024:
            channels = [1, 16, 16, 32, 32, 64, 64, 128, 128]
        elif signal_sample_rate == 256:
            channels = [1, 16, 32, 64, 64, 128, 128]
        else:
            raise ValueError(
                f"{signal_sample_rate} is not a valid resample rate. "
                "Channel progression cannot be assigned"
            )

        channel_progression = list(zip(channels, channels[1:]))

        blocks = [
            layer
            for input_dim, output_dim in channel_progression
            for layer in (
                ResidualBlock(c_in=input_dim, c_out=output_dim),
                nn.InstanceNorm1d(output_dim, affine=True),
            )
        ]

        self.encoder = nn.Sequential(*blocks)
        self.epoch_dim = (
            channels[-1] * 4
        )  # Flattened dimension for time-distributed dense layer
        self.dense = nn.Linear(self.epoch_dim, FEATURE_DIM)
        self.activation = ACTIVATION_FUNCTION

    def forward(self, x: Tensor) -> Tensor:
        """Encode biosignal to a sequence of features

        Args:
            x: A raw signal Tensor; shape = (batch_size, 1, signal_measurements)

        Returns:
            Tensor: A sequence of per-epoch feature vectors; shape = (batch_size, T, feature_dim)
        """
        batch_size = x.shape[0]

        # Split into epochs — treat each epoch independently (time-distributed)
        x = x.view(
            batch_size * self.T, 1, self.signal_sample_rate
        )  # (batch_size*T, 1, k)

        z = self.encoder(x)  # (batch_size*T, feature_dim, 4)
        # Flatten spatial dim for the dense layer
        z = z.view(batch_size * self.T, -1)  # (batch_size*T, 512)
        # Time-distributed dense: same weights applied to every epoch
        z = self.activation(self.dense(z))
        # Reassemble the time axis
        z = z.view(batch_size, self.T, FEATURE_DIM)
        return z


class EpochMixer(nn.Module):
    """Architecture for the Epoch Mixer as described in Section 3.2

    Provides a unified representation of sleep epochs.
    Uses a transformer encoder with a learnable CLS vector
    that fuses information among a set of modalities.

    Args:
        num_transformer_layers: The number of transformer layers to use
        hidden_dimension: The hidden dimension of a transformer layer
        num_attention_heads: The number of attention heads to use for a transformer layer
        modalities: The list of selected modalities (order of modalities matters)
        stochastic_mask_probabilities: Probability that a modality will be masked during training
    """

    def __init__(
        self,
        num_transformer_layers: int = 2,
        hidden_dimension: int = 512,
        num_attention_heads: int = 8,
        modalities: Optional[List[str]] = None,
        stochastic_mask_probabilities: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        if modalities is None:
            modalities = ["ECG", "PPG", "THX", "ABD"]
        if stochastic_mask_probabilities is None:
            stochastic_mask_probabilities = {
                "ECG": 0.5,
                "PPG": 0.1,
                "THX": 0.7,
                "ABD": 0.7,
            }
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=FEATURE_DIM,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dimension,
            dropout=DROPOUT_RATE,
            activation=ACTIVATION_FUNCTION,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_transformer_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, FEATURE_DIM))
        self.stochastic_mask_probabilities = stochastic_mask_probabilities
        self.modalities = modalities

    def _build_attention_mask(
        self, batch: int, T: int, availability_mask: Tensor
    ) -> Tensor:
        """Builds an attention mask for the transformer

        Builds a stochastic mask first, then joins it with `availability_mask`.
        Ensures that the CLS token remains unmasked, and that
        at least one modality is available for the transformer.

        Args:
            batch: The batch size (number of sequences of the input batch)
            T: The total number of sleep epochs
            availability_mask: Mask indicating which modalities are available

        Returns:
            Tensor: The complete mask to pass to the transformer
        """

        device = self.cls_token.device

        probs = torch.tensor(
            [self.stochastic_mask_probabilities[m] for m in self.modalities]
        )
        probs = probs.unsqueeze(0).expand(batch, -1)
        stochastic_mask = torch.bernoulli(probs).bool()

        # Combine availability and stochastics masks
        complete_mask = availability_mask | stochastic_mask

        # Guarantee at least one modality is visible per recording
        all_masked = complete_mask.all(dim=1)
        if all_masked.any():  # if all modalities get masked
            # unmask the first available modality (index 0)
            complete_mask[all_masked, 0] = False

        # Expand across T epochs and fold into batch dimension
        complete_mask = (
            complete_mask.unsqueeze(1).expand(-1, T, -1).reshape(batch * T, -1)
        )

        # Prepend False for CLS — never masked
        cls_mask = torch.zeros(batch * T, 1, dtype=torch.bool, device=device)

        mask = torch.cat([cls_mask, complete_mask], dim=1)
        return mask

    def forward(self, x: Tensor, availability_mask: Tensor) -> Tensor:
        """Fuse modalities into one unified representation of sleep epoch sequences

        Args:
            x: Stacked modality encodings per epoch;
                shape = (batch_size, T, num_modalities, feature_dim)
            availability_mask: Mask indicating which modalities are available

        Returns:
            Tensor: A unified feature sequence; shape = (batch_size, T, feature_dim)
        """
        batch_size = x.shape[0]
        T = x.shape[1]

        x = x.reshape(batch_size * T, -1, FEATURE_DIM)
        cls_tokens = self.cls_token.expand(batch_size * T, 1, FEATURE_DIM)
        x = torch.cat(
            [cls_tokens, x], dim=1
        )  # (batch*T, num_modalities + 1, feature_dim)

        # Create a per-recording mask
        mask = (
            self._build_attention_mask(batch_size, T, availability_mask)
            if self.training
            else None
        )  # (batch*T, num_modalities + 1, feature_dim)

        out = self.transformer(x, src_key_padding_mask=mask)

        # Slice CLS position — the unified summary for each epoch
        z = out[:, 0, :]  # (batch*T, feature_dim)
        z = z.reshape(batch_size, T, FEATURE_DIM)

        return z


class SequenceMixer(nn.Module):
    """Architecture for the Sequence Mixer as described in Section 3.3

    Captures temporal dependencies among encoded sequences using a stack
    of dilated convolutional blocks with increasing dilation factors.

    Args:
        dilated_blocks: Number of dilated blocks to use
        kernel_size: kernel_size for the dilated blocks
    """

    def __init__(self, dilated_blocks: int = 2, kernel_size: int = 7) -> None:
        super().__init__()
        dilations = [1, 2, 4, 8, 16, 32]

        blocks = [
            DilatedConvBlock(d, kernel_size)
            for _ in range(dilated_blocks)
            for d in dilations
        ]

        self.dilated_cnns = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        """Processes a sequence of feature vectors through the mixer

        Applies 1D dilated convolutions over the time dimension to capture
        long‑range temporal dependencies, while preserving the sequence
        length and feature dimension.

        Args:
            x: The unified sequence of feature vectors; shape = (batch_size, T, feature_dim)

        Returns:
            Tensor: The transformed sequence; shape = (batch_size, T, feature_dim)
        """

        out = x.transpose(1, 2)
        out = self.dilated_cnns(out)
        out = out.transpose(1, 2)
        return out


class DilatedConvBlock(nn.Module):
    """Dilated Convolution Block

    Applies a 1D dilated convolution followed by layer normalization,
    an activation function, and dropout, with a residual connection
    that adds the input back to the output. Used to capture long‑range
    temporal dependencies while preserving the sequence length.

    Args:
        dilation: Dilation factor for convolution
        kernel_size: Size of the convolutional kernel
    """

    def __init__(self, dilation: int, kernel_size: int = 7) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(
            FEATURE_DIM, FEATURE_DIM, kernel_size, dilation=dilation, padding=padding
        )
        self.norm = nn.LayerNorm(FEATURE_DIM)
        self.activation = ACTIVATION_FUNCTION
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the dilated convolution block

        Args:
            x: Input sequence of feature vectors; shape = (batch_size, feature_dim, T)

        Returns:
            Tensor: Output sequence of feature vectors; shape = (batch_size, feature_dim, T)
        """
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = self.norm(out)
        out = out.transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        out += x
        return out


class Wav2Sleep(BaseModel):
    """The wav2sleep model

    Classifies sleep stage sequences from sets of time-series biosignals.
    A trained model can applied to any subset of the signal modalities seen during training.

    The model consists of
        Signal Encoders for each modality
        An Epoch Mixer to fuse cross-modal information for each sleep epoch
        A Sequence Mixer to mix temporal information

    Args:
        dataset: The dataset used to train the model
        modalities: The list of modalities to train the model with

    Example:
        >>> from pyhealth.datasets import Wav2SleepDataset
        >>> from pyhealth.tasks import Wav2SleepStaging
        >>> wav2sleep_dataset = Wav2SleepDataset(root = "path/to/root")
        >>> task = Wav2SleepStaging()
        >>> samples = wav2sleep_dataset.set_task(task)
        >>> wav2sleep_model = Wav2Sleep(samples) # train with all modalities (default)
        >>> train_loader = get_dataloader(samples, batch_size=2, shuffle=False)
        >>> data_batch = next(iter(train_loader))
        >>> output = wav2sleep_model(**data_batch)
        >>> print(output)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        modalities: Optional[List[str]] = None,
        stochastic_mask_probabilities: Optional[Dict[str, float]] = None,
    ) -> None:
        if modalities is None:
            modalities = ["ECG", "PPG", "THX", "ABD"]

        # Validate stochastic_mask_probabilities: all modalities present, values are numbers in [0, 1]
        if stochastic_mask_probabilities is not None:
            for modality in modalities:
                if modality not in stochastic_mask_probabilities:
                    raise ValueError(
                        f"Missing mask probability for modality '{modality}'"
                    )
                probability = stochastic_mask_probabilities[modality]
                if probability is None:
                    raise ValueError("The mask probability must not be None")
                if (probability < 0) or (probability > 1):
                    raise ValueError(
                        f"The mask probability must be in [0, 1], got {probability}"
                    )

        super(Wav2Sleep, self).__init__(dataset=dataset)

        # signal_type : resample_rate
        self.all_modalities = {"ECG": 1024, "PPG": 1024, "THX": 256, "ABD": 256}

        self.selected_modalities = {
            k: self.all_modalities[k] for k in modalities if k in self.all_modalities
        }

        # Initialize Signal Encoders for each modality
        self.signal_encoders = nn.ModuleDict(
            {
                signal_type: SignalEncoder(signal_sample_rate=resample_rate)
                for signal_type, resample_rate in self.selected_modalities.items()
            }
        )
        # Initialize Epoch Mixer to learn attention between the modalities at each epoch
        self.epoch_mixer = EpochMixer(
            modalities=modalities,
            stochastic_mask_probabilities=stochastic_mask_probabilities,
        )
        # Initialize Sequence Mixer to mix temporal information and output predicted sleep stages
        self.sequence_mixer = SequenceMixer()

        # 4 classes total (Wake, Light Sleep, Deep Sleep, REM)
        self.num_classes = 4
        self.classifier = nn.Linear(
            in_features=FEATURE_DIM, out_features=self.num_classes
        )

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        """Forward pass for the Wav2Sleep model

        Transforms a set of raw polysomnography biosignals into sleep-stage predictions.

        Args:
            **kwargs: Batch dictionary containing:
                - signals (Dict[str, Tensor]):
                    Input modality tensors; shape = (batch, signal_length)
                - availability_mask (Tensor):
                    Indicates unavailable modalities; shape = (batch, num_modalities)
                - stages (Tensor, optional): Sleep stage labels; shape = (batch, T)

        Returns:
            Dict[str, Tensor]:
                y_prob: logits; shape = (batch, T, num_classes)
                y_hat: sleep stage predictions; shape = (batch, T)
                loss: cross-entropy loss (if ground truth labels `stages` was provided)
        """

        signals = {m: kwargs[m] for m in self.selected_modalities.keys()}
        availability_mask = kwargs["availability_mask"]
        stages = kwargs.get("stages", None)

        encoded_signals = torch.stack(
            [
                self.signal_encoders[m](signals[m])
                for m in self.selected_modalities.keys()
            ],
            dim=2,
        )  # → (batch, T, num_modalities, feature_dim)

        selected_indices = [
            list(self.all_modalities.keys()).index(m)
            for m in self.selected_modalities.keys()
        ]
        filtered_mask = availability_mask[:, selected_indices].bool()

        mixed = self.epoch_mixer(encoded_signals, filtered_mask)

        logits = self.sequence_mixer(mixed)

        y_T = self.classifier(logits)

        y_prob = torch.softmax(y_T, dim=-1)
        output = {"y_prob": y_prob, "y_hat": y_prob.argmax(dim=-1)}
        if stages is not None:
            loss = F.cross_entropy(
                y_T.view(-1, self.num_classes), stages.view(-1).long(), ignore_index=-1
            )
            output["loss"] = loss
        return output


if __name__ == "__main__":
    batch_size = 2
    T = 1200

    from pyhealth.datasets import Wav2SleepDataset, get_dataloader
    from pyhealth.tasks import Wav2SleepStaging

    wav2sleep_dataset = Wav2SleepDataset(root="../../../full_sample_PSG/")
    task = Wav2SleepStaging()
    samples = wav2sleep_dataset.set_task(task)
    wav2sleep_model = Wav2Sleep(samples)

    train_loader = get_dataloader(samples, batch_size=2, shuffle=False)
    data_batch = next(iter(train_loader))

    output = wav2sleep_model(**data_batch)
    print(output)
