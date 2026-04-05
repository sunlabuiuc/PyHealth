# Paper: wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification
#         from Physiological Signals
# Paper Link: https://arxiv.org/abs/2411.04644
# Description: wav2sleep implementation for PyHealth

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class ResidualBlock1D(nn.Module):
    """1D residual block for physiological signal encoding.

    Each block applies three Conv1d layers followed by MaxPool1d downsampling
    (factor of 2). A residual connection projects the input to match the output
    channels using a 1x1 convolution with average pooling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size for the three convolutional layers. Default is 3.

    Example:
        >>> block = ResidualBlock1D(1, 16)
        >>> x = torch.randn(4, 1, 256)
        >>> out = block(x)
        >>> out.shape
        torch.Size([4, 16, 128])
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=padding, bias=False
        )
        self.bn3 = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2)
        self.activation = nn.GELU()

        # Residual shortcut: match channels + downsample in one step
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, length).

        Returns:
            Output tensor of shape (batch, out_channels, length // 2).
        """
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool(self.bn3(self.conv3(out)))
        return self.activation(out + residual)


class SignalEncoder(nn.Module):
    """CNN encoder for a single 1D physiological signal modality.

    Stacks multiple :class:`ResidualBlock1D` layers following the SleepPPG-Net
    design used in the wav2sleep paper. After the convolutional stack, global
    average pooling collapses the time axis, and a linear projection maps to
    the target feature dimension.

    Args:
        in_channels: Number of input signal channels (1 for univariate signals).
        feature_dim: Output feature dimension. Default is 128.
        channel_schedule: Ordered list of output channels for each residual
            block. Determines model depth and width. Default is
            ``[16, 32, 64, 64, 128, 128]`` (suitable for shorter signals such
            as ABD/THX at ~256 samples); use
            ``[16, 16, 32, 32, 64, 64, 128, 128]`` for longer signals such as
            ECG/PPG at ~1024 samples.
        kernel_size: Kernel size for all convolutional layers. Default is 3.

    Example:
        >>> enc = SignalEncoder(in_channels=1, feature_dim=128)
        >>> x = torch.randn(4, 1, 256)
        >>> z = enc(x)
        >>> z.shape
        torch.Size([4, 128])
    """

    def __init__(
        self,
        in_channels: int,
        feature_dim: int = 128,
        channel_schedule: Optional[List[int]] = None,
        kernel_size: int = 3,
    ):
        super().__init__()
        if channel_schedule is None:
            channel_schedule = [16, 32, 64, 64, 128, 128]

        layers: List[nn.Module] = []
        ch = in_channels
        for out_ch in channel_schedule:
            layers.append(ResidualBlock1D(ch, out_ch, kernel_size))
            ch = out_ch

        self.conv_layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(ch, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a raw 1D signal into a fixed-length feature vector.

        Args:
            x: Input tensor. Accepts shape ``(batch, length)`` (univariate) or
                ``(batch, channels, length)`` (multivariate).

        Returns:
            Feature tensor of shape ``(batch, feature_dim)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)
        out = self.conv_layers(x)           # (batch, ch, reduced_length)
        out = self.pool(out).squeeze(-1)    # (batch, ch)
        return self.proj(out)               # (batch, feature_dim)


class Wav2Sleep(BaseModel):
    """wav2sleep: A unified multi-modal model for sleep stage classification.

    Paper: Jonathan F. Carter and Lionel Tarassenko. "wav2sleep: A Unified
    Multi-Modal Approach to Sleep Stage Classification from Physiological
    Signals." arXiv:2411.04644, 2024.
    https://arxiv.org/abs/2411.04644

    The model classifies 30-second polysomnography epochs into sleep stages
    (e.g. Wake, N1, N2, N3, REM) from one or more cardiorespiratory signals.
    Its key design principle is that all input modalities are optional: the
    same trained model handles any subset of signals available at inference
    time without retraining or architectural changes.

    **Architecture overview:**

    1. **Signal Encoders** — one :class:`SignalEncoder` (CNN) per modality,
       each producing a ``feature_dim``-dimensional vector from raw samples.
    2. **Epoch Mixer** — a Transformer encoder with a learnable CLS token that
       fuses modality features via self-attention, yielding a single unified
       epoch embedding.
    3. **Output head** — a linear layer mapping the CLS embedding to class
       logits.

    **Usage with PyHealth:**

    Map each signal modality to a separate feature key in the dataset's
    ``input_schema`` with the ``"tensor"`` processor type.  Any modality whose
    key is present in a batch will be encoded; absent keys are silently skipped
    so the model degrades gracefully to fewer modalities.

    Note:
        For best results, use ``output_schema={"label": "multiclass"}`` with
        integer sleep-stage labels (e.g. 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).
        The model supports any number of classes determined automatically from
        the dataset.

    Args:
        dataset: The :class:`~pyhealth.datasets.SampleDataset` used for
            training.  Each feature key in ``input_schema`` must use the
            ``"tensor"`` processor, and ``output_schema`` must have exactly one
            ``"multiclass"`` label key.
        feature_dim: Internal feature dimension shared across all encoders and
            the transformer. Default is 128.
        n_transformer_layers: Number of layers in the Transformer encoder
            (epoch mixer). Default is 2.
        n_attention_heads: Number of self-attention heads. Must divide
            ``feature_dim`` evenly. Default is 8.
        transformer_ff_dim: Feed-forward hidden dimension inside the
            Transformer. Default is 512.
        dropout: Dropout rate applied inside the Transformer. Default is 0.1.
        kernel_size: Conv kernel size used in every :class:`SignalEncoder`.
            Default is 3.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "ecg": np.random.randn(1, 256).astype(np.float32),
        ...         "ppg": np.random.randn(1, 256).astype(np.float32),
        ...         "label": 0,
        ...     },
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "ecg": np.random.randn(1, 256).astype(np.float32),
        ...         "ppg": np.random.randn(1, 256).astype(np.float32),
        ...         "label": 1,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ecg": "tensor", "ppg": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="test",
        ... )
        >>> from pyhealth.models import Wav2Sleep
        >>> model = Wav2Sleep(dataset=dataset)
        >>> loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> batch = next(iter(loader))
        >>> ret = model(**batch)
        >>> print(sorted(ret.keys()))
        ['logit', 'loss', 'y_prob', 'y_true']
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_dim: int = 128,
        n_transformer_layers: int = 2,
        n_attention_heads: int = 8,
        transformer_ff_dim: int = 512,
        dropout: float = 0.1,
        kernel_size: int = 3,
    ):
        super().__init__(dataset=dataset)

        assert len(self.label_keys) == 1, (
            "Wav2Sleep requires exactly one label key in output_schema."
        )
        assert len(self.feature_keys) >= 1, (
            "Wav2Sleep requires at least one signal modality in input_schema."
        )

        self.feature_dim = feature_dim
        self.label_key = self.label_keys[0]

        # --- Signal Encoders (one per modality) ---
        self.encoders = nn.ModuleDict()
        for key in self.feature_keys:
            in_channels, length = self._infer_signal_shape(key)
            # Deeper encoder for longer signals (ECG/PPG ≥ 512 samples)
            if length >= 512:
                channel_schedule = [16, 16, 32, 32, 64, 64, 128, 128]
            else:
                channel_schedule = [16, 32, 64, 64, 128, 128]
            self.encoders[key] = SignalEncoder(
                in_channels=in_channels,
                feature_dim=feature_dim,
                channel_schedule=channel_schedule,
                kernel_size=kernel_size,
            )

        # --- Epoch Mixer (Transformer with CLS token) ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=n_attention_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm for training stability
        )
        self.epoch_mixer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_transformer_layers
        )

        # --- Output head ---
        output_size = self.get_output_size()
        self.fc = nn.Linear(feature_dim, output_size)

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _infer_signal_shape(self, feature_key: str):
        """Return ``(in_channels, length)`` for *feature_key* from the dataset.

        Args:
            feature_key: Name of the feature in ``input_schema``.

        Returns:
            Tuple ``(in_channels, length)``.

        Raises:
            ValueError: If the feature is not found or has unexpected dimensions.
        """
        for sample in self.dataset:
            if feature_key not in sample:
                continue
            sig = sample[feature_key]
            if sig.dim() == 1:
                return 1, sig.shape[0]
            if sig.dim() == 2:
                return sig.shape[0], sig.shape[1]
            raise ValueError(
                f"Feature '{feature_key}' must be 1-D or 2-D, got shape {sig.shape}."
            )
        raise ValueError(
            f"Cannot determine signal shape for '{feature_key}': "
            "no valid samples found in dataset."
        )

    def _init_weights(self) -> None:
        """Kaiming-normal initialisation for Conv/Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Classify a batch of sleep epochs from one or more signal modalities.

        The method encodes every modality present in *kwargs*, concatenates the
        resulting feature vectors with a CLS token, runs them through the epoch
        mixer Transformer, and passes the CLS output through the classification
        head.

        Args:
            **kwargs: Batch dictionary produced by a PyHealth
                :class:`~torch.utils.data.DataLoader`.  Expected keys:

                - One tensor per modality listed in ``self.feature_keys``
                  (shape ``(batch, channels, length)`` or
                  ``(batch, length)``).  Keys absent from the batch are
                  silently ignored, so the model works with any subset of
                  the training modalities.
                - The label key (``self.label_key``), integer tensor of
                  shape ``(batch,)``.
                - Optional ``"embed"`` (bool): if ``True``, the CLS embedding
                  is included in the returned dict under ``"embed"``.

        Returns:
            Dictionary with the following entries:

            - **loss** (*torch.Tensor*): Scalar cross-entropy loss.
            - **y_prob** (*torch.Tensor*): Softmax class probabilities,
              shape ``(batch, num_classes)``.
            - **y_true** (*torch.Tensor*): Ground-truth labels,
              shape ``(batch,)``.
            - **logit** (*torch.Tensor*): Raw logits,
              shape ``(batch, num_classes)``.
            - **embed** (*torch.Tensor*, optional): CLS token embeddings,
              shape ``(batch, feature_dim)``, only present when
              ``kwargs["embed"] is True``.

        Raises:
            ValueError: If none of the model's feature keys are found in
                *kwargs*.
        """
        batch_size: Optional[int] = None
        modality_tokens: List[torch.Tensor] = []

        for key in self.feature_keys:
            if key not in kwargs:
                continue
            x = kwargs[key].to(self.device)
            if batch_size is None:
                batch_size = x.shape[0]
            feat = self.encoders[key](x)           # (batch, feature_dim)
            modality_tokens.append(feat.unsqueeze(1))  # (batch, 1, feature_dim)

        if not modality_tokens:
            raise ValueError(
                f"None of the expected modalities {self.feature_keys} were found "
                "in the batch. Check your dataset's input_schema."
            )

        # Prepend CLS token → (batch, 1 + n_modalities, feature_dim)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls] + modality_tokens, dim=1)

        # Epoch mixer
        out = self.epoch_mixer(tokens)  # (batch, 1 + n_modalities, feature_dim)
        emb = out[:, 0, :]              # CLS output: (batch, feature_dim)

        logits = self.fc(emb)           # (batch, num_classes)
        y_prob = self.prepare_y_prob(logits)
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)

        results: Dict[str, torch.Tensor] = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = emb
        return results
