# Paper: wav2sleep: A Unified Multi-Modal Approach to Sleep Stage Classification
#         from Physiological Signals
# Paper Link: https://arxiv.org/abs/2411.04644
# Description: wav2sleep implementation for PyHealth
#
# This module implements the wav2sleep model for multimodal sleep stage classification
# from physiological signals. The model is designed to handle variable numbers of input
# signal modalities (EEG, ECG, PPG, respiratory signals, etc.) and gracefully degrades
# when signals are missing at inference time.
#
# Architecture Overview:
# =====================
# The model consists of three main components:
#
# 1. Signal Encoders (CNN-based)
#    - One ResidualBlock1D-based SignalEncoder per modality
#    - Compresses raw signal into fixed-size feature vectors
#    - Automatically scales depth based on signal length
#
# 2. Epoch Mixer (Transformer-based)
#    - Multi-head self-attention mechanism to fuse modalities
#    - Learnable CLS token aggregates information across modalities
#    - Pre-norm architecture for training stability
#
# 3. Classification Head
#    - Linear layer mapping CLS embedding to class logits
#    - Supports variable number of sleep stage classes
#
# Key Features:
# =============
# - Multi-modal fusion: Can work with any subset of available signals
# - Graceful degradation: Model works with missing modalities at test time
# - Flexible: Supports variable numbers of channels per signal
# - Dataset support: Includes helpers for SHHS and CFS datasets
# - Standards-compliant: Follows PyHealth BaseModel interface
#
# Usage Examples:
# ===============
# Basic usage with custom data:
#     >>> from pyhealth.datasets import create_sample_dataset
#     >>> from pyhealth.models import Wav2Sleep
#     >>> samples = [...]  # Your samples with signal modalities
#     >>> dataset = create_sample_dataset(
#     ...     samples=samples,
#     ...     input_schema={"ecg": "tensor", "ppg": "tensor"},
#     ...     output_schema={"label": "multiclass"},
#     ... )
#     >>> model = Wav2Sleep(dataset=dataset)
#
# With SHHS dataset:
#     >>> from pyhealth.models.wav2sleep import load_shhs_samples
#     >>> samples = load_shhs_samples("/path/to/shhs")
#     >>> dataset = create_sample_dataset(
#     ...     samples=samples,
#     ...     input_schema={"ecg": "tensor", "abd": "tensor", "thx": "tensor"},
#     ...     output_schema={"label": "multiclass"},
#     ... )
#     >>> model = Wav2Sleep(dataset=dataset)
#
# With CFS dataset:
#     >>> from pyhealth.datasets import CFSDataset
#     >>> from pyhealth.tasks import SleepStagingCFS
#     >>> from pyhealth.models.wav2sleep import load_cfs_samples
#     >>> cfs_dataset = CFSDataset(root="/path/to/cfs")
#     >>> cfs_dataset.set_task(SleepStagingCFS())
#     >>> samples = load_cfs_samples(cfs_dataset)
#     >>> dataset = create_sample_dataset(
#     ...     samples=samples,
#     ...     input_schema={"eeg": "tensor", "ecg": "tensor"},
#     ...     output_schema={"label": "multiclass"},
#     ... )
#     >>> model = Wav2Sleep(dataset=dataset)
#
# References:
# ===========
# [1] Carter, J.F., & Tarassenko, L. (2024). "wav2sleep: A Unified Multi-Modal
#     Approach to Sleep Stage Classification from Physiological Signals."
#     arXiv:2411.04644

import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import numpy as np
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

    # ------------------------------------------------------------------
    # Evaluation Methods
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataloader,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Evaluate the model on a dataloader using wav2sleep metrics.

        Computes confusion matrix, accuracy, Cohen's Kappa, and per-class
        metrics on the provided dataset.

        Args:
            dataloader: PyHealth DataLoader with batches of samples.
                Each batch should contain modality tensors and labels.
            class_names: Optional list of class names for the report.
                Default: ["Wake", "N1", "N2", "N3", "REM", ...] up to num_classes.

        Returns:
            Dictionary with evaluation metrics:
            - "confusion_matrix": Confusion matrix
            - "accuracy": Overall accuracy [0, 1]
            - "kappa": Cohen's Kappa [0, 1]
            - "precision": Per-class precision
            - "recall": Per-class recall
            - "f1": Per-class F1-score
            - "macro_precision", "macro_recall", "macro_f1": Macro-averaged metrics
            - "weighted_f1": Weighted F1-score
            - "class_names": List of class names
            - "class_support": Sample count per class

        Example:
            >>> model = Wav2Sleep(dataset=dataset)
            >>> val_loader = get_dataloader(dataset, batch_size=32)
            >>> results = model.evaluate(val_loader)
            >>> print(f"Accuracy: {results['accuracy']:.2%}")
            >>> print(f"Cohen's Kappa: {results['kappa']:.3f}")
        """
        from pyhealth.metrics import evaluate_model as evaluate_model_fn

        self.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                output = self(**batch)
                preds = output["y_prob"].argmax(dim=-1)
                labels = output["y_true"]

                all_predictions.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        num_classes = self.get_output_size()

        if class_names is None:
            default_names = ["Wake", "N1", "N2", "N3", "REM"]
            class_names = default_names[:num_classes] + [
                f"Class {i}" for i in range(len(default_names), num_classes)
            ]

        return evaluate_model_fn(
            predictions=predictions,
            labels=labels,
            num_classes=num_classes,
            class_names=class_names,
        )

    def evaluate_modalities(
        self,
        dataloader,
        modality_subsets: Optional[Dict[str, List[str]]] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, any]]:
        """Evaluate model on different signal modality subsets.

        Demonstrates the key wav2sleep feature: the same trained model works
        with any available subset of modalities without retraining.

        Args:
            dataloader: PyHealth DataLoader with batches of samples.
            modality_subsets: Dictionary mapping subset names to lists of
                modality keys to use. Default subsets based on available modalities:
                - "All": all modalities
                - "Single modalities": one at a time
                - "Common combinations": practical subsets
            class_names: Optional list of class names.

        Returns:
            Dictionary mapping subset name to evaluation results dict.

        Example:
            >>> results_by_subset = model.evaluate_modalities(val_loader)
            >>> for subset_name, metrics in results_by_subset.items():
            ...     print(f"{subset_name}: Kappa={metrics['kappa']:.3f}")
        """
        if modality_subsets is None:
            # Auto-generate default subsets
            modality_subsets = {}
            modality_subsets["All"] = self.feature_keys

            # Single modalities
            for key in self.feature_keys:
                modality_subsets[f"Single: {key}"] = [key]

        results = {}

        for subset_name, subset_keys in modality_subsets.items():
            # Filter batch to only include requested modalities
            subset_results = self._evaluate_subset(
                dataloader, subset_keys, class_names
            )
            results[subset_name] = subset_results

        return results

    def _evaluate_subset(
        self,
        dataloader,
        modality_keys: List[str],
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """Helper: evaluate on a subset of modalities."""
        from pyhealth.metrics import evaluate_model as evaluate_model_fn

        self.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                # Filter batch to only available modalities
                filtered_batch = {
                    k: v for k, v in batch.items()
                    if k in modality_keys or k == self.label_key
                }

                # Skip if none of the requested modalities are in this batch
                if not any(k in filtered_batch for k in modality_keys):
                    continue

                output = self(**filtered_batch)
                preds = output["y_prob"].argmax(dim=-1)
                labels = output["y_true"]

                all_predictions.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        if not all_predictions:
            # No valid samples for this subset
            return {
                "confusion_matrix": np.zeros((self.get_output_size(),) * 2),
                "accuracy": 0.0,
                "kappa": 0.0,
                "precision": np.zeros(self.get_output_size()),
                "recall": np.zeros(self.get_output_size()),
                "f1": np.zeros(self.get_output_size()),
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "class_names": class_names or [f"Class {i}" for i in range(self.get_output_size())],
                "class_support": np.zeros(self.get_output_size()),
            }

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        num_classes = self.get_output_size()

        if class_names is None:
            default_names = ["Wake", "N1", "N2", "N3", "REM"]
            class_names = default_names[:num_classes] + [
                f"Class {i}" for i in range(len(default_names), num_classes)
            ]

        return evaluate_model_fn(
            predictions=predictions,
            labels=labels,
            num_classes=num_classes,
            class_names=class_names,
        )


# ---------------------------------------------------------------------------
# SHHS preprocessing helper
# ---------------------------------------------------------------------------

def load_shhs_samples(
    shhs_root: str,
    epoch_seconds: int = 30,
    ecg_samples_per_epoch: int = 1024,
    resp_samples_per_epoch: int = 256,
    max_recordings: Optional[int] = None,
    label_map: Optional[Dict[int, int]] = None,
) -> List[Dict]:
    """Load SHHS recordings and return samples ready for :class:`Wav2Sleep`.

    Reads SHHS polysomnography EDF files and paired XML annotation files,
    extracts ECG, abdominal (ABD), and thoracic (THX) respiratory signals,
    resamples each to the target length per epoch, and returns one sample
    dict per 30-second sleep epoch.

    The returned list is directly compatible with
    :func:`pyhealth.datasets.create_sample_dataset`::

        input_schema  = {"ecg": "tensor", "abd": "tensor", "thx": "tensor"}
        output_schema = {"label": "multiclass"}

    This mirrors the cardiorespiratory signal setup used in the wav2sleep paper
    (Carter & Tarassenko, arXiv:2411.04644).

    Expected directory layout::

        shhs_root/
            edfs/
                shhs1/   *.edf
                shhs2/   *.edf                        (optional)
            annotations-events-profusion/
                shhs1/   *-profusion.xml
                shhs2/   *-profusion.xml              (optional)

    Args:
        shhs_root: Path to the SHHS polysomnography root directory.
        epoch_seconds: Duration of each sleep epoch in seconds. Default is 30.
        ecg_samples_per_epoch: Target ECG samples per epoch after resampling.
            Default is 1024 (~34 Hz, matching the wav2sleep paper).
        resp_samples_per_epoch: Target ABD/THX samples per epoch after
            resampling. Default is 256 (~8 Hz).
        max_recordings: Process at most this many EDF files. Default is None
            (process all). Useful for quick experiments.
        label_map: Maps raw SHHS stage integers to output labels. Default is
            5-class AASM::

                {0: 0,  # Wake
                 1: 1,  # N1
                 2: 2,  # N2
                 3: 3,  # N3
                 4: 3,  # N3 (legacy duplicate code)
                 5: 4}  # REM

            For the 4-class setup from the wav2sleep paper (N1+N2 merged as
            "Light"), pass ``{0:0, 1:1, 2:1, 3:2, 4:2, 5:3}``.

    Returns:
        List of sample dicts, each containing:

        - ``patient_id`` (str): Recording ID, e.g. ``"shhs1-200001"``.
        - ``visit_id`` (str): Epoch ID, e.g. ``"shhs1-200001-42"``.
        - ``ecg`` (np.ndarray): Shape ``(1, ecg_samples_per_epoch)``, float32.
        - ``abd`` (np.ndarray): Shape ``(1, resp_samples_per_epoch)``, float32.
        - ``thx`` (np.ndarray): Shape ``(1, resp_samples_per_epoch)``, float32.
        - ``label`` (int): Mapped sleep stage label.

    Raises:
        FileNotFoundError: If no EDF+XML pairs are found under ``shhs_root``.
        RuntimeError: If a recording is missing one of the required channels.

    Examples:
        >>> from pyhealth.models.wav2sleep import load_shhs_samples
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = load_shhs_samples(
        ...     "/data/shhs/polysomnography",
        ...     label_map={0:0, 1:1, 2:1, 3:2, 4:2, 5:3},  # 4-class
        ...     max_recordings=10,
        ... )
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"ecg": "tensor", "abd": "tensor", "thx": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="shhs_wav2sleep",
        ... )
        >>> model = Wav2Sleep(dataset=dataset)
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "mne is required for SHHS loading: pip install mne"
        ) from exc

    try:
        from scipy.signal import resample as _scipy_resample
        def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
            return _scipy_resample(x, n_out)
    except ImportError:
        def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
            idx = np.linspace(0, len(x) - 1, n_out)
            return np.interp(idx, np.arange(len(x)), x)

    if label_map is None:
        label_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}

    ECG_NAMES = ["ECG", "EKG", "ecg", "ekg"]
    ABD_NAMES = ["ABDO RES", "ABDO", "ABD", "abdo res", "abdo", "abd"]
    THX_NAMES = ["THOR RES", "THOR", "THX", "thor res", "thor", "thx"]

    def _find_ch(ch_names: List[str], candidates: List[str]) -> Optional[str]:
        for name in candidates:
            if name in ch_names:
                return name
        return None

    def _collect_paths(root: str) -> List[tuple]:
        paths = []
        for visit in ("shhs1", "shhs2"):
            edf_dir = os.path.join(root, "edfs", visit)
            xml_dir = os.path.join(root, "annotations-events-profusion", visit)
            if not os.path.isdir(edf_dir):
                continue
            for fname in sorted(os.listdir(edf_dir)):
                if not fname.endswith(".edf"):
                    continue
                pid = fname.split(".")[0]
                xml_path = os.path.join(xml_dir, f"{pid}-profusion.xml")
                if not os.path.isfile(xml_path):
                    continue
                paths.append((os.path.join(edf_dir, fname), xml_path, pid))
        return paths

    edf_paths = _collect_paths(shhs_root)
    if not edf_paths:
        raise FileNotFoundError(
            f"No EDF+XML pairs found under '{shhs_root}'. "
            "Check that edfs/shhs1/ and annotations-events-profusion/shhs1/ exist."
        )
    if max_recordings is not None:
        edf_paths = edf_paths[:max_recordings]

    samples: List[Dict] = []

    for edf_path, xml_path, pid in edf_paths:
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception:
            continue

        ch_names = raw.ch_names
        ecg_ch = _find_ch(ch_names, ECG_NAMES)
        abd_ch = _find_ch(ch_names, ABD_NAMES)
        thx_ch = _find_ch(ch_names, THX_NAMES)

        missing = [n for n, ch in [("ECG", ecg_ch), ("ABD", abd_ch), ("THX", thx_ch)] if ch is None]
        if missing:
            raise RuntimeError(
                f"{pid}: channels {missing} not found. Available: {ch_names}"
            )

        sfreq = raw.info["sfreq"]
        epoch_len = int(sfreq * epoch_seconds)

        ecg_sig, _ = raw[ecg_ch]  # (1, n_times)
        abd_sig, _ = raw[abd_ch]
        thx_sig, _ = raw[thx_ch]

        try:
            tree = ET.parse(xml_path)
            stages = [
                int(s.text)
                for s in tree.getroot().find("SleepStages").findall("SleepStage")
            ]
        except Exception:
            continue

        n_epochs = min(
            ecg_sig.shape[1] // epoch_len,
            abd_sig.shape[1] // epoch_len,
            thx_sig.shape[1] // epoch_len,
            len(stages),
        )

        for i in range(n_epochs):
            if stages[i] not in label_map:
                continue

            ecg_e = _resample(ecg_sig[0, i * epoch_len:(i + 1) * epoch_len], ecg_samples_per_epoch).astype(np.float32)
            abd_e = _resample(abd_sig[0, i * epoch_len:(i + 1) * epoch_len], resp_samples_per_epoch).astype(np.float32)
            thx_e = _resample(thx_sig[0, i * epoch_len:(i + 1) * epoch_len], resp_samples_per_epoch).astype(np.float32)

            for arr in (ecg_e, abd_e, thx_e):
                std = arr.std()
                if std > 0:
                    arr -= arr.mean()
                    arr /= std

            samples.append({
                "patient_id": pid,
                "visit_id": f"{pid}-{i}",
                "ecg": ecg_e[np.newaxis, :],
                "abd": abd_e[np.newaxis, :],
                "thx": thx_e[np.newaxis, :],
                "label": label_map[stages[i]],
            })

    return samples


# ---------------------------------------------------------------------------
# CFS preprocessing helper
# ---------------------------------------------------------------------------

def load_cfs_samples(
    cfs_dataset,
    channel_mapping: Optional[Dict[str, int]] = None,
    max_recordings: Optional[int] = None,
) -> List[Dict]:
    """Convert CFS dataset samples to Wav2Sleep format with separate signal modalities.

    This helper converts CFS sleep staging task output (which provides multi-channel
    signals in a single tensor) into the format expected by Wav2Sleep, which requires
    separate feature keys for each signal modality (e.g., "eeg", "eog", "ecg").

    The CFS dataset task returns samples with a "signal" key containing a (n_channels, n_samples)
    tensor where channels are typically: EEG, EOG-L, EOG-R, EMG-Chin, ECG.

    Args:
        cfs_dataset: A PyHealth dataset with CFS sleep staging samples.
            Expected to have samples with "signal" tensor and "label" keys.
        channel_mapping: Dictionary mapping modality names to channel indices in the signal tensor.
            Default assumes the standard CFS channel order:
            ``{"eeg": 0, "eog_left": 1, "eog_right": 2, "emg": 3, "ecg": 4}``
            You can override to use a subset, e.g., ``{"eeg": 0, "ecg": 4}``.
        max_recordings: Process at most this many samples. Default is None (process all).

    Returns:
        List of sample dicts, each containing:

        - ``patient_id`` (str): Patient ID from the CFS dataset.
        - ``visit_id`` (str): Unique sample identifier.
        - ``<modality>`` (np.ndarray): Signal modality tensor of shape ``(1, length)``,
          e.g., ``eeg``, ``eog_left``, ``ecg``, etc., depending on ``channel_mapping``.
        - ``label`` (int): Sleep stage label (0=Wake, 1=N1, 2=N2, 3=N3, 4=REM).

    Examples:
        >>> from pyhealth.datasets import CFSDataset
        >>> from pyhealth.tasks import SleepStagingCFS
        >>> from pyhealth.models.wav2sleep import load_cfs_samples
        >>> from pyhealth.datasets import create_sample_dataset
        >>>
        >>> # Load CFS dataset with sleep staging task
        >>> cfs_dataset = CFSDataset(root="/path/to/cfs")
        >>> task = SleepStagingCFS()
        >>> cfs_dataset.set_task(task)
        >>>
        >>> # Convert to Wav2Sleep format
        >>> samples = load_cfs_samples(
        ...     cfs_dataset,
        ...     channel_mapping={"eeg": 0, "ecg": 4},
        ...     max_recordings=100,
        ... )
        >>>
        >>> # Create PyHealth dataset
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"eeg": "tensor", "ecg": "tensor"},
        ...     output_schema={"label": "multiclass"},
        ...     dataset_name="cfs_wav2sleep",
        ... )
        >>>
        >>> # Train Wav2Sleep model
        >>> model = Wav2Sleep(dataset=dataset)
    """
    if channel_mapping is None:
        channel_mapping = {
            "eeg": 0,           # EEG channel
            "eog_left": 1,      # Left EOG channel
            "eog_right": 2,     # Right EOG channel
            "emg": 3,           # Chin EMG channel
            "ecg": 4,           # ECG channel
        }

    samples: List[Dict] = []
    count = 0

    for cfs_sample in cfs_dataset:
        if max_recordings is not None and count >= max_recordings:
            break

        try:
            # Extract signal and label
            if "signal" not in cfs_sample or "label" not in cfs_sample:
                continue

            full_signal = cfs_sample["signal"]  # Shape: (n_channels, n_samples)
            label = int(cfs_sample["label"])

            # Get patient information
            patient_id = cfs_sample.get("patient_id", f"cfs_{count}")
            study_id = cfs_sample.get("study_id", f"{patient_id}_0")

            # Extract individual modalities based on channel mapping
            modalities = {}
            for modality_name, channel_idx in channel_mapping.items():
                if channel_idx < full_signal.shape[0]:
                    # Extract single channel and add channel dimension: (length,) -> (1, length)
                    channel_data = full_signal[channel_idx, :].astype(np.float32)
                    modalities[modality_name] = channel_data[np.newaxis, :]  # (1, length)

            if not modalities:
                continue

            # Create sample with separated modalities
            sample = {
                "patient_id": str(patient_id),
                "visit_id": str(study_id),
                "label": label,
                **modalities,
            }
            samples.append(sample)
            count += 1

        except Exception:
            continue

    return samples
