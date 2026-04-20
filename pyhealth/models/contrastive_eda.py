"""
Contrastive EDA Encoder model for PyHealth.

Implements the SimCLR-style contrastive pre-training framework from:
    Matton, K., Lewis, R., Guttag, J., & Picard, R. (2023).
    "Contrastive Learning of Electrodermal Activity Representations
    for Stress Detection." CHIL 2023.

Authors:
    Megan Saunders, Jennifer Miranda, Jesus Torres
    {meganas4, jm123, jesusst2}@illinois.edu
"""

import copy
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class GaussianNoise:
    """Adds Gaussian noise scaled to signal power."""
    def __init__(self, sigma_scale: float = 0.1):
        self.sigma_scale = sigma_scale

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sigma = np.mean(np.abs(x - np.mean(x))) * self.sigma_scale
        return x + np.random.normal(scale=sigma, size=len(x))


class TemporalCutout:
    """Zeros out a random contiguous segment of the signal."""
    def __init__(self, cutout_size: int = 10):
        self.cutout_size = cutout_size

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = copy.deepcopy(x)
        start = np.random.randint(0, max(1, len(x) - self.cutout_size))
        x[start:start + self.cutout_size] = 0.0
        return x


class ExtractPhasic:
    """Extracts the phasic (high-frequency) component of EDA."""
    def __init__(self, data_freq: int = 4, cutoff_hz: float = 0.05):
        self.data_freq = data_freq
        self.b, self.a = scipy.signal.butter(
            4, [cutoff_hz], btype="highpass", output="ba", fs=data_freq
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return scipy.signal.filtfilt(self.b, self.a, x)


class ExtractTonic:
    """Extracts the tonic (low-frequency) component of EDA."""
    def __init__(self, data_freq: int = 4, cutoff_hz: float = 0.05):
        self.data_freq = data_freq
        self.b, self.a = scipy.signal.butter(
            4, [cutoff_hz], btype="lowpass", output="ba", fs=data_freq
        )

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return scipy.signal.filtfilt(self.b, self.a, x)


class LooseSensorArtifact:
    """Simulates a loose sensor dropout artifact."""
    def __init__(self, width: int = 4, smooth_width: int = 2):
        self.width = width
        self.smooth_width = smooth_width

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = copy.deepcopy(x)
        artifact_width = self.width
        if len(x) <= artifact_width:
            return x
        artifact_start = np.random.randint(0, len(x) - artifact_width + 1)
        artifact_end = artifact_start + artifact_width - 1
        drop_start = artifact_start + self.smooth_width
        drop_end = artifact_end - self.smooth_width
        if drop_start >= drop_end:
            return x
        mean_amp = np.mean(x[drop_start:drop_end + 1])
        x[drop_start:drop_end + 1] -= mean_amp
        x[x < 0] = 0.0
        return x


class AmplitudeScaling:
    """Scales signal amplitude by a random constant factor."""
    def __init__(self, scale_min: float = 0.5, scale_max: float = 1.5):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_min, self.scale_max)
        return x * scale


# Augmentation registry: name -> class
AUGMENTATION_REGISTRY = {
    "gaussian_noise": GaussianNoise,
    "temporal_cutout": TemporalCutout,
    "extract_phasic": ExtractPhasic,
    "extract_tonic": ExtractTonic,
    "loose_sensor_artifact": LooseSensorArtifact,
    "amplitude_scaling": AmplitudeScaling,
}

# Preset augmentation groups for ablation
AUGMENTATION_GROUPS = {
    "full": [
        "gaussian_noise", "temporal_cutout", "amplitude_scaling",
        "extract_phasic", "extract_tonic", "loose_sensor_artifact",
    ],
    "generic_only": [
        "gaussian_noise", "temporal_cutout", "amplitude_scaling",
    ],
    "eda_specific_only": [
        "extract_phasic", "extract_tonic", "loose_sensor_artifact",
    ],
}


def apply_augmentation_pair(
    x: np.ndarray,
    augmentation_names: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Applies two independently sampled augmentations to produce a positive pair.

    Args:
        x: Raw EDA window as numpy array of shape (window_size,).
        augmentation_names: List of augmentation names to sample from.

    Returns:
        Tuple of two augmented views, each of shape (window_size,).
    """
    aug_classes = [AUGMENTATION_REGISTRY[n] for n in augmentation_names]

    def _apply_one(signal):
        aug_fn = np.random.choice(aug_classes)()
        out = aug_fn(signal)
        # ensure output length matches input (some filters may shift length)
        if len(out) != len(signal):
            out = out[:len(signal)]
        return out.astype(np.float32)

    return _apply_one(x.copy()), _apply_one(x.copy())


# ---------------------------------------------------------------------------
# NT-Xent / NCE Loss
# ---------------------------------------------------------------------------

class NCELoss(nn.Module):
    """Noise Contrastive Estimation loss for contrastive pre-training.

    Ported directly from the authors' loss/nt_xent.py implementation.

    Args:
        temperature: Softmax temperature. Lower = sharper distribution.

    Example::
        >>> loss_fn = NCELoss(temperature=0.1)
        >>> z1, z2 = torch.randn(8, 64), torch.randn(8, 64)
        >>> loss = loss_fn(z1, z2)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings_v1: torch.Tensor,
        embeddings_v2: torch.Tensor,
    ) -> torch.Tensor:
        """Computes symmetric NCE loss between two sets of embeddings.

        Args:
            embeddings_v1: View 1 embeddings, shape (N, D).
            embeddings_v2: View 2 embeddings, shape (N, D).

        Returns:
            Scalar loss tensor.
        """
        norm1 = embeddings_v1.norm(dim=1).unsqueeze(0)
        norm2 = embeddings_v2.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(embeddings_v1, embeddings_v2.t())
        norm_matrix = torch.mm(norm1.t(), norm2)
        sim_matrix = sim_matrix / (norm_matrix * self.temperature)
        sim_matrix_exp = torch.exp(sim_matrix)

        # positive pairs are on the diagonal
        pos_mask = torch.eye(
            len(embeddings_v1), dtype=torch.bool, device=embeddings_v1.device
        )

        row_sum = sim_matrix_exp.sum(dim=1)
        sim_row = sim_matrix_exp / row_sum.unsqueeze(1)
        view1_loss = -torch.mean(torch.log(sim_row[pos_mask]))

        col_sum = sim_matrix_exp.sum(dim=0)
        sim_col = sim_matrix_exp / col_sum.unsqueeze(0)
        view2_loss = -torch.mean(torch.log(sim_col[pos_mask]))

        return (view1_loss + view2_loss) / 2.0


# ---------------------------------------------------------------------------
# 1D CNN Encoder
# ---------------------------------------------------------------------------

class EDAEncoder(nn.Module):
    """Lightweight 1D CNN encoder for EDA windows.

    Architecture follows the authors' implementation: three convolutional
    blocks with batch normalization and ReLU, followed by global average
    pooling and a linear projection head for contrastive training.

    Args:
        window_size: Length of the input EDA window in samples.
        embed_dim: Dimension of the output embedding.
        proj_dim: Dimension of the contrastive projection head output.

    Example::
        >>> encoder = EDAEncoder(window_size=60)
        >>> x = torch.randn(8, 60)
        >>> z = encoder(x)          # projected, shape (8, proj_dim)
        >>> h = encoder.encode(x)   # raw embedding, shape (8, embed_dim)
    """

    def __init__(
        self,
        window_size: int = 60,
        embed_dim: int = 128,
        proj_dim: int = 64,
    ):
        super().__init__()
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3
            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

        # Global average pooling -> (N, embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection head for contrastive loss
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns encoder embeddings without projection head.

        Args:
            x: Input tensor of shape (N, window_size).

        Returns:
            Embeddings of shape (N, embed_dim).
        """
        x = x.unsqueeze(1)          # (N, 1, window_size)
        x = self.encoder(x)         # (N, embed_dim, T')
        x = self.pool(x).squeeze(2) # (N, embed_dim)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns projected embeddings for contrastive loss.

        Args:
            x: Input tensor of shape (N, window_size).

        Returns:
            Projected embeddings of shape (N, proj_dim).
        """
        h = self.encode(x)
        return self.projector(h)


# ---------------------------------------------------------------------------
# Full ContrastiveEDAModel
# ---------------------------------------------------------------------------

class ContrastiveEDAModel(BaseModel):
    """Contrastive EDA encoder model for PyHealth.

    Supports two modes:
    - **pretrain**: SimCLR-style contrastive pre-training with NT-Xent loss.
      Input EDA windows are augmented into positive pairs and the encoder
      is trained to bring them together in embedding space.
    - **finetune**: Supervised stress detection. The encoder is loaded from
      a pre-trained checkpoint, a linear classifier head is appended, and
      the full network (or encoder-frozen variant) is fine-tuned with
      cross-entropy loss.

    Args:
        window_size: EDA window length in samples.
        embed_dim: Encoder output dimension.
        proj_dim: Contrastive projection head dimension.
        num_classes: Number of output classes for finetune mode.
        augmentation_group: One of 'full', 'generic_only', 'eda_specific_only',
            or a custom list of augmentation names. Controls which augmentations
            are applied during pre-training. Defaults to 'full'.
        temperature: NT-Xent loss temperature.
        freeze_encoder: If True in finetune mode, encoder weights are frozen
            and only the classifier head is trained.

    Example::
        >>> model = ContrastiveEDAModel(window_size=60, num_classes=2)
        >>> # Pretrain
        >>> loss = model.pretrain_step(batch_eda)
        >>> # Finetune
        >>> model.set_finetune_mode(num_classes=2)
        >>> logits = model(batch_eda)
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset] = None,
        window_size: int = 60,
        embed_dim: int = 128,
        proj_dim: int = 64,
        num_classes: int = 2,
        augmentation_group: str = "full",
        temperature: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        """Initializes ContrastiveEDAModel.

        ``dataset`` is optional because this model supports a two-stage
        pretrain/finetune workflow that does not require a SampleDataset at
        construction time. Pass ``None`` (default) when instantiating for
        contrastive pre-training; provide a dataset only when integrating with
        the standard PyHealth SampleDataset pipeline.

        Args:
            dataset (Optional[SampleDataset]): PyHealth SampleDataset. May be
                None when using the standalone pretrain/finetune API.
            window_size (int): EDA window length in samples. Defaults to 60.
            embed_dim (int): Encoder output embedding dimension. Defaults to
                128.
            proj_dim (int): Contrastive projection head output dimension.
                Defaults to 64.
            num_classes (int): Number of output classes for finetune mode.
                Defaults to 2.
            augmentation_group (str): Augmentation preset — one of 'full',
                'generic_only', 'eda_specific_only' — or a list of
                augmentation names. Defaults to 'full'.
            temperature (float): NT-Xent loss temperature. Defaults to 0.1.
            freeze_encoder (bool): If True in finetune mode, freezes encoder
                weights. Defaults to False.

        Raises:
            ValueError: If augmentation_group is not a known preset and not
                a list.

        Example::
            >>> model = ContrastiveEDAModel(window_size=60, num_classes=2)
            >>> loss = model.pretrain_step(torch.randn(8, 60))
        """
        super().__init__(dataset=dataset)
        self.window_size = window_size
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self._mode = "pretrain"

        # Resolve augmentation list
        if isinstance(augmentation_group, list):
            self.augmentation_names = augmentation_group
        else:
            if augmentation_group not in AUGMENTATION_GROUPS:
                raise ValueError(
                    f"augmentation_group must be one of "
                    f"{list(AUGMENTATION_GROUPS.keys())} or a list of names. "
                    f"Got: {augmentation_group}"
                )
            self.augmentation_names = AUGMENTATION_GROUPS[augmentation_group]

        self.encoder = EDAEncoder(
            window_size=window_size,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
        )
        self.loss_fn = NCELoss(temperature=temperature)

        # Classifier head (added in finetune mode)
        self.classifier: Optional[nn.Linear] = None

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_finetune_mode(self, num_classes: Optional[int] = None) -> None:
        """Switches model to finetune mode and attaches classifier head.

        Args:
            num_classes: Number of output classes. Uses self.num_classes
                if not provided.
        """
        self._mode = "finetune"
        if num_classes is not None:
            self.num_classes = num_classes
        self.classifier = nn.Linear(self.encoder.embed_dim, self.num_classes)
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen. Training classifier head only.")
        else:
            logger.info("Fine-tuning full network.")

    def set_pretrain_mode(self) -> None:
        """Switches model back to contrastive pre-training mode."""
        self._mode = "pretrain"
        for param in self.encoder.parameters():
            param.requires_grad = True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        In pretrain mode: returns projected embeddings of shape (N, proj_dim).
        In finetune mode: returns a dict with keys ``logit``, ``y_prob``, and
        optionally ``loss`` and ``y_true`` when ``y`` is provided.

        Args:
            x: EDA windows of shape (N, window_size).
            y: Integer class labels of shape (N,). Required for ``loss`` and
                ``y_true`` to appear in the finetune-mode output dict.

        Returns:
            Pretrain mode — projected embeddings of shape (N, proj_dim).
            Finetune mode — dict with ``logit`` (N, num_classes), ``y_prob``
            (N, num_classes), and when y is provided, ``loss`` (scalar) and
            ``y_true`` (N,).
        """
        if self._mode == "pretrain":
            return self.encoder(x)

        if self.classifier is None:
            raise RuntimeError(
                "Call set_finetune_mode() before running in finetune mode."
            )
        logits = self.classifier(self.encoder.encode(x))
        result: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": F.softmax(logits, dim=-1),
        }
        if y is not None:
            result["loss"] = F.cross_entropy(logits, y)
            result["y_true"] = y
        return result

    # ------------------------------------------------------------------
    # Training steps
    # ------------------------------------------------------------------

    def pretrain_step(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Computes contrastive loss for a batch of EDA windows.

        Augments each window into two views and computes NT-Xent loss
        between the projected embeddings.

        Args:
            x: EDA windows of shape (N, window_size).

        Returns:
            Scalar contrastive loss tensor.
        """
        device = x.device
        x_np = x.cpu().numpy()

        views1, views2 = [], []
        for window in x_np:
            v1, v2 = apply_augmentation_pair(window, self.augmentation_names)
            views1.append(v1)
            views2.append(v2)

        v1_tensor = torch.tensor(np.stack(views1), dtype=torch.float32).to(device)
        v2_tensor = torch.tensor(np.stack(views2), dtype=torch.float32).to(device)

        z1 = self.encoder(v1_tensor)
        z2 = self.encoder(v2_tensor)

        return self.loss_fn(z1, z2)

    def finetune_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cross-entropy loss and logits for supervised fine-tuning.

        Args:
            x: EDA windows of shape (N, window_size).
            y: Integer class labels of shape (N,).

        Returns:
            Tuple of (loss scalar, logits of shape (N, num_classes)).
        """
        result = self.forward(x, y=y)
        return result["loss"], result["logit"]

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_encoder(self, path: str) -> None:
        """Saves encoder weights to disk.

        Args:
            path: File path for the checkpoint (.pt file).
        """
        torch.save(self.encoder.state_dict(), path)
        logger.info(f"Encoder saved to {path}")

    def load_encoder(self, path: str, strict: bool = True) -> None:
        """Loads encoder weights from a checkpoint.

        Args:
            path: File path to the encoder checkpoint (.pt file).
            strict: Whether to strictly enforce key matching.
        """
        state = torch.load(path, map_location="cpu")
        self.encoder.load_state_dict(state, strict=strict)
        logger.info(f"Encoder loaded from {path}")