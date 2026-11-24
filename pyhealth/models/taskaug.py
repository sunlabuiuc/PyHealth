"""TaskAug: learned, class-conditioned data augmentation for ECGs.

Authors:
    Allen Zhang             (allenz3@illinois.edu)
    Bjorn Thorleifsson      (bthor5@illinois.edu)
    Rohan Vanjani           (vanjani3@illinois.edu)

Paper:
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms." CHIL 2022.

Paper link:
    https://proceedings.mlr.press/v174/raghu22a.html

Description:
    :class:`TaskAugPolicy` applies a few augmentations in sequence, each
    picked from a learned distribution and applied with a learned strength
    per class. :class:`TaskAug` wraps a classifier (by default
    :class:`ResNet1D`, the paper's) so that it trains on augmented signals.
    :func:`train_taskaug` trains the classifier on the training loss and the
    policy on the validation loss, through the hypergradient of
    :func:`hyper_step`. The operations and the hypergradient follow the
    authors' code, https://github.com/aniruddhraghu/ecg_aug (Apache-2.0).
"""

from __future__ import annotations

import copy
import functools
import os
import time
from collections.abc import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import RelaxedOneHotCategorical

from ..datasets import SampleDataset
from .base_model import BaseModel

OPERATIONS = ("warp", "wander", "noise", "mask", "displace", "scale", "noop")
"""The paper's operations: temporal warp, baseline wander, Gaussian noise,
time mask, temporal displacement, magnitude scale and no-op."""

# Initial strength of each learned operation, as in the reference code. The
# time mask has a fixed width and the no-op has no strength.
_INITIAL_MAGNITUDE = {
    "warp": 1.0,
    "wander": 0.0,
    "noise": 0.0,
    "displace": 0.5,
    "scale": 0.0,
}


def _resample(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Reads ``x`` at positions ``t + flow`` by linear interpolation.

    Positions outside the signal read zero. Differentiable in ``flow``, which
    is what makes warp and displacement strengths learnable.
    """
    length = x.shape[-1]
    position = torch.arange(length, device=x.device, dtype=x.dtype) + flow
    low = position.floor()
    fraction = position - low
    low = low.long()

    def take(index: torch.Tensor) -> torch.Tensor:
        inside = (index >= 0) & (index < length)
        index = index.clamp(0, length - 1).expand(-1, x.shape[1], -1)
        return x.gather(2, index) * inside

    return take(low) * (1 - fraction) + take(low + 1) * fraction


@functools.cache
def _smoothing_kernel(size: int = 51) -> torch.Tensor:
    """The warp's smoothing filter, built as the reference code does with scipy.

    ``scipy.ndimage.gaussian_filter1d`` of a centred impulse, sigma
    ``(size - 1) / 2``: a Gaussian with its tails folded back.
    """
    radius = (size - 1) // 2
    reach = 4 * radius  # scipy truncates the Gaussian at 4 sigma
    gaussian = np.exp(-0.5 * (np.arange(-reach, reach + 1) / radius) ** 2)
    impulse = np.zeros(size)
    impulse[radius] = 1.0
    kernel = np.convolve(
        np.pad(impulse, reach, mode="symmetric"),
        gaussian / gaussian.sum(),
        mode="valid",
    )
    return torch.from_numpy(kernel.astype(np.float32))


def _temporal_warp(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Locally stretches and compresses time without changing signal length.

    Each signal gets a random displacement field shared by all its leads,
    keeping their time alignment. Its initial Gaussian noise is scaled by
    ``100 * magnitude**2``; this is not a bound on the final displacement.
    The field is integrated with seven scaling-and-squaring steps at quarter
    resolution, upsampled, and smoothed with a 51-sample Gaussian kernel.
    Linear interpolation reads the signal at the resulting time positions;
    values beyond either end are zero.

    Reference: ``functional.rand_temporal_warp`` and ``warp_ops.RandWarpAug``
    in the authors' repository:
    https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py and
    https://github.com/aniruddhraghu/ecg_aug/blob/main/warp_ops.py.
    """
    batch, _, length = x.shape
    small = length // 4
    noise = torch.randn(batch, 1, length, device=x.device, dtype=x.dtype)
    flow = 100 * magnitude**2 * noise
    flow = F.interpolate(flow, size=small, mode="linear", align_corners=False)
    flow = flow * (small / length) / 2**7
    for _ in range(7):
        flow = flow + _resample(flow, flow)
    flow = F.interpolate(flow, size=length, mode="linear", align_corners=False)
    kernel = _smoothing_kernel().to(x).view(1, 1, -1)
    flow = F.conv1d(flow * (length / small), kernel, padding=kernel.shape[-1] // 2)
    return _resample(x, flow)


def _uniform(x: torch.Tensor) -> torch.Tensor:
    """One Uniform(0, 1) draw per signal of ``x``, shaped ``(batch, 1, 1)``."""
    return torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype)


def _baseline_wander(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Adds a slow baseline drift to each signal.

    A random sinusoid is shared across the signal's leads. Its amplitude is
    ``0.25 * sigmoid(magnitude) * U(0, 1)`` in input-signal units. Phase and
    frequency are sampled independently per signal. The sine argument uses
    normalized time from 0 to 1 and a coefficient in [5/3, 5), so it spans
    less than one cycle regardless of sampling rate. This follows the code's
    convention rather than specifying a physical frequency in Hz.

    Reference: ``functional.baseline_wander`` in the authors' repository:
    https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py.
    """
    amplitude = 0.25 * torch.sigmoid(magnitude) * _uniform(x)
    frequency = (_uniform(x) * 20 + 10) * 10 / 60
    phase = _uniform(x) * 2 * torch.pi
    time = torch.linspace(0, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    return x + amplitude * torch.sin(time * frequency + phase)


def _gaussian_noise(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Adds independent Gaussian noise at every lead and time point.

    Noise has zero mean and standard deviation
    ``0.25 * sigmoid(magnitude) * std(x)``. The signal standard deviation is
    measured separately for each lead over time and detached from autograd;
    the learned magnitude remains differentiable. Thus noisier or larger
    leads receive proportionally larger perturbations.

    Reference: ``functional.gaussian_noise`` in the authors' repository:
    https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py.
    """
    std = x.std(dim=2, keepdim=True).detach()
    return x + 0.25 * std * torch.sigmoid(magnitude) * torch.randn_like(x)


def _time_mask(x: torch.Tensor, width: float) -> torch.Tensor:
    """Removes a contiguous time window by replacing it with zeros.

    The window contains ``int(width * length)`` samples. One start position
    is drawn per call and shared across every signal and lead in the batch.
    Signal length is preserved: this masks samples rather than cropping
    the output. Width is fixed, not a learned magnitude.

    Reference: ``functional.rand_crop`` and ``operations.RandCrop`` in the
    authors' repository (despite their names, they also zero a window):
    https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py and
    https://github.com/aniruddhraghu/ecg_aug/blob/main/operations.py.
    PyHealth defaults to width 0.1; the reference class defaults to 0.05.
    """
    length = x.shape[-1]
    span = int(width * length)
    start = int(torch.randint(0, length - span, (1,)))
    keep = torch.ones(length, device=x.device, dtype=x.dtype)
    keep[start : start + span] = 0
    return x * keep


def _displace(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Translates each signal in time while preserving its length.

    A constant offset is sampled uniformly between
    ``-100 * magnitude**2`` and ``100 * magnitude**2`` samples per signal,
    shared across its leads. Unlike temporal warp, the offset does not vary
    over time. Fractional offsets use linear interpolation, with zeros
    outside the signal. A positive offset reads later input samples, so
    waveform features move toward the beginning of the output.

    Reference: ``functional.rand_displacement`` and ``warp_ops.DispAug`` in
    the authors' repository:
    https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py and
    https://github.com/aniruddhraghu/ecg_aug/blob/main/warp_ops.py.
    """
    return _resample(x, 100 * magnitude**2 * (1 - 2 * _uniform(x)))


def _magnitude_scale(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Multiplies each signal by a random amplitude factor.

    The factor ``sigmoid(magnitude) * Uniform(0.75, 1.25)`` is sampled per
    signal and shared across all leads and time points, preserving relative
    lead amplitudes and timing. Magnitude zero is not the identity: its
    sigmoid is 0.5, giving factors between 0.375 and 0.625.

    Reference: ``functional.magnitude_scale`` in the authors' repository:
    https://github.com/aniruddhraghu/ecg_aug/blob/main/functional.py.
    """
    return x * torch.sigmoid(magnitude) * (1.25 - 0.5 * _uniform(x))


_LEARNED_OPERATIONS = {
    "warp": _temporal_warp,
    "wander": _baseline_wander,
    "noise": _gaussian_noise,
    "displace": _displace,
    "scale": _magnitude_scale,
}


class TaskAugPolicy(nn.Module):
    """A TaskAug augmentation policy for signals with a binary label.

    Each of ``num_stages`` stages picks one operation for the batch with a
    Gumbel-softmax sample over ``logits[stage]`` and applies it with the
    strength ``magnitudes[name][stage, label]``. Frozen with
    ``requires_grad_(False)``, it gives the baselines:
    ``operations=("noop",)`` is no augmentation, ``("mask",)`` time masking.

    Args:
        operations: Operations to pick from, a subset of :data:`OPERATIONS`.
        num_stages: Operations applied in sequence (K in the paper).
        temperature: Temperature of the Gumbel-softmax sample.
        mask_width: Fraction of the signal the time mask zeroes; not learned.
            The default is the paper's (the reference code uses 0.05).

    Attributes:
        logits: Selection logits, ``(num_stages, len(operations))``.
        magnitudes: Strengths of each learned operation, ``(num_stages, 2)``.

    Raises:
        ValueError: If ``operations`` is empty or unknown, ``num_stages`` is
            not positive or ``mask_width`` is not in [0, 1).

    Examples:
        >>> import torch
        >>> from pyhealth.models import TaskAugPolicy
        >>> policy = TaskAugPolicy()
        >>> policy(torch.randn(4, 12, 1000), torch.tensor([0, 1, 1, 0])).shape
        torch.Size([4, 12, 1000])
    """

    def __init__(
        self,
        operations: Sequence[str] = OPERATIONS,
        num_stages: int = 2,
        temperature: float = 0.1,
        mask_width: float = 0.1,
    ) -> None:
        super().__init__()
        unknown = sorted(set(operations) - set(OPERATIONS))
        if unknown or not operations:
            raise ValueError(
                f"operations must be a non-empty subset of {OPERATIONS}, "
                f"got {tuple(operations)}"
            )
        if num_stages < 1:
            raise ValueError(f"num_stages must be positive, got {num_stages}")
        if not 0 <= mask_width < 1:
            raise ValueError(f"mask_width must be in [0, 1), got {mask_width}")
        self.operations = tuple(operations)
        self.temperature = temperature
        self.mask_width = mask_width
        self.logits = nn.Parameter(torch.ones(num_stages, len(self.operations)))
        self.magnitudes = nn.ParameterDict(
            {
                name: nn.Parameter(
                    torch.full((num_stages, 2), _INITIAL_MAGNITUDE[name])
                )
                for name in self.operations
                if name in _INITIAL_MAGNITUDE
            }
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Augments a batch.

        Args:
            x: Signals, ``(batch, channels, length)``.
            y: Binary labels, one per signal.

        Returns:
            The augmented signals, shaped like ``x``.
        """
        y = y.reshape(-1).long()
        temperature = torch.tensor(self.temperature, device=x.device)
        for stage, logits in enumerate(self.logits):
            sample = RelaxedOneHotCategorical(temperature, logits=logits).rsample()
            index = int(sample.argmax())
            name = self.operations[index]
            if name in self.magnitudes:
                magnitude = self.magnitudes[name][stage][y].view(-1, 1, 1)
                # Perform the chosen operation (name) on the input x and scale by magnitude
                x = _LEARNED_OPERATIONS[name](x, magnitude)
            elif name == "mask":
                x = _time_mask(x, self.mask_width)
            # A factor of exactly one that carries the gradient to the logits.
            x = x * (sample[index] / sample[index].detach())
        return x


class _ResidualBlock(nn.Module):
    """A 1D residual block computing ``ReLU(F(x) + shortcut(x))``.

    ``F`` uses two convolutions with batch normalization and an intermediate
    ReLU. The shortcut passes the input through, or projects it with a 1x1
    convolution and batch normalization when channels or stride change.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
        """Builds the convolution and shortcut paths.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            kernel_size: Odd temporal kernel size for both convolutions.
            stride: Temporal stride of the first convolution and shortcut.
        """
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the learned features to the shortcut and applies ReLU.

        Args:
            x: Input features, ``(batch, in_channels, length)``.

        Returns:
            Features with ``out_channels`` and time downsampled by ``stride``.
        """
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D(BaseModel):
    """The 1D ResNet-18 of Raghu et al. (2022) for multi-lead signals.

    A strided convolution (no max pooling, as in the reference code), stages
    of residual blocks, average pooling over time and a linear layer.

    Args:
        dataset: A dataset with one signal feature, ``(channels, length)``,
            and one label.
        kernel_size: Kernel size of every convolution.
        channels: Channels of each stage.
        blocks: Residual blocks of each stage.

    Raises:
        ValueError: If the dataset does not have one feature and one label, or
            ``channels`` and ``blocks`` differ in length.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import ResNet1D
        >>> signals = np.random.randn(4, 12, 500).astype(np.float32)
        >>> samples = [
        ...     {"patient_id": str(i), "signal": x, "label": i % 2}
        ...     for i, x in enumerate(signals)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"signal": "tensor"}, {"label": "binary"}
        ... )
        >>> model = ResNet1D(dataset)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> model(**batch)["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        kernel_size: int = 15,
        channels: Sequence[int] = (32, 64, 128, 256),
        blocks: Sequence[int] = (2, 2, 2, 2),
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1 or len(self.label_keys) != 1:
            raise ValueError("ResNet1D needs exactly one feature and one label")
        if len(channels) != len(blocks):
            raise ValueError("channels and blocks must have the same length")
        self.kernel_size = kernel_size
        in_channels = dataset[0][self.feature_keys[0]].shape[0]
        layers: list[nn.Module] = [
            # Padding 3, not kernel_size // 2, as in the reference code.
            nn.Conv1d(in_channels, channels[0], kernel_size, 2, 3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        ]
        width = channels[0]
        for stage, (out_channels, count) in enumerate(zip(channels, blocks)):
            for block in range(count):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(_ResidualBlock(width, out_channels, kernel_size, stride))
                width = out_channels
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(width, self.label_processor.output_size())
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Classifies a batch.

        Args:
            **kwargs: The signal, ``(batch, channels, length)``, under the
                feature key and, optionally, labels under the label key.

        Returns:
            ``logit`` and ``y_prob``, plus ``loss`` and ``y_true`` with labels.
        """
        x = kwargs[self.feature_keys[0]].to(self.device)
        logit = self.fc(self.encoder(x).mean(dim=2))
        results = {
            "logit": logit,
            "y_prob": self.label_processor.activation(logit),
        }
        y_true = kwargs.get(self.label_keys[0])
        if y_true is not None:
            y_true = y_true.to(self.device)
            results["loss"] = self.label_processor.loss(logit, y_true)
            results["y_true"] = y_true
        return results


class TaskAug(BaseModel):
    """A classifier that trains on signals augmented by a TaskAug policy.

    In training mode, labelled batches go through ``policy`` before
    ``backbone``; otherwise the signal is left as it is, so inference needs no
    labels. Train with :func:`train_taskaug` to learn the policy as well.

    Args:
        dataset: A dataset with one signal feature, ``(channels, length)``,
            and one binary label.
        backbone: Any PyHealth model for ``dataset``, such as
            :class:`~pyhealth.models.SparcNet`. Defaults to :class:`ResNet1D`.
        policy: Defaults to ``TaskAugPolicy()``.

    Raises:
        ValueError: If the dataset does not have one feature and a binary
            label.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TaskAug
        >>> signals = np.random.randn(4, 12, 500).astype(np.float32)
        >>> samples = [
        ...     {"patient_id": str(i), "signal": x, "label": i % 2}
        ...     for i, x in enumerate(signals)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"signal": "tensor"}, {"label": "binary"}
        ... )
        >>> model = TaskAug(dataset)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> sorted(model(**batch))  # trains on an augmented batch
        ['logit', 'loss', 'y_prob', 'y_true']
        >>> _ = model.eval()  # predicts on the clean signal
        >>> model(signal=batch["signal"])["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        backbone: BaseModel | None = None,
        policy: TaskAugPolicy | None = None,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1 or self.mode != "binary":
            raise ValueError(
                "TaskAug needs exactly one feature and one binary label: its "
                "augmentation strengths are indexed by a binary label"
            )
        self.backbone = backbone if backbone is not None else ResNet1D(dataset)
        self.policy = policy if policy is not None else TaskAugPolicy()

    def forward(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Augments the batch in training mode, then calls the backbone.

        Args:
            **kwargs: The signal under the feature key and, optionally, labels
                under the label key.

        Returns:
            The backbone's output.
        """
        feature_key, label_key = self.feature_keys[0], self.label_keys[0]
        if self.training and kwargs.get(label_key) is not None:
            x = kwargs[feature_key].to(self.device)
            y = kwargs[label_key].to(self.device)
            kwargs = {**kwargs, feature_key: self.policy(x, y)}
        return self.backbone(**kwargs)


def _grad(
    outputs: torch.Tensor | Sequence[torch.Tensor],
    inputs: Sequence[torch.Tensor],
    **kwargs,
) -> list[torch.Tensor]:
    """``torch.autograd.grad`` with zeros for inputs the outputs do not use."""
    grads = torch.autograd.grad(outputs, inputs, allow_unused=True, **kwargs)
    return [
        torch.zeros_like(param) if grad is None else grad
        for grad, param in zip(grads, inputs)
    ]


def hyper_step(
    model: TaskAug,
    train_batch: dict,
    val_batch: dict,
    lr: float,
    neumann_terms: int = 1,
) -> None:
    """Writes the policy's hypergradient into ``.grad`` of its parameters.

    By the implicit function theorem (Lorraine et al., 2020), with training
    loss ``L_T``, validation loss ``L_V``, weights ``w`` and policy ``p``::

        dL_V/dp = - dL_V/dw  [d2L_T/dw2]^-1  d2L_T/dw dp

    The inverse Hessian is approximated by a Neumann series. As in the
    reference code, both losses use batch statistics in batch norm and the
    validation batch is not augmented. Step a policy optimizer afterwards.

    Args:
        model: The model whose policy is differentiated.
        train_batch: A training batch; it is augmented.
        val_batch: A validation batch.
        lr: The classifier's learning rate, which scales the Neumann series.
        neumann_terms: Neumann terms after the identity.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TaskAug, hyper_step
        >>> signals = np.random.randn(4, 12, 500).astype(np.float32)
        >>> samples = [
        ...     {"patient_id": str(i), "signal": x, "label": i % 2}
        ...     for i, x in enumerate(signals)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"signal": "tensor"}, {"label": "binary"}
        ... )
        >>> model = TaskAug(dataset)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> hyper_step(model, batch, batch, lr=1e-3)
        >>> model.policy.logits.grad.shape
        torch.Size([2, 7])
    """
    policy_params = [p for p in model.policy.parameters() if p.requires_grad]
    weights = [p for p in model.backbone.parameters() if p.requires_grad]
    model.train()
    train_loss = model(**train_batch)["loss"]
    train_grads = torch.autograd.grad(
        train_loss, weights, create_graph=True, allow_unused=True
    )
    # Drop weights the loss does not use, such as BaseModel's device marker.
    used = [(w, g) for w, g in zip(weights, train_grads) if g is not None]
    weights = [w for w, _ in used]
    train_grads = [g for _, g in used]

    val_loss = model.backbone(**val_batch)["loss"]
    term = _grad(val_loss, weights)
    total = term
    for _ in range(neumann_terms):
        hessian_term = _grad(train_grads, weights, grad_outputs=term, retain_graph=True)
        term = [t - lr * h for t, h in zip(term, hessian_term)]
        total = [a + t for a, t in zip(total, term)]
    mixed = _grad(train_grads, policy_params, grad_outputs=[lr * t for t in total])
    for param, grad in zip(policy_params, mixed):
        param.grad = -grad


def train_taskaug(
    model: TaskAug,
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    epochs: int = 100,
    lr: float = 1e-3,
    policy_lr: float = 1e-2,
    warmup_epochs: int = 5,
    neumann_terms: int = 1,
    policy_every: int = 1,
    checkpoint_path: str | os.PathLike | None = None,
) -> dict[str, list[float]]:
    """Trains a :class:`TaskAug` classifier and, if it is trainable, its policy.

    Every step updates the classifier with Adam on an augmented batch. After
    ``warmup_epochs``, every ``policy_every`` steps also update the policy
    with RMSprop on the hypergradient of :func:`hyper_step`. The classifier
    ends with the weights of the epoch with the lowest validation loss
    (without augmentation). With a frozen policy this trains a
    fixed-augmentation baseline under the same rules.

    Args:
        model: The model to train, already on its device.
        train_loader: Training batches.
        val_loader: Validation batches; read once per epoch and kept in memory.
        epochs: Passes over ``train_loader``.
        lr: Learning rate of the classifier.
        policy_lr: Learning rate of the policy.
        warmup_epochs: Initial epochs that leave the policy unchanged.
        neumann_terms: Neumann terms of the hypergradient.
        policy_every: Classifier steps per policy step (P in the paper).
        checkpoint_path: If given, the run is saved here after every epoch and
            resumed from here. A resumed run restarts the loaders' shuffling,
            so it is not identical to an uninterrupted one.

    Returns:
        Per epoch: ``train_loss``, ``val_loss`` and ``seconds``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TaskAug, train_taskaug
        >>> signals = np.random.randn(4, 12, 500).astype(np.float32)
        >>> samples = [
        ...     {"patient_id": str(i), "signal": x, "label": i % 2}
        ...     for i, x in enumerate(signals)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"signal": "tensor"}, {"label": "binary"}
        ... )
        >>> model = TaskAug(dataset)
        >>> loader = get_dataloader(dataset, batch_size=4, shuffle=True)
        >>> history = train_taskaug(model, loader, loader, epochs=2, warmup_epochs=1)
        >>> len(history["val_loss"])
        2
    """
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    backbone_optimizer = torch.optim.Adam(backbone_params, lr=lr)
    policy_params = [p for p in model.policy.parameters() if p.requires_grad]
    policy_optimizer = (
        torch.optim.RMSprop(policy_params, lr=policy_lr) if policy_params else None
    )
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "seconds": []}
    best_state = None
    step = 0
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=model.device)
        model.load_state_dict(state["model"])
        backbone_optimizer.load_state_dict(state["optimizer"])
        if policy_optimizer is not None:
            policy_optimizer.load_state_dict(state["policy_optimizer"])
        history, best_state, step = state["history"], state["best_state"], state["step"]
        torch.set_rng_state(state["rng"])
    best_loss = min(history["val_loss"], default=float("inf"))
    for epoch in range(len(history["val_loss"]), epochs):
        started = time.perf_counter()
        # One pass per epoch serves the policy steps and the epoch's
        # validation loss, so the loader is never read twice at once.
        val_batches = list(val_loader)
        train_losses = []
        for batch in train_loader:
            # Update the classifier on augmented training signals.
            model.train()
            loss = model(**batch)["loss"]
            backbone_optimizer.zero_grad()
            loss.backward(inputs=backbone_params)
            backbone_optimizer.step()
            train_losses.append(loss.item())

            # Update the policy using clean validation performance.
            should_update_policy = (
                policy_optimizer is not None
                and epoch >= warmup_epochs
                and step % policy_every == 0
            )
            if should_update_policy:
                val_batch = val_batches[step % len(val_batches)]
                hyper_step(model, batch, val_batch, lr, neumann_terms)
                policy_optimizer.step()

            step += 1

        model.eval()
        val_loss, count = 0.0, 0
        with torch.no_grad():
            for batch in val_batches:
                output = model(**batch)
                val_loss += output["loss"].item() * len(output["y_true"])
                count += len(output["y_true"])
        history["train_loss"].append(sum(train_losses) / len(train_losses))
        history["val_loss"].append(val_loss / count)
        if history["val_loss"][-1] < best_loss:
            best_loss = history["val_loss"][-1]
            best_state = copy.deepcopy(model.backbone.state_dict())
        history["seconds"].append(time.perf_counter() - started)
        if checkpoint_path is not None:
            state = {
                "model": model.state_dict(),
                "optimizer": backbone_optimizer.state_dict(),
                "policy_optimizer": policy_optimizer and policy_optimizer.state_dict(),
                "history": history,
                "best_state": best_state,
                "step": step,
                "rng": torch.get_rng_state(),
            }
            # Written beside the file and moved over it, so a run that is
            # killed while saving keeps the previous checkpoint.
            torch.save(state, f"{checkpoint_path}.tmp")
            os.replace(f"{checkpoint_path}.tmp", checkpoint_path)
    if best_state is not None:
        model.backbone.load_state_dict(best_state)
    return history
