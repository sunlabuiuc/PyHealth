"""TaskAug: learned, class-conditioned data augmentation for ECGs.

Authors:
    Bjorn Elvar (bthor5@illinois.edu), Rohan Vanjani

Paper:
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms." CHIL 2022.

Paper link:
    https://proceedings.mlr.press/v174/raghu22a.html

Description:
    A re-implementation of TaskAug for PyHealth. The augmentation formulas,
    initial strengths and the hypergradient follow the authors' reference
    code, https://github.com/aniruddhraghu/ecg_aug (Apache-2.0), rewritten
    here in plain PyTorch for signals of any length.

    A :class:`TaskAugPolicy` applies ``num_stages`` augmentations in sequence.
    Each stage samples one operation from a learned categorical distribution
    and applies it with a learned strength that depends on the binary label.
    :class:`TaskAug` wraps a classifier (by default :class:`ResNet1D`, the
    paper's backbone) so that it trains on augmented signals and predicts on
    clean ones. :func:`train_taskaug` trains both: the classifier on the
    augmented training loss, the policy on the validation loss, through the
    implicit-function-theorem hypergradient of :func:`hyper_step`.
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
"""The paper's operation set: temporal warp, baseline wander, Gaussian noise,
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
    """Reads ``x`` at the positions ``t + flow`` by linear interpolation.

    Positions outside the signal read zero. The result is differentiable in
    ``flow``, which is what lets a warp or displacement strength be learned.

    Args:
        x: Signals of shape ``(batch, channels, length)``.
        flow: Offsets in samples, broadcastable to ``(batch, 1, length)``.

    Returns:
        The resampled signals, with the shape of ``x``.
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
    """The warp's smoothing filter: a Gaussian with its tails folded back.

    Equals ``scipy.ndimage.gaussian_filter1d`` applied to a centred unit
    impulse of ``size`` samples with sigma ``(size - 1) / 2``, which is how the
    reference code builds it.
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
    """Stretches and squeezes the signal in time with a smooth random flow.

    White noise of scale ``100 * magnitude**2`` samples is integrated into a
    diffeomorphic flow at a quarter of the resolution (7 scaling-and-squaring
    steps), upsampled and smoothed.
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
    """Adds one slow sinusoid to all leads.

    Its amplitude is ``0.25 * sigmoid(magnitude) * Uniform(0, 1)``. As in the
    reference code, the sinusoid's argument is ``t * f + phase`` with ``t`` in
    [0, 1] and ``f`` in [1.67, 5], which is under one cycle per signal.
    """
    amplitude = 0.25 * torch.sigmoid(magnitude) * _uniform(x)
    frequency = (_uniform(x) * 20 + 10) * 10 / 60
    phase = _uniform(x) * 2 * torch.pi
    time = torch.linspace(0, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    return x + amplitude * torch.sin(time * frequency + phase)


def _gaussian_noise(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Adds noise of ``0.25 * sigmoid(magnitude)`` times each lead's std."""
    std = x.std(dim=2, keepdim=True).detach()
    return x + 0.25 * std * torch.sigmoid(magnitude) * torch.randn_like(x)


def _time_mask(x: torch.Tensor, width: float) -> torch.Tensor:
    """Zeroes one random window, ``width`` of the signal, for the whole batch."""
    length = x.shape[-1]
    span = int(width * length)
    start = int(torch.randint(0, length - span, (1,)))
    keep = torch.ones(length, device=x.device, dtype=x.dtype)
    keep[start : start + span] = 0
    return x * keep


def _displace(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Shifts each signal in time by up to ``100 * magnitude**2`` samples."""
    return _resample(x, 100 * magnitude**2 * (1 - 2 * _uniform(x)))


def _magnitude_scale(x: torch.Tensor, magnitude: torch.Tensor) -> torch.Tensor:
    """Scales each signal by ``sigmoid(magnitude) * Uniform(0.75, 1.25)``."""
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

    The policy applies ``num_stages`` operations in sequence. Stage ``k``
    draws one operation for the batch from ``softmax(logits[k])`` with a
    Gumbel-softmax sample ``u``, applies it, and multiplies the result by
    ``u_i / stopgrad(u_i)``: a factor of exactly one that carries the gradient
    to ``logits``. Each learned operation has one strength per stage and per
    class, ``magnitudes[name][k, label]``, so positives and negatives can be
    augmented differently.

    Call ``policy.requires_grad_(False)`` for a fixed policy: with
    ``operations=("mask",)`` it is the time-masking baseline, with
    ``("noop",)`` no augmentation, and with the defaults the paper's
    initial-policy ablation.

    Args:
        operations: Names of the operations each stage chooses from, a subset
            of :data:`OPERATIONS`. Defaults to all seven.
        num_stages: Number of operations applied in sequence, K in the paper.
            Defaults to 2.
        temperature: Temperature of the Gumbel-softmax sample. Defaults to 0.1.
        mask_width: Fraction of the signal the time mask zeroes. It is not
            learned. Defaults to 0.1, the paper's value (the reference code
            builds its policy with 0.05).

    Attributes:
        logits: Selection parameters of shape ``(num_stages, len(operations))``,
            initialized to a uniform choice.
        magnitudes: Maps each learned operation among ``operations`` to its
            strengths, of shape ``(num_stages, 2)``, indexed by stage and label.

    Raises:
        ValueError: If an operation is unknown, ``operations`` is empty,
            ``num_stages`` is not positive or ``mask_width`` is not in [0, 1).

    Examples:
        >>> import torch
        >>> from pyhealth.models import TaskAugPolicy
        >>> policy = TaskAugPolicy()
        >>> signal = torch.randn(4, 12, 1000)
        >>> label = torch.tensor([0.0, 1.0, 1.0, 0.0])
        >>> policy(signal, label).shape
        torch.Size([4, 12, 1000])
        >>> policy.logits.softmax(-1).shape  # selection probabilities
        torch.Size([2, 7])
        >>> policy.magnitudes["warp"].shape  # strengths: (stage, label)
        torch.Size([2, 2])
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
        """Augments a batch of signals.

        Args:
            x: Signals of shape ``(batch, channels, length)``.
            y: Binary labels, ``batch`` values in {0, 1} in any shape.

        Returns:
            The augmented signals, with the shape of ``x``.
        """
        y = y.reshape(-1).long()
        temperature = torch.tensor(self.temperature, device=x.device)
        for stage, logits in enumerate(self.logits):
            sample = RelaxedOneHotCategorical(temperature, logits=logits).rsample()
            index = int(sample.argmax())
            name = self.operations[index]
            if name in self.magnitudes:
                magnitude = self.magnitudes[name][stage][y].view(-1, 1, 1)
                x = _LEARNED_OPERATIONS[name](x, magnitude)
            elif name == "mask":
                x = _time_mask(x, self.mask_width)
            x = x * (sample[index] / sample[index].detach())
        return x


class _ResidualBlock(nn.Module):
    """Two convolutions with batch norm and a skip connection."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, stride: int
    ) -> None:
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
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResNet1D(BaseModel):
    """The 1D ResNet-18 of Raghu et al. (2022) for multi-lead signals.

    A strided convolution, four stages of residual blocks, average pooling
    over time and a linear layer. It follows the paper's reference code: no
    max pooling after the first convolution, and a stride of 2 at the first
    convolution and at the start of every stage but the first.

    Args:
        dataset: The dataset to train on. It needs one input feature, a
            signal of shape ``(channels, length)``, and one label.
        kernel_size: Size of every convolution kernel. Defaults to 15.
        channels: Number of channels of each stage. Defaults to
            ``(32, 64, 128, 256)``.
        blocks: Number of residual blocks of each stage. Defaults to
            ``(2, 2, 2, 2)``.

    Raises:
        ValueError: If the dataset does not have exactly one feature and one
            label, or ``channels`` and ``blocks`` differ in length.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import ResNet1D
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "signal": np.random.randn(12, 500).astype(np.float32),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
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
        """Forward propagation.

        Args:
            **kwargs: A batch with the signal, of shape
                ``(batch, channels, length)``, under the dataset's feature key
                and, optionally, the labels under its label key.

        Returns:
            A dict with ``logit`` and ``y_prob`` and, when the batch has
            labels, ``loss`` and ``y_true``.
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

    In training mode, a batch with labels goes through ``policy`` before it
    reaches ``backbone``. In evaluation mode, or without labels, the signal
    reaches ``backbone`` untouched, so inference needs no labels. Train it
    with :func:`train_taskaug`, which also learns the policy; any other
    training loop trains the backbone under the policy as it is.

    Args:
        dataset: The dataset to train on. It needs one input feature, a
            signal of shape ``(channels, length)``, and one binary label.
        backbone: The classifier, any PyHealth model for ``dataset``, such as
            :class:`~pyhealth.models.SparcNet`. Defaults to
            :class:`ResNet1D`, the paper's backbone.
        policy: The augmentation policy. Defaults to ``TaskAugPolicy()``,
            the paper's two-stage policy over all seven operations.

    Raises:
        ValueError: If the dataset does not have exactly one feature and one
            binary label.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TaskAug
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "signal": np.random.randn(12, 500).astype(np.float32),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
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
        """Forward propagation.

        Args:
            **kwargs: A batch with the signal, of shape
                ``(batch, channels, length)``, under the dataset's feature key
                and, optionally, the labels under its label key.

        Returns:
            The backbone's output: ``logit``, ``y_prob`` and, when the batch
            has labels, ``loss`` and ``y_true``.
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
    """Writes the validation loss's gradient into the policy parameters.

    The policy affects the validation loss only through the classifier
    weights it leads to. By the implicit function theorem (Lorraine et al.,
    2020), with training loss ``L_T``, validation loss ``L_V``, weights ``w``
    and policy parameters ``p``::

        dL_V/dp = - dL_V/dw  [d2L_T/dw2]^-1  d2L_T/dw dp

    The inverse Hessian is approximated by a Neumann series,
    ``lr * sum_{j=0..neumann_terms} (I - lr * H)^j``, computed with
    Hessian-vector products. The result is stored in ``.grad`` of every
    trainable policy parameter; step a policy optimizer afterwards.

    The model is put in training mode, as in the reference code: both losses
    use batch statistics in batch norm, and the validation batch is not
    augmented.

    Args:
        model: The model whose policy to differentiate.
        train_batch: A training batch. It is augmented by the policy.
        val_batch: A validation batch.
        lr: The classifier's learning rate, which scales the Neumann series.
        neumann_terms: Number of Neumann terms after the identity. Defaults
            to 1, the paper's value.

    Examples:
        >>> import numpy as np
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TaskAug, hyper_step
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "signal": np.random.randn(12, 500).astype(np.float32),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = TaskAug(dataset)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=4)))
        >>> hyper_step(model, train_batch=batch, val_batch=batch, lr=1e-3)
        >>> model.policy.logits.grad.shape
        torch.Size([2, 7])
        >>> torch.optim.RMSprop(model.policy.parameters(), lr=1e-2).step()
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
    """Trains a :class:`TaskAug` model and, if it is trainable, its policy.

    Each step updates the classifier with Adam on an augmented training
    batch. After ``warmup_epochs``, every ``policy_every`` steps also update
    the policy with RMSprop on the hypergradient of :func:`hyper_step`,
    computed from the same training batch, augmented anew, and the next
    validation batch. After each epoch the validation loss is measured
    without augmentation; when training ends, the classifier has the weights
    of the epoch where it was lowest.

    A model whose policy has no trainable parameter
    (``policy.requires_grad_(False)``) trains under that fixed policy, so
    baselines can share this loop and its checkpoint rule.

    Move the model to its device first; batches follow it.

    With ``checkpoint_path``, the state of the run is saved there after every
    epoch, and a call that finds the file continues after the last epoch it
    holds, so a run that was cut short loses at most one epoch. The file
    holds the model, both optimizers, the best weights so far, the history
    and torch's CPU random state. Loaders start their shuffling order over,
    so a resumed run is not identical to an uninterrupted one.

    Args:
        model: The model to train.
        train_loader: Training batches.
        val_loader: Validation batches. It is read once per epoch and held in
            memory, so it should fit there.
        epochs: Number of passes over ``train_loader``. Defaults to 100.
        lr: Learning rate of the classifier. Defaults to 1e-3.
        policy_lr: Learning rate of the policy. Defaults to 1e-2.
        warmup_epochs: Number of initial epochs that leave the policy as it
            is. Defaults to 5.
        neumann_terms: Neumann terms of the hypergradient. Defaults to 1.
        policy_every: Classifier steps per policy step, P in the paper.
            Defaults to 1.
        checkpoint_path: Where to save the run after every epoch and to
            resume from. Defaults to None, which saves nothing.

    Returns:
        The per-epoch ``train_loss`` (augmented, averaged over steps),
        ``val_loss`` (not augmented, averaged over samples) and ``seconds``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import TaskAug, train_taskaug
        >>> samples = [
        ...     {
        ...         "patient_id": f"p{i}",
        ...         "signal": np.random.randn(12, 500).astype(np.float32),
        ...         "label": i % 2,
        ...     }
        ...     for i in range(8)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"signal": "tensor"},
        ...     output_schema={"label": "binary"},
        ... )
        >>> model = TaskAug(dataset)
        >>> loader = get_dataloader(dataset, batch_size=4, shuffle=True)
        >>> history = train_taskaug(
        ...     model, loader, loader, epochs=2, warmup_epochs=1
        ... )
        >>> len(history["val_loss"])
        2
    """
    weights = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(weights, lr=lr)
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
        optimizer.load_state_dict(state["optimizer"])
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
            model.train()
            loss = model(**batch)["loss"]
            optimizer.zero_grad()
            loss.backward(inputs=weights)
            optimizer.step()
            train_losses.append(loss.item())
            if (
                policy_optimizer is not None
                and epoch >= warmup_epochs
                and step % policy_every == 0
            ):
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
                "optimizer": optimizer.state_dict(),
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
