"""Joint sequence and image model for irregular medical time series.

Paper:
    Liuqing Chen, Shuhong Xiao, Shixian Ding, Shanhai Hu and Lingyun Sun.
    "Integrating Sequence and Image Modeling in Irregular Medical Time
    Series Through Self-Supervised Learning." AAAI 2025.
    https://arxiv.org/abs/2502.06134

This is a reimplementation from the paper. The authors' repository
(https://github.com/zju-d3/AAAI25-Irregular-Medical-Time-Series) holds no
code, and the paper's appendix with the remaining hyperparameters is not
public, so the choices the paper leaves open are documented on
:class:`JointSeqImage`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..datasets import SampleDataset
from .base_model import BaseModel
from .time_series_images import IMAGE_TYPES, grid_shape, render_images

SWIN_BASE_IN22K = "microsoft/swin-base-patch4-window7-224-in22k"
"""Swin-Base pretrained on ImageNet-21k, patch size 4 and window size 7."""

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def time_lags(times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Time since each variable's previous observation, ``[B, L, V]``.

    The lag matrix of GRU-D (Che et al., 2018) and the paper: 0 at the first
    step, else the gap to the previous step plus, if the variable was not
    observed at the previous step, that step's lag.

    Args:
        times: ``[B, L]``; non-finite (padding) times count as no gap.
        mask: ``[B, L, V]``, true where a value was observed.

    Examples:
        >>> import torch
        >>> t = torch.tensor([[0.0, 1.0, 3.0]])
        >>> m = torch.tensor([[[1.0], [0.0], [1.0]]])
        >>> time_lags(t, m)[0, :, 0]
        tensor([0., 1., 3.])
    """
    gaps = torch.nan_to_num(times.diff(dim=1), nan=0.0, posinf=0.0).clamp(min=0)
    gaps = torch.cat([gaps.new_zeros(gaps.shape[0], 1), gaps], dim=1)
    lags = [torch.zeros_like(mask[:, 0], dtype=times.dtype)]
    for step in range(1, mask.shape[1]):
        carried = lags[-1] * (1 - mask[:, step - 1].to(times.dtype))
        lags.append(gaps[:, step].unsqueeze(-1) + carried)
    return torch.stack(lags, dim=1)


def reverse_padded(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Reverses the first ``lengths[b]`` steps of each ``x[b]`` (``[B, L, ...]``).

    Padding steps stay at the end, so a backward recurrence starts at each
    sequence's last real step.

    Examples:
        >>> import torch
        >>> reverse_padded(torch.tensor([[1, 2, 3, 0]]), torch.tensor([3]))
        tensor([[3, 2, 1, 0]])
    """
    length = x.shape[1]
    steps = torch.arange(length, device=x.device).unsqueeze(0)
    index = torch.where(steps < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - steps, steps)
    index = index.view(*index.shape, *([1] * (x.dim() - 2))).expand_as(x)
    return torch.gather(x, 1, index)


class DecayImputer(nn.Module):
    """One direction of the imputing generator (paper Eqs. 1-3).

    A stack of GRU cells reads the series step by step. Before each step the
    hidden states are decayed by ``gamma_t = exp(-relu(W_g lag_t + b_g))``,
    and the input is the observed value where there is one and the previous
    step's reconstruction ``x_hat_t`` where there is not, plus the decayed
    last observation ``x_delta = x_last * exp(-relu(W_d lag_t + b_d))``. The
    top hidden state predicts the next step's reconstruction.

    The paper writes the input as ``x_hat_t + x_delta``; feeding the observed
    values where they exist, as BRITS and GRU-D do, is our reading of how
    the observations enter the generator.

    Args:
        num_variables: Variables per step.
        hidden_dim: Units per GRU layer (128 in the paper).
        num_layers: GRU layers (4 in the paper).

    Examples:
        >>> import torch
        >>> x, m = torch.randn(2, 5, 3), torch.ones(2, 5, 3)
        >>> lags = time_lags(torch.arange(5.0).expand(2, 5), m)
        >>> DecayImputer(3, 8, 2)(x, m, lags).shape
        torch.Size([2, 5, 3])
    """

    def __init__(self, num_variables: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hidden_decay = nn.Linear(num_variables, hidden_dim)
        self.input_decay = nn.Linear(num_variables, num_variables)
        self.cells = nn.ModuleList(
            nn.GRUCell(2 * num_variables if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        )
        self.reconstruct = nn.Linear(hidden_dim, num_variables)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, lags: torch.Tensor
    ) -> torch.Tensor:
        """Reconstructions ``[B, L, V]``; step ``t`` sees steps before ``t`` only.

        Args:
            x: Values ``[B, L, V]``, 0 where not observed.
            mask: ``[B, L, V]``, 1 where observed.
            lags: :func:`time_lags` of the series, ``[B, L, V]``.
        """
        batch, length, n_vars = x.shape
        hidden = [x.new_zeros(batch, self.hidden_dim) for _ in self.cells]
        last = x.new_zeros(batch, n_vars)
        x_hat = x.new_zeros(batch, n_vars)
        outputs = []
        for step in range(length):
            x_t, m_t, lag_t = x[:, step], mask[:, step], lags[:, step]
            outputs.append(x_hat)
            gamma = torch.exp(-F.relu(self.hidden_decay(lag_t)))
            x_delta = last * torch.exp(-F.relu(self.input_decay(lag_t)))
            filled = m_t * x_t + (1 - m_t) * (x_hat + x_delta)
            layer_input = torch.cat([filled, m_t], dim=-1)
            for i, cell in enumerate(self.cells):
                hidden[i] = cell(layer_input, gamma * hidden[i])
                layer_input = hidden[i]
            x_hat = self.reconstruct(hidden[-1])
            last = m_t * x_t + (1 - m_t) * last
        return torch.stack(outputs, dim=1)


class MaskDiscriminator(nn.Module):
    """Predicts which entries of an imputed series were observed (Eq. 8).

    A stack of GRUs with the paper's sizes ``{128, 64, 16, 64, 128}``
    followed by a per-variable logit.

    Args:
        num_variables: Variables per step.
        sizes: Units of the stacked GRUs.

    Examples:
        >>> import torch
        >>> MaskDiscriminator(3)(torch.randn(2, 5, 3)).shape
        torch.Size([2, 5, 3])
    """

    def __init__(self, num_variables: int, sizes: tuple[int, ...] = (128, 64, 16, 64, 128)) -> None:
        super().__init__()
        dims = (num_variables, *sizes)
        self.layers = nn.ModuleList(
            nn.GRU(dims[i], dims[i + 1], batch_first=True) for i in range(len(sizes))
        )
        self.out = nn.Linear(sizes[-1], num_variables)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits ``[B, L, V]`` that each entry of ``x`` was observed."""
        for layer in self.layers:
            x = layer(x)[0]
        return self.out(x)


def nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    """NT-Xent loss (Eq. 6) with ``(a[i], b[i])`` as the positive pairs.

    Each of the ``2B`` vectors is contrasted with the other ``2B - 1``.

    Args:
        a: ``[B, D]``.
        b: ``[B, D]``.
        temperature: Softmax temperature.

    Examples:
        >>> import torch
        >>> x = torch.eye(4)
        >>> bool(nt_xent(x, x, 0.1) < nt_xent(x, x.roll(1, 0), 0.1))
        True
    """
    z = F.normalize(torch.cat([a, b]), dim=-1)
    sim = z @ z.T / temperature
    sim = sim.masked_fill(torch.eye(len(z), dtype=torch.bool, device=z.device), float("-inf"))
    n = len(a)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(sim, target)


def margin_contrastive(
    s: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    margin: float,
) -> torch.Tensor:
    """Sequence-image contrastive loss with a class margin (Eq. 9).

    For sample ``i``, ``(v_i, s_i)`` is the positive pair. Pairs with a
    sample of another class get ``margin`` added to their similarity, so
    they must be pushed further apart. The losses of both directions
    (image to sequence and sequence to image) are summed. Similarities are
    cosines.

    Args:
        s: Sequence features ``[B, D]``.
        v: Image features ``[B, D]``.
        labels: Class of each sample, ``[B]``.
        temperature: Softmax temperature.
        margin: Added to the similarity of pairs from different classes.

    Examples:
        >>> import torch
        >>> s, labels = torch.eye(4), torch.tensor([0, 0, 1, 1])
        >>> bool(margin_contrastive(s, s, labels, 0.1, 0.2)
        ...      < margin_contrastive(s, s.roll(1, 0), labels, 0.1, 0.2))
        True
    """
    sim = F.normalize(v, dim=-1) @ F.normalize(s, dim=-1).T  # [image, sequence]
    other = labels.view(-1, 1) != labels.view(1, -1)
    logits = (sim + margin * other) / temperature
    target = torch.arange(len(s), device=s.device)
    return F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)


def focal_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, binary: bool = True
) -> torch.Tensor:
    """Focal loss (Lin et al., 2017) for binary or multiclass logits.

    Args:
        logits: ``[B, 1]`` (binary) or ``[B, C]``.
        target: ``[B, 1]`` of 0/1, or ``[B]`` class indices.
        gamma: Focusing parameter; 0 gives cross-entropy.
        binary: Whether ``logits`` are binary.

    Examples:
        >>> import torch
        >>> logit, y = torch.tensor([[2.0]]), torch.tensor([[1.0]])
        >>> bool(focal_loss(logit, y) < F.binary_cross_entropy_with_logits(logit, y))
        True
    """
    if binary:
        target = target.float().view_as(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    else:
        ce = F.cross_entropy(logits, target.long().view(-1), reduction="none")
    p_true = torch.exp(-ce)
    return ((1 - p_true) ** gamma * ce).mean()


def _mlp(sizes: tuple[int, ...], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers += [nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class JointSeqImage(BaseModel):
    """Joint sequence and image model for irregular time series (AAAI 2025).

    Three parts (paper Fig. 1):

    - **Sequence branch**: a bidirectional generator of
      :class:`DecayImputer` s fills in the missing values; the imputed
      series ``X' = M*X + (1-M)*mean(forward, backward)`` gives the sequence
      feature ``s = Dropout(W LayerNorm(X'))``. A
      :class:`MaskDiscriminator` tells imputed entries from observed ones.
    - **Image branch**: each sample is drawn as a grid image
      (:func:`~pyhealth.models.time_series_images.render_images`, on the
      fly) and encoded by a Swin Transformer pretrained on ImageNet-21k.
    - **Joint module**: an MLP projects each feature into a joint space;
      their concatenation ``u`` is classified. A margin contrastive loss
      aligns the two projections (Eq. 9) and a clustering loss pulls ``u``
      towards its k-means centre (Eqs. 10-12).

    Training alternates three steps per batch, as in the paper
    (:meth:`train_step`): update the discriminator on the imputations;
    update the generator with ``L_adv + alpha * L_rec``; then update the
    whole model with ``L_clf + beta1 * L_cont + beta2 * L_cluster``, where
    ``L_clf`` is the focal loss for binary and multiclass labels alike
    (``focal_gamma=0`` gives cross-entropy). Call
    :meth:`update_clusters` at the end of each epoch to recompute the
    centres from the features :meth:`train_step` saw in it. :meth:`forward`
    computes the third objective, so the model also trains with
    :class:`~pyhealth.trainer.Trainer`, but then without the adversarial
    imputation steps and without cluster updates (nothing is stored for
    them).

    The model reads ``x`` (values ``[B, L, V]``, NaN where not observed,
    ideally z-scored) and ``x_time`` (``[B, L]`` in hours, NaN for padding),
    as :class:`~pyhealth.tasks.PhysioNet2012Mortality` produces, and one
    binary or multiclass label.

    Choices the paper leaves open (its appendix is not public), each an
    argument: the loss weights and temperatures, the contrastive margin, the
    number of clusters, the image value range, how the observations enter
    the generator (see :class:`DecayImputer`) and the use of one image type
    per model (the paper lists six types without saying how they combine).

    Args:
        dataset: The samples; used for the field names, the number of
            variables and the label type.
        image_type: One of
            :data:`~pyhealth.models.time_series_images.IMAGE_TYPES`.
        image_size: Side of the rendered image in pixels (384 in the paper
            for PhysioNet 2012 and 2019), split evenly into the grid cells;
            the image is ``image_size // max(rows, cols)`` times the grid,
            so with 36 variables (6 x 6) 224 gives 222 x 222.
        image_model: Hugging Face name of the pretrained Swin, or ``None``
            to build one from ``swin_config`` with random weights.
        swin_config: Arguments of ``transformers.SwinConfig`` when
            ``image_model`` is ``None`` (e.g. a tiny model for tests).
        time_range: The time axis of the images, in the units of ``x_time``.
        value_range: The value axis of the images; values outside are
            clipped.
        hidden_dim: Units of the generator GRUs (128).
        num_gru_layers: Layers of each generator direction (4).
        projection_dims: Hidden and output sizes of each projection MLP
            (``(1024, 512, 1024)``).
        dropout: Dropout rate.
        alpha: Weight of the reconstruction loss in the generator step.
        beta1: Weight of the contrastive loss.
        beta2: Weight of the clustering loss.
        temperature: Temperature of the NT-Xent, contrastive and clustering
            losses.
        margin: Contrastive margin for pairs of different classes.
        num_clusters: k of the k-means clustering.
        focal_gamma: Focusing parameter of the focal loss.
        reconstruction: ``"nt_xent"`` (the paper's) or ``"mse"`` (the
            observed values, its ablation).
        value_key: Input field with the values.
        time_key: Input field with the times.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> rng = np.random.default_rng(0)
        >>> def sample(i):
        ...     x = rng.normal(size=(6, 4)).astype("float32")
        ...     x[rng.random((6, 4)) < 0.5] = np.nan
        ...     t = np.arange(6, dtype="float32") * 8
        ...     return {"patient_id": f"p{i}", "x": x, "x_time": t, "label": i % 2}
        >>> dataset = create_sample_dataset(
        ...     [sample(i) for i in range(4)],
        ...     {"x": "tensor", "x_time": "tensor"}, {"label": "binary"},
        ... )
        >>> model = JointSeqImage(
        ...     dataset, image_size=32, image_model=None,
        ...     swin_config={"embed_dim": 8, "depths": [1, 1], "num_heads": [1, 1],
        ...                  "window_size": 4},
        ...     hidden_dim=8, num_gru_layers=1, projection_dims=(16, 8),
        ... )
        >>> out = model(**next(iter(get_dataloader(dataset, batch_size=4))))
        >>> out["y_prob"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        image_type: str = "line",
        image_size: int = 384,
        image_model: str | None = SWIN_BASE_IN22K,
        swin_config: dict[str, Any] | None = None,
        time_range: tuple[float, float] = (0.0, 48.0),
        value_range: tuple[float, float] = (-3.0, 3.0),
        hidden_dim: int = 128,
        num_gru_layers: int = 4,
        projection_dims: tuple[int, ...] = (1024, 512, 1024),
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta1: float = 0.1,
        beta2: float = 0.1,
        temperature: float = 0.1,
        margin: float = 0.2,
        num_clusters: int = 8,
        focal_gamma: float = 2.0,
        reconstruction: str = "nt_xent",
        value_key: str = "x",
        time_key: str = "x_time",
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("JointSeqImage expects exactly one label field.")
        if image_type not in IMAGE_TYPES:
            raise ValueError(f"image_type must be one of {IMAGE_TYPES}")
        if reconstruction not in ("nt_xent", "mse"):
            raise ValueError('reconstruction must be "nt_xent" or "mse"')
        self.label_key = self.label_keys[0]
        if self.mode not in ("binary", "multiclass"):
            raise ValueError("JointSeqImage supports binary and multiclass labels.")
        self.value_key, self.time_key = value_key, time_key
        sample = dataset[0]
        length, n_vars = (int(d) for d in torch.as_tensor(sample[value_key]).shape)
        self.num_variables = n_vars
        self.sequence_length = length
        rows, cols = grid_shape(n_vars)
        self.cell_size = image_size // max(rows, cols)
        if self.cell_size < 1:
            raise ValueError("image_size is too small for the number of variables.")
        self.image_type = image_type
        self.time_range = tuple(time_range)
        self.value_range = tuple(value_range)
        self.alpha, self.beta1, self.beta2 = alpha, beta1, beta2
        self.temperature, self.margin = temperature, margin
        self.num_clusters = num_clusters
        self.focal_gamma = focal_gamma
        self.reconstruction = reconstruction

        self.forward_imputer = DecayImputer(n_vars, hidden_dim, num_gru_layers)
        self.backward_imputer = DecayImputer(n_vars, hidden_dim, num_gru_layers)
        self.discriminator = MaskDiscriminator(n_vars)
        joint_dim = projection_dims[-1]
        self.sequence_norm = nn.LayerNorm(length * n_vars)
        self.sequence_feature = nn.Sequential(
            nn.Linear(length * n_vars, joint_dim), nn.Dropout(dropout)
        )

        from transformers import SwinConfig, SwinModel

        if image_model is not None:
            self.image_encoder = SwinModel.from_pretrained(image_model)
        else:
            self.image_encoder = SwinModel(SwinConfig(**(swin_config or {})))
        image_dim = self.image_encoder.config.hidden_size
        self.register_buffer("pixel_mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        self.sequence_projection = _mlp((joint_dim, *projection_dims), dropout)
        self.image_projection = _mlp((image_dim, *projection_dims), dropout)
        output_size = self.label_processor.output_size()
        self.classifier = nn.Linear(2 * joint_dim, output_size)
        self.register_buffer("cluster_centers", torch.empty(0, 2 * joint_dim))
        self._epoch_features: list[torch.Tensor] = []

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs) -> None:
        # The centres are empty until the first update_clusters(); take the
        # saved shape, so a checkpoint loads into a freshly built model.
        centers = state_dict.get(prefix + "cluster_centers")
        if centers is not None:
            self.cluster_centers = self.cluster_centers.new_empty(centers.shape)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def generator_parameters(self) -> list[nn.Parameter]:
        """Parameters of the imputing generator (both directions)."""
        return [*self.forward_imputer.parameters(), *self.backward_imputer.parameters()]

    def _inputs(self, kwargs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        """``(values, times, mask, valid)``, cut to the batch's longest series.

        The steps after every sample's last real time are padding for all of
        them; dropping them leaves every output unchanged and saves the
        recurrences those steps.
        """
        x = kwargs[self.value_key].to(self.device).float()
        times = kwargs[self.time_key].to(self.device).float()
        valid = torch.isfinite(times)
        length = max(int(valid.sum(1).max()), 1)
        x, times, valid = x[:, :length], times[:, :length], valid[:, :length]
        mask = (torch.isfinite(x) & valid.unsqueeze(-1)).float()
        values = torch.nan_to_num(x) * mask
        return values, times, mask, valid

    def impute(
        self, values: torch.Tensor, times: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Runs the generator; returns ``(imputed, forward_hat, backward_hat)``.

        ``imputed`` is ``X'`` (observed values kept, the mean of the two
        directions' reconstructions elsewhere, 0 at padding steps); the
        reconstructions are ``[B, L, V]``.
        """
        lengths = valid.sum(1)
        forward_hat = self.forward_imputer(values, mask, time_lags(times, mask))
        rev_values = reverse_padded(values, lengths)
        rev_mask = reverse_padded(mask, lengths)
        rev_times = reverse_padded(times, lengths)
        # Reversed times run backwards; their gaps are made positive by negation.
        rev_times = torch.where(torch.isfinite(rev_times), -rev_times, rev_times)
        backward_hat = reverse_padded(
            self.backward_imputer(rev_values, rev_mask, time_lags(rev_times, rev_mask)),
            lengths,
        )
        steps = valid.unsqueeze(-1).float()
        forward_hat, backward_hat = forward_hat * steps, backward_hat * steps
        imputed = mask * values + (1 - mask) * (forward_hat + backward_hat) / 2
        return imputed * steps, forward_hat, backward_hat

    def images(
        self, values: torch.Tensor, times: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """The rendered images ``[B, 3, H, W]`` in [0, 1], before normalization."""
        return render_images(
            values, times, mask, self.image_type, self.cell_size,
            self.time_range, self.value_range,
        )

    def discriminator_loss(self, **kwargs: Any) -> torch.Tensor:
        """Eq. 8: BCE of the discriminator on the (detached) imputation."""
        values, times, mask, valid = self._inputs(kwargs)
        with torch.no_grad():
            imputed = self.impute(values, times, mask, valid)[0]
        return self._discriminator_loss(imputed, mask, valid)

    def _discriminator_loss(
        self, imputed: torch.Tensor, mask: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        logits = self.discriminator(imputed.detach())
        weight = valid.unsqueeze(-1).expand_as(mask).float()
        return F.binary_cross_entropy_with_logits(logits, mask, weight=weight, reduction="sum") / weight.sum()

    def generator_loss(self, **kwargs: Any) -> torch.Tensor:
        """``L_adv + alpha * L_rec`` (Eqs. 5-7) for the generator step.

        ``L_adv`` asks the discriminator to call the imputed entries
        observed. ``L_rec`` is NT-Xent between the two directions'
        reconstructions and between the input and its imputation (or, with
        ``reconstruction="mse"``, the MSE on the observed entries).
        """
        values, times, mask, valid = self._inputs(kwargs)
        return self._generator_loss(values, mask, valid, *self.impute(values, times, mask, valid))

    def _generator_loss(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        valid: torch.Tensor,
        imputed: torch.Tensor,
        forward_hat: torch.Tensor,
        backward_hat: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.discriminator(imputed)
        missing = (1 - mask) * valid.unsqueeze(-1).float()
        adversarial = (
            F.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits), weight=missing, reduction="sum"
            )
            / missing.sum().clamp(min=1)
        )
        batch = len(values)
        if self.reconstruction == "nt_xent":
            rec = nt_xent(
                forward_hat.reshape(batch, -1), backward_hat.reshape(batch, -1), self.temperature
            ) + nt_xent(values.reshape(batch, -1), imputed.reshape(batch, -1), self.temperature)
        else:
            observed = mask.sum().clamp(min=1)
            rec = (
                ((forward_hat - values) ** 2 * mask).sum()
                + ((backward_hat - values) ** 2 * mask).sum()
            ) / observed
        return adversarial + self.alpha * rec

    def encode(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Projected sequence and image features, each ``[B, joint_dim]``."""
        values, times, mask, valid = self._inputs(kwargs)
        imputed = self.impute(values, times, mask, valid)[0]
        return self._encode(values, times, mask, imputed)

    def _encode(
        self,
        values: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        imputed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if imputed.shape[1] > self.sequence_length:
            raise ValueError(
                f"series of {imputed.shape[1]} steps is longer than the "
                f"{self.sequence_length} the model was built for; pad every sample "
                "to the same length (as PhysioNet2012Mortality does)"
            )
        # Pad back to the full length, so the feature sees the whole grid.
        imputed = F.pad(imputed, (0, 0, 0, self.sequence_length - imputed.shape[1]))
        s = self.sequence_feature(self.sequence_norm(imputed.flatten(1)))
        pixels = (self.images(values, times, mask) - self.pixel_mean) / self.pixel_std
        v = self.image_encoder(pixel_values=pixels).pooler_output
        return self.sequence_projection(s), self.image_projection(v)

    def cluster_loss(self, u: torch.Tensor) -> torch.Tensor:
        """Eq. 11: contrast each joint feature with the k-means centres.

        The positive is the nearest centre (Eq. 10). 0 before the first
        :meth:`update_clusters`.
        """
        if len(self.cluster_centers) == 0:
            return u.sum() * 0
        nearest = torch.cdist(u.detach(), self.cluster_centers).argmin(1)
        sim = F.normalize(u, dim=-1) @ F.normalize(self.cluster_centers, dim=-1).T
        return F.cross_entropy(sim / self.temperature, nearest)

    @torch.no_grad()
    def update_clusters(self, max_iter: int = 100, tol: float = 1e-4) -> None:
        """Recomputes the cluster centres from this epoch's joint features.

        k-means on the joint features of the :meth:`train_step` calls since
        the last call, starting from the current centres (or from ``k``
        random features the first time). Each centre is the member feature
        nearest the members' mean, the minimiser of Eq. 12. Stops when the
        centres move less than ``tol`` in total (the paper's threshold).
        Clears the stored features.
        """
        if not self._epoch_features:
            return
        features = torch.cat(self._epoch_features).to(self.cluster_centers.device)
        self._epoch_features = []
        k = min(self.num_clusters, len(features))
        if len(self.cluster_centers) == k:
            centers = self.cluster_centers.clone()
        else:
            generator = torch.Generator().manual_seed(0)
            centers = features[torch.randperm(len(features), generator=generator)[:k].to(features.device)]
        for _ in range(max_iter):
            assign = torch.cdist(features, centers).argmin(1)
            new = centers.clone()
            for j in range(k):
                members = features[assign == j]
                if len(members):
                    new[j] = members[(members - members.mean(0)).norm(dim=1).argmin()]
            shift = (new - centers).norm(dim=1).sum()
            centers = new
            if shift < tol:
                break
        self.cluster_centers = centers

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Classifies the batch; with labels, the joint-step loss.

        Returns:
            A dict with ``logit``, ``y_prob`` and the joint feature
            ``embed``; with labels also ``loss`` (``L_clf + beta1 * L_cont
            + beta2 * L_cluster``) and ``y_true``.
        """
        values, times, mask, valid = self._inputs(kwargs)
        imputed = self.impute(values, times, mask, valid)[0]
        return self._classify(kwargs, values, times, mask, imputed)

    def _classify(
        self,
        kwargs: dict[str, Any],
        values: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor,
        imputed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        s, v = self._encode(values, times, mask, imputed)
        u = torch.cat([s, v], dim=-1)
        logit = self.classifier(u)
        out = {"logit": logit, "y_prob": self.label_processor.activation(logit), "embed": u}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            binary = self.mode == "binary"
            labels = y_true.view(-1).long()
            loss = focal_loss(logit, y_true, self.focal_gamma, binary)
            loss = loss + self.beta1 * margin_contrastive(s, v, labels, self.temperature, self.margin)
            loss = loss + self.beta2 * self.cluster_loss(u)
            out["loss"] = loss
            out["y_true"] = y_true
        return out

    def train_step(
        self,
        batch: dict[str, Any],
        optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        generator_optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """One batch of training.

        With ``generator_optimizer``, the paper's three steps, each with its
        own forward pass: update the discriminator, then the generator on
        ``L_adv + alpha * L_rec``, then everything but the discriminator on
        the joint loss.

        Without it, a faster variant that runs the generator once: the
        discriminator is updated on that imputation, then everything but
        the discriminator on the sum of the joint and generator losses. The
        generator's two updates become one, so it costs about half as much
        (the recurrent generator dominates the cost of a step).

        Args:
            batch: A batch from :func:`~pyhealth.datasets.get_dataloader`.
            optimizer: Over all parameters except the discriminator's.
            discriminator_optimizer: Over ``self.discriminator``.
            generator_optimizer: Over :meth:`generator_parameters`, or
                ``None`` for the single-pass variant.

        Returns:
            The discriminator, generator and joint losses as floats. The
            batch's joint features are stored for :meth:`update_clusters`.
        """
        self.train()
        if generator_optimizer is not None:
            discriminator_optimizer.zero_grad()
            d_loss = self.discriminator_loss(**batch)
            d_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad()
            g_loss = self.generator_loss(**batch)
            g_loss.backward()
            generator_optimizer.step()

            optimizer.zero_grad()
            out = self(**batch)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
        else:
            values, times, mask, valid = self._inputs(batch)
            imputed, forward_hat, backward_hat = self.impute(values, times, mask, valid)
            discriminator_optimizer.zero_grad()
            d_loss = self._discriminator_loss(imputed, mask, valid)
            d_loss.backward()
            discriminator_optimizer.step()

            optimizer.zero_grad()
            g_loss = self._generator_loss(values, mask, valid, imputed, forward_hat, backward_hat)
            out = self._classify(batch, values, times, mask, imputed)
            loss = out["loss"]
            (loss + g_loss).backward()
            optimizer.step()
        # Kept on the CPU for update_clusters (about 8 KB per sample).
        self._epoch_features.append(out["embed"].detach().cpu())
        return {
            "discriminator": d_loss.item(), "generator": g_loss.item(), "joint": loss.item()
        }
