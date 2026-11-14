# Parts of this module are ported from https://github.com/zengsihang/TrajSurv,
# which is distributed under the following license:
#
# MIT License
#
# Copyright (c) 2026 Sihang Zeng
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""TrajSurv: neural controlled differential equations for survival.

Implements TrajSurv from Zeng et al., "TrajSurv: Learning Continuous Latent
Trajectories from Electronic Health Records for Trustworthy Survival
Prediction" (MLHC 2025, https://arxiv.org/abs/2508.00657). A neural
controlled differential equation (CDE) turns a patient's irregular
multivariate time series into a continuous latent trajectory; a Cox head on
the final latent state gives the risk, and an optional contrastive term
aligns the latent states with a clinical progression signal (the paper uses
the SOFA score and its trend).

Ported with attribution from the authors' code
(https://github.com/zengsihang/TrajSurv, MIT License, Copyright (c) 2026
Sihang Zeng; the full license is in the comment above). Differences from
that code:

- The CDE is solved in pure PyTorch (a natural cubic spline of the input
  path and a fixed-step Runge-Kutta 4 solver), so ``torchcde`` is not
  needed. The authors' code uses ``torchcde.CubicSpline`` and ``cdeint``.
- The Cox partial likelihood is averaged over the batch's events instead of
  summed, so the loss scale does not depend on the batch size.
- The contrastive term is a Rank-N-Contrast loss (Zha et al., NeurIPS 2023)
  written here; the authors import it from a separate Rank-N-Contrast
  checkout. "Time-aware" is implemented by adding the time difference,
  scaled by ``alignment_time_weight``, to the label distance.
- The DSM-Weibull head, the clustering and the vector-field analysis are
  not ported.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.survival_mdn import _value_tensor as _first
from pyhealth.processors import LabelProcessor

__all__ = ["TrajSurv", "cox_partial_likelihood_loss", "rank_n_contrast_loss"]


# --------------------------------------------------------------------- losses
def cox_partial_likelihood_loss(risk: Tensor, duration: Tensor, event: Tensor) -> Tensor:
    """Negative Cox partial log-likelihood (Breslow ties), mean over events.

    Args:
        risk: Log hazard ratios, shape ``[n]``.
        duration: Observed times, shape ``[n]``.
        event: 1 for an observed event, 0 for censoring, shape ``[n]``.

    Returns:
        A scalar loss; 0 (with gradient) when the batch has no event.

    Examples:
        >>> import torch
        >>> loss = cox_partial_likelihood_loss(
        ...     torch.tensor([0.0, 0.0]), torch.tensor([1.0, 2.0]),
        ...     torch.tensor([1.0, 0.0]))
        >>> round(loss.item(), 4)  # log(2): two at risk at the first event
        0.6931
    """
    risk = risk.reshape(-1)
    duration = duration.reshape(-1)
    event = event.reshape(-1).to(risk.dtype)
    at_risk = duration[None, :] >= duration[:, None]  # [i, j]: j at risk at t_i
    masked = risk[None, :].expand_as(at_risk).masked_fill(~at_risk, float("-inf"))
    log_denominator = torch.logsumexp(masked, dim=1)
    n_events = event.sum()
    if n_events == 0:
        return risk.sum() * 0.0
    return -((risk - log_denominator) * event).sum() / n_events


def rank_n_contrast_loss(
    features: Tensor, labels: Tensor, temperature: float = 2.0
) -> Tensor:
    """Rank-N-Contrast loss: feature similarity ordered like label distance.

    For an anchor ``i`` and another sample ``j``, the loss is
    ``-log(exp(s_ij) / sum_k exp(s_ik))`` over the samples ``k`` at least as
    far from ``i`` in label space as ``j`` (Zha et al., NeurIPS 2023), with
    ``s`` the negative L2 feature distance divided by ``temperature`` and
    label distance in L1.

    Args:
        features: Shape ``[n, d]``.
        labels: Shape ``[n, k]``.
        temperature: Similarity temperature.

    Returns:
        A scalar loss (0 for fewer than 3 samples).

    Examples:
        >>> import torch
        >>> z = torch.tensor([[0.0], [1.0], [2.0]])
        >>> y = torch.tensor([[0.0], [1.0], [2.0]])
        >>> ordered = rank_n_contrast_loss(z, y)
        >>> shuffled = rank_n_contrast_loss(z, y[[0, 2, 1]])
        >>> bool(ordered < shuffled)
        True
    """
    n = features.shape[0]
    if n < 3:
        return features.sum() * 0.0
    logits = -torch.cdist(features, features, p=2) / temperature
    label_diff = torch.cdist(labels.to(features.dtype), labels.to(features.dtype), p=1)
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=features.device)
    logits = logits[off_diagonal].view(n, n - 1)
    label_diff = label_diff[off_diagonal].view(n, n - 1)
    # Sort each anchor's others by label distance; the denominator for the
    # j-th closest is a log-sum-exp over it and everything farther.
    order = torch.argsort(label_diff, dim=1)
    sorted_logits = torch.gather(logits, 1, order)
    suffix = torch.flip(torch.logcumsumexp(torch.flip(sorted_logits, [1]), dim=1), [1])
    return (suffix - sorted_logits).mean()


# --------------------------------------------------------------- CDE solving
def _natural_cubic_second_derivatives(
    x: Tensor, lengths: Tensor | None = None
) -> Tensor:
    """Second derivatives of the natural cubic spline through unit-spaced knots.

    Each series is fitted on its own knots only, so the natural end
    condition sits at its last real knot and padding after it has no effect.

    Args:
        x: Knot values, shape ``[B, T, C]``.
        lengths: Knots per series, shape ``[B]``; all ``T`` when omitted.

    Returns:
        Shape ``[B, T, C]``, zero at each series' ends and after them.
    """
    batch, steps, _ = x.shape
    if lengths is None:
        lengths = torch.full((batch,), steps, device=x.device)
    second = torch.zeros_like(x)
    for length in torch.unique(lengths).tolist():
        length = int(length)
        if length < 3:
            continue
        rows = torch.nonzero(lengths == length, as_tuple=True)[0]
        knots = x[rows, :length]
        n = length - 2
        system = (
            4.0 * torch.eye(n, dtype=x.dtype, device=x.device)
            + torch.diag(torch.ones(n - 1, dtype=x.dtype, device=x.device), 1)
            + torch.diag(torch.ones(n - 1, dtype=x.dtype, device=x.device), -1)
        )
        rhs = 6.0 * (knots[:, 2:] - 2.0 * knots[:, 1:-1] + knots[:, :-2])
        second[rows, 1 : length - 1] = torch.linalg.solve(system, rhs)
    return second


class _CubicPath:
    """Derivative of a natural cubic (or linear) spline on unit-spaced knots."""

    def __init__(
        self, x: Tensor, cubic: bool = True, lengths: Tensor | None = None
    ) -> None:
        self.delta = x[:, 1:] - x[:, :-1]  # [B, T-1, C]
        second = (
            _natural_cubic_second_derivatives(x, lengths)
            if cubic
            else torch.zeros_like(x)
        )
        self.m_left = second[:, :-1]
        self.m_right = second[:, 1:]

    def derivative(self, segment: int, s: float) -> Tensor:
        """dX/dt on ``segment`` at fraction ``s`` in [0, 1], shape ``[B, C]``."""
        left = -(3.0 * (1.0 - s) ** 2 - 1.0) / 6.0
        right = (3.0 * s**2 - 1.0) / 6.0
        return (
            self.delta[:, segment]
            + left * self.m_left[:, segment]
            + right * self.m_right[:, segment]
        )


class _VectorField(nn.Module):
    """``f(z)`` of ``dz = f(z) dX``: maps ``[B, H]`` to ``[B, H, C]``."""

    def __init__(self, hidden: int, channels: int, width: int, layers: int) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("vector_field_layers must be at least 2")
        modules: list[nn.Module] = [nn.Linear(hidden, width), nn.ReLU()]
        for _ in range(layers - 2):
            modules += [nn.Linear(width, width), nn.ReLU()]
        modules += [nn.Linear(width, hidden * channels), nn.Tanh()]
        self.net = nn.Sequential(*modules)
        self.hidden = hidden
        self.channels = channels

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).view(*z.shape[:-1], self.hidden, self.channels)


def _solve_cde(
    field: _VectorField,
    z0: Tensor,
    path: _CubicPath,
    lengths: Tensor,
    substeps: int,
) -> Tensor:
    """Fixed-step RK4 solve of ``dz = f(z) dX``; returns ``z`` at every knot.

    Segments at or after a sequence's last knot do not move its state.
    """
    states = [z0]
    z = z0
    h = 1.0 / substeps
    n_segments = path.delta.shape[1]
    for segment in range(n_segments):
        active = (segment < lengths - 1).to(z.dtype).unsqueeze(-1)  # [B, 1]

        def step(state: Tensor, s: float, seg: int = segment) -> Tensor:
            return torch.bmm(field(state), path.derivative(seg, s).unsqueeze(-1)).squeeze(-1)

        for sub in range(substeps):
            s0 = sub * h
            k1 = step(z, s0)
            k2 = step(z + 0.5 * h * k1, s0 + 0.5 * h)
            k3 = step(z + 0.5 * h * k2, s0 + 0.5 * h)
            k4 = step(z + h * k3, s0 + h)
            z = z + active * (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        states.append(z)
    return torch.stack(states, dim=1)


# ---------------------------------------------------------------------- model
class TrajSurv(BaseModel):
    """Neural-CDE survival model with continuous latent trajectories.

    The input is an hourly multivariate series, such as the ``dynamic``
    field of :class:`~pyhealth.tasks.InHospitalSurvivalMIMIC4`
    (a :class:`~pyhealth.processors.JaggedFeatureProcessor` series, already
    imputed and standardized by the processor) or a dense ``[T, F]`` tensor.
    Only single-level ``jagged`` series are supported; every series
    needs at least one record.
    The CDE's control path has ``1 + 2F`` channels: time (divided by
    ``time_scale``), the ``F`` values, and for each feature the fraction of
    steps observed so far (from the processor's missing-value mask; omitted
    without a mask). The natural cubic spline through the hourly knots is
    the continuous path, as in the paper.

    A Cox head on the last latent state gives ``risk`` (the log hazard
    ratio). Labels follow :class:`~pyhealth.models.SurvivalMDN`: output
    fields ``duration`` (positive time) and ``event`` (1 for an observed
    event, 0 for censoring), and ``y_true`` stacks them. ``y_prob`` holds
    survival curves ``S(t | x) = exp(-H0(t) exp(risk))`` on ``time_grid``.
    Call :meth:`fit_baseline_hazard` on the training data to estimate the
    Breslow baseline ``H0``; until then ``H0(t) = t`` is used, which ranks
    patients correctly but is not calibrated. The fitted baseline is part of
    the ``state_dict``, and it loads into a freshly built model.

    With ``alignment_key``, a Rank-N-Contrast loss (weight
    ``alignment_weight``) pulls together latent states whose alignment
    targets are close, as TrajSurv does with SOFA and the SOFA trend. The
    target is a dense ``[T, K]`` tensor field on the same hours as the
    series; the task in this module provides no SOFA series, so this is off
    by default. As in the authors' code, the contrastive term trains the
    projection head on detached latent states, and the Cox head reads the
    projected last state.

    Args:
        dataset: The dataset (used for the schemas and input width).
        feature_key: The series input field.
        duration_key: Output field with strictly positive durations.
        event_key: Output field with 0/1 event indicators.
        static_key: Optional static tensor field, concatenated to the path's
            first value to set the initial latent state.
        alignment_key: Optional ``[T, K]`` tensor field for the alignment loss.
        hidden_dim: Latent state width (64 in the paper).
        vector_field_width: Hidden width of the vector field MLP.
        vector_field_layers: Linear layers in the vector field (3 in the
            paper's best configuration).
        interpolation: ``"cubic"`` (natural cubic spline) or ``"linear"``.
        substeps: Runge-Kutta steps per hour.
        time_scale: Divides the time channel (hours / 36 by default).
        alignment_weight: Weight of the alignment loss.
        alignment_temperature: Temperature of the alignment loss.
        alignment_time_weight: Weight of the time difference in the
            alignment label distance ("time-aware"); 0 ignores time.
        alignment_samples: Latent states sampled per batch for the alignment
            loss, which is quadratic in their number.
        time_grid: Times at which ``y_prob`` gives survival probabilities.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": f"p{i}", "series": [[0.1 * i, 1.0]] * 4,
        ...      "duration": [1.0 + i], "event": [float(i % 2)]}
        ...     for i in range(4)
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"series": "tensor"},
        ...     output_schema={"duration": "tensor", "event": "tensor"},
        ...     dataset_name="toy",
        ... )
        >>> model = TrajSurv(dataset, feature_key="series", hidden_dim=8)
        >>> model.input_channels
        3

    References:
        Zeng, S., et al. TrajSurv: Learning Continuous Latent Trajectories
        from Electronic Health Records for Trustworthy Survival Prediction.
        Machine Learning for Healthcare, 2025.
    """

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """None: the survival outputs are scored by the model, not as a label,
        so the Trainer reports only the loss."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str = "dynamic",
        duration_key: str = "duration",
        event_key: str = "event",
        static_key: str | None = None,
        alignment_key: str | None = None,
        hidden_dim: int = 64,
        vector_field_width: int = 128,
        vector_field_layers: int = 3,
        interpolation: str = "cubic",
        substeps: int = 1,
        time_scale: float = 36.0,
        alignment_weight: float = 1.0,
        alignment_temperature: float = 2.0,
        alignment_time_weight: float = 0.0,
        alignment_samples: int = 256,
        time_grid: Tensor | None = None,
    ) -> None:
        super().__init__(dataset)
        for key in (feature_key, static_key, alignment_key):
            if key is not None and key not in dataset.input_schema:
                raise ValueError(f"{key!r} is not in dataset.input_schema")
        for key in (duration_key, event_key):
            if key not in dataset.output_schema:
                raise ValueError(f"output key {key!r} is not in dataset.output_schema")
        if interpolation not in ("cubic", "linear"):
            raise ValueError("interpolation must be 'cubic' or 'linear'")
        if substeps < 1:
            raise ValueError("substeps must be at least 1")
        self.feature_key = feature_key
        self.duration_key = duration_key
        self.event_key = event_key
        self.static_key = static_key
        self.alignment_key = alignment_key
        self.hidden_dim = hidden_dim
        self.interpolation = interpolation
        self.substeps = substeps
        self.time_scale = float(time_scale)
        self.alignment_weight = float(alignment_weight)
        self.alignment_temperature = float(alignment_temperature)
        self.alignment_time_weight = float(alignment_time_weight)
        self.alignment_samples = int(alignment_samples)

        self._roles = self._series_roles(dataset, feature_key)
        sample = dataset[0]
        values = self._sample_values(sample[feature_key])
        n_features = int(values.shape[-1])
        has_mask = "mask" in self._roles
        self.n_features = n_features
        self.input_channels = 1 + n_features * (2 if has_mask else 1)
        static_dim = 0
        if static_key is not None:
            static_dim = int(torch.as_tensor(_first(sample[static_key])).numel())

        self.initial = nn.Linear(self.input_channels + static_dim, hidden_dim)
        self.vector_field = _VectorField(
            hidden_dim, self.input_channels, vector_field_width, vector_field_layers
        )
        self.projection = (
            nn.Sequential(
                nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, hidden_dim)
            )
            if alignment_key is not None
            else nn.Identity()
        )
        self.cox = nn.Linear(hidden_dim, 1)

        if time_grid is None:
            time_grid = torch.linspace(0.0, 30.0, 61)
        self.register_buffer(
            "time_grid", torch.as_tensor(time_grid, dtype=torch.float32).reshape(-1)
        )
        # Breslow baseline cumulative hazard; empty means H0(t) = t.
        self.register_buffer("baseline_times", torch.zeros(0))
        self.register_buffer("baseline_cumhaz", torch.zeros(0))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ) -> None:
        # The Breslow baseline has one entry per event time, so its size is
        # only known after fit_baseline_hazard. Take the checkpoint's size so
        # a fitted (or unfitted) baseline loads into any TrajSurv.
        for name in ("baseline_times", "baseline_cumhaz"):
            incoming = state_dict.get(prefix + name)
            if incoming is not None:
                current = getattr(self, name)
                setattr(self, name, current.new_empty(incoming.shape))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------ input
    @staticmethod
    def _series_roles(dataset: SampleDataset, key: str) -> dict[str, int]:
        processor = dataset.input_processors.get(key)
        schema = getattr(processor, "schema", None)
        if not callable(schema):
            return {}
        roles = {}
        entries = schema()
        if any(isinstance(entry, tuple) and entry[0] != 0 for entry in entries):
            raise ValueError(
                f"{key!r} has more than one level; TrajSurv reads a single-level "
                "series (one record per time step)"
            )
        for index, entry in enumerate(entries):
            role = entry[1] if isinstance(entry, tuple) else entry
            # single-tensor processors call their output "value"
            roles.setdefault("values" if role == "value" else role, index)
        if "values" not in roles:
            raise ValueError(f"{key!r} has no values; TrajSurv needs numeric series")
        if any(isinstance(entry, tuple) for entry in entries):
            # JaggedFeatureProcessor's collate appends the offsets of the packed
            # records after the schema's tensors.
            roles.setdefault("offsets", len(entries))
        return roles

    def _sample_values(self, value) -> Tensor:
        if isinstance(value, tuple):
            return torch.as_tensor(value[self._roles.get("values", 0)])
        return torch.as_tensor(value)

    def _dense(self, value) -> tuple[Tensor, Tensor | None, Tensor]:
        """Batch field to padded ``values [B, T, F]``, ``observed`` and ``lengths``.

        Short sequences are padded with their last record, so their path is
        constant after the end. The spline is fitted per series length and
        the solver freezes the state at each series' last knot, so a series'
        risk does not depend on the other series in its batch.
        """
        if not isinstance(value, tuple) or "offsets" not in self._roles:
            values = torch.as_tensor(_first(value), dtype=torch.float32)
            if values.ndim == 2:
                values = values.unsqueeze(0)
            values = values.to(self.device)
            lengths = torch.full(
                (values.shape[0],), values.shape[1], device=self.device
            )
            return values, None, lengths
        flat = value[self._roles["values"]].to(self.device, torch.float32)
        flat = flat.reshape(flat.shape[0], -1)
        mask = value[self._roles["mask"]] if "mask" in self._roles else None
        offsets = value[self._roles["offsets"]].to(self.device)
        lengths = offsets[1:] - offsets[:-1]
        if len(lengths) == 0 or bool((lengths == 0).any()):
            raise ValueError("TrajSurv needs at least one record per series")
        steps = int(lengths.max().item())
        position = torch.arange(steps, device=self.device)
        index = offsets[:-1, None] + torch.minimum(
            position[None, :], (lengths - 1).clamp(min=0)[:, None]
        )
        values = flat[index]  # [B, T, F]
        observed = None
        if mask is not None:
            missing = mask.to(self.device).reshape(mask.shape[0], -1)[index]
            observed = (~missing).to(torch.float32)
        return values, observed, lengths

    def _path(self, value) -> tuple[Tensor, Tensor]:
        values, observed, lengths = self._dense(value)
        batch, steps, _ = values.shape
        time = torch.arange(steps, device=self.device, dtype=torch.float32)
        channels = [time.expand(batch, steps).unsqueeze(-1) / self.time_scale, values]
        if observed is not None:
            observed = observed * (time[None, :, None] < lengths[:, None, None])
            seen = torch.cumsum(observed, dim=1) / (time[None, :, None] + 1.0)
            channels.append(seen)
        return torch.cat(channels, dim=-1), lengths

    # ------------------------------------------------------------ model
    def trajectory(self, **kwargs) -> tuple[Tensor, Tensor]:
        """Latent states at every hour, ``[B, T, hidden_dim]``, and the lengths."""
        path, lengths = self._path(kwargs[self.feature_key])
        start = path[:, 0]
        if self.static_key is not None:
            static = torch.as_tensor(_first(kwargs[self.static_key])).to(
                self.device, torch.float32
            )
            start = torch.cat([start, static.reshape(static.shape[0], -1)], dim=-1)
        z0 = self.initial(start)
        if path.shape[1] < 2:
            return z0.unsqueeze(1), lengths
        spline = _CubicPath(
            path, cubic=self.interpolation == "cubic", lengths=lengths
        )
        states = _solve_cde(self.vector_field, z0, spline, lengths, self.substeps)
        return states, lengths

    def _risk_from_states(self, states: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        last = states[torch.arange(states.shape[0]), (lengths - 1).clamp(min=0)]
        embedding = self.projection(last)
        return self.cox(embedding).squeeze(-1), embedding

    def survival(self, risk: Tensor, times: Tensor) -> Tensor:
        """``S(t | x)`` for log hazard ratios ``risk`` at ``times``, ``[B, m]``."""
        times = times.to(risk.device, risk.dtype)
        if self.baseline_times.numel() == 0:
            cumhaz = times
        else:
            index = torch.searchsorted(self.baseline_times, times, right=True) - 1
            padded = torch.cat([self.baseline_cumhaz.new_zeros(1), self.baseline_cumhaz])
            cumhaz = padded[index + 1]
        return torch.exp(-cumhaz[None, :] * torch.exp(risk)[:, None])

    def _alignment_loss(self, states: Tensor, lengths: Tensor, target) -> Tensor:
        target = torch.as_tensor(_first(target)).to(self.device, torch.float32)
        if target.ndim == 2:
            target = target.unsqueeze(-1)
        steps = min(states.shape[1], target.shape[1])
        time = torch.arange(steps, device=self.device, dtype=torch.float32)
        valid = time[None, :] < lengths[:, None]
        features = self.projection(states[:, :steps].detach())[valid]
        labels = target[:, :steps][valid]
        if self.alignment_time_weight:
            hours = time.expand(states.shape[0], steps)[valid].unsqueeze(-1)
            labels = torch.cat([labels, self.alignment_time_weight * hours], dim=-1)
        if features.shape[0] > self.alignment_samples:
            pick = torch.randperm(features.shape[0], device=self.device)
            pick = pick[: self.alignment_samples]
            features, labels = features[pick], labels[pick]
        return rank_n_contrast_loss(features, labels, self.alignment_temperature)

    def forward(self, **kwargs) -> dict[str, Tensor]:
        """Runs TrajSurv on a batch.

        Args:
            **kwargs: Batch fields from the PyHealth dataloader; labels are
                optional at inference.

        Returns:
            ``risk`` and ``logit`` (log hazard ratio), ``y_prob`` (survival
            on ``time_grid``), and with labels ``loss`` and ``y_true``
            (``[duration, event]``). ``embed=True`` adds ``embed``, the
            (projected) last latent state.
        """
        states, lengths = self.trajectory(**kwargs)
        risk, embedding = self._risk_from_states(states, lengths)
        out = {
            "risk": risk,
            "logit": risk.unsqueeze(-1),
            "y_prob": self.survival(risk, self.time_grid),
        }
        if self.duration_key in kwargs and self.event_key in kwargs:
            duration = torch.as_tensor(_first(kwargs[self.duration_key])).to(
                self.device, torch.float32
            ).reshape(-1)
            event = torch.as_tensor(_first(kwargs[self.event_key])).to(
                self.device, torch.float32
            ).reshape(-1)
            loss = cox_partial_likelihood_loss(risk, duration, event)
            if self.alignment_key is not None and self.alignment_key in kwargs:
                loss = loss + self.alignment_weight * self._alignment_loss(
                    states, lengths, kwargs[self.alignment_key]
                )
            out["loss"] = loss
            out["y_true"] = torch.stack([duration, event], dim=-1)
        if kwargs.get("embed", False):
            out["embed"] = embedding
        return out

    @torch.no_grad()
    def fit_baseline_hazard(self, dataloader) -> None:
        """Estimates the Breslow baseline cumulative hazard from labelled data.

        Args:
            dataloader: Batches with the duration and event fields, usually
                the training data.
        """
        was_training = self.training
        self.eval()
        risks, durations, events = [], [], []
        try:
            for batch in dataloader:
                out = self(**batch)
                risks.append(out["risk"])
                durations.append(out["y_true"][:, 0])
                events.append(out["y_true"][:, 1])
        finally:
            self.train(was_training)
        risk = torch.cat(risks)
        duration = torch.cat(durations)
        event = torch.cat(events).bool()
        times = torch.unique(duration[event])
        hazard_ratio = torch.exp(risk)
        increments = torch.stack(
            [
                (event & (duration == t)).sum() / hazard_ratio[duration >= t].sum()
                for t in times
            ]
        ) if len(times) else torch.zeros(0, device=risk.device)
        self.baseline_times = times.to(self.baseline_times.device)
        self.baseline_cumhaz = torch.cumsum(increments, 0).to(self.baseline_times.device)
