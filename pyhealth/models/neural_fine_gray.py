"""Monotonic Neural Fine-Gray with an exact competing-risk likelihood.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
Architecture adapted from the official MIT-licensed Paper release.
Copyright (c) 2021 Vincent Jeanselme.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from itertools import pairwise
from typing import Any, Literal

import torch
from torch import Tensor, nn

from ..datasets import SampleDataset
from ..processors import LabelProcessor
from .base_model import BaseModel

MaskMode = Literal["original", "constant", "observed", "augmented"]


class PositiveLinear(nn.Module):
    """Affine layer with nonnegative squared weights for monotonic time heads.

    Args:
        in_features: Positive input width.
        out_features: Positive output width.

    Examples:
        >>> layer = PositiveLinear(3, 2)
        >>> tuple(layer(torch.zeros(4, 3)).shape)
        (4, 2)
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if min(in_features, out_features) < 1:
            raise ValueError("Layer dimensions must be positive")
        self.raw_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.xavier_uniform_(self.raw_weight)
        with torch.no_grad():
            self.raw_weight.abs_().sqrt_()
        bound = in_features**-0.25
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: Tensor) -> Tensor:
        """Transform features using squared weights.

        Args:
            features: Tensor with final dimension in_features.

        Returns:
            Tensor with final dimension out_features.
        """
        return nn.functional.linear(features, self.raw_weight.square(), self.bias)


def _mlp(dimensions: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    """Build the model's reference Tanh encoder or event-allocation network.

    Args:
        dimensions: Input width followed by output widths for the linear layers.
        dropout: Dropout probability after each linear layer.

    Returns:
        Sequential linear, optional dropout, and Tanh layers.
    """
    modules: list[nn.Module] = []
    for before, after in pairwise(dimensions):
        modules.append(nn.Linear(before, after))
        if dropout:
            modules.append(nn.Dropout(dropout))
        modules.append(nn.Tanh())
    return nn.Sequential(*modules)


def _positive_head(dimensions: Sequence[int]) -> nn.Sequential:
    """Build a time-monotonic head for the model's conditional survival.

    Args:
        dimensions: Input width followed by hidden widths and the final width.

    Returns:
        Squared-weight layers with Tanh hidden activations and a positive
        Softplus output. No dropout is applied to these time-derivative heads.
    """
    modules: list[nn.Module] = []
    for position, (before, after) in enumerate(pairwise(dimensions)):
        modules.extend(
            [
                PositiveLinear(before, after),
                nn.Softplus() if position == len(dimensions) - 2 else nn.Tanh(),
            ]
        )
    return nn.Sequential(*modules)


class NeuralFineGray(BaseModel):
    """Direct cumulative incidence with autograd event densities.

    CIFs are b_r*(1-exp(-t*m_r(t,h))), with softmax b and positive, monotonic
    m. Labels are 0=censored, 1..num_risks=observed first event. Trainer evaluates
    the joint first-event loss only; use survival-specific metrics separately.
    Evaluation respects the caller's gradient context. Labeled no_grad
    evaluation enables local time differentiation; labeled
    inference_mode is unsupported. Density derivatives use the reference
    1e-10 floor for log-domain numerical safety.

    Args:
        dataset: PyHealth SampleDataset, or None for direct array/tensor use.
        input_dim: Positive clinical width, excluding mask channels.
        num_risks: Positive number of competing outcomes.
        hidden_dims: Positive encoder/balance hidden widths.
        survival_dims: Positive monotonic-head hidden widths.
        dropout: Encoder dropout in [0, 1).
        mask_mode: Original features, constant-zero masks, observed masks,
            or observed masks plus training-time augmentation.
        augmentation_rate: MCAR training masking probability in [0, 1].
        mask_indices: Eligible clinical indices; None selects every feature.
        dtype: Floating parameter/likelihood precision; float64 matches reference.

    Examples:
        >>> model = NeuralFineGray(None, 3, hidden_dims=(4,), survival_dims=(4,))
        >>> model.eval()  # doctest: +ELLIPSIS
        NeuralFineGray(...)
        >>> tuple(model.predict_cif(torch.zeros(2, 3), [0., 1.]).shape)
        (2, 2, 2)
    """

    def __init__(
        self,
        dataset: SampleDataset | None,
        input_dim: int,
        num_risks: int = 2,
        hidden_dims: Sequence[int] = (25,),
        survival_dims: Sequence[int] = (25,),
        dropout: float = 0.0,
        mask_mode: MaskMode = "original",
        augmentation_rate: float = 0.1,
        mask_indices: Sequence[int] | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(dataset)
        if any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            for width in (input_dim, num_risks, *hidden_dims, *survival_dims)
        ):
            raise ValueError("All dimensions must be positive integers")
        if mask_mode not in {"original", "constant", "observed", "augmented"}:
            raise ValueError("Unknown mask_mode")
        if not 0 <= dropout < 1 or not 0 <= augmentation_rate <= 1:
            raise ValueError("Invalid dropout or augmentation probability")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be float32 or float64")
        indices = list(range(input_dim)) if mask_indices is None else list(mask_indices)
        if len(set(indices)) != len(indices) or any(
            not isinstance(index, int) or not 0 <= index < input_dim
            for index in indices
        ):
            raise ValueError("mask_indices must be unique clinical indices")
        self.input_dim, self.num_risks = input_dim, num_risks
        self.mask_mode, self.augmentation_rate = mask_mode, augmentation_rate
        self.register_buffer("mask_indices", torch.tensor(indices, dtype=torch.long))
        embedding_input = input_dim if mask_mode == "original" else 2 * input_dim
        self.embed = _mlp([embedding_input, *hidden_dims, input_dim], dropout)
        self.balance = _mlp([input_dim, *hidden_dims, num_risks])
        self.outcome = nn.ModuleList(
            _positive_head([input_dim + 1, *survival_dims, 1]) for _ in range(num_risks)
        )
        self.to(dtype=dtype)

    @property
    def label_processors(self) -> dict[str, LabelProcessor]:
        """Use the joint first-event loss rather than classification metrics."""
        return {}

    def _tensor(self, value: Any) -> Tensor:
        return torch.as_tensor(
            value,
            dtype=self.embed[0].weight.dtype,
            device=self.device,
        )

    def _embedding(self, features: Any, missing_mask: Any) -> Tensor:
        features = self._tensor(features)
        if (
            features.ndim != 2
            or features.shape[0] == 0
            or features.shape[1] != self.input_dim
            or not torch.isfinite(features).all()
        ):
            raise ValueError("features must be finite, nonempty [batch, input_dim]")
        mask = None if missing_mask is None else self._tensor(missing_mask)
        if mask is not None and (
            mask.shape != features.shape or not ((mask == 0) | (mask == 1)).all()
        ):
            raise ValueError("missing_mask must be binary and match features")
        if self.mask_mode == "original":
            return self.embed(features)
        if mask is None:
            if self.mask_mode != "constant":
                raise ValueError("Observed/augmented modes require missing_mask")
            mask = torch.zeros_like(features)
        if self.mask_mode == "constant":
            mask = torch.zeros_like(features)
        elif self.mask_mode == "augmented" and self.training:
            injected = torch.zeros_like(features, dtype=torch.bool)
            injected[:, self.mask_indices] = (
                torch.rand_like(features[:, self.mask_indices]) < self.augmentation_rate
            )
            features = features.masked_fill(injected, 0.0)
            mask = torch.maximum(mask, injected.to(mask.dtype))
        return self.embed(torch.cat((features, mask), dim=1))

    def forward(
        self,
        features: Any,
        time: Any = None,
        event: Any = None,
        missing_mask: Any = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Compute subject-time CIFs and optional exact negative log likelihood.

        Args:
            features: Finite imputed/scaled array-like [batch, input_dim].
            time: One finite nonnegative time per subject in preprocessing units.
            event: One integer label per subject, or None for prediction only.
            missing_mask: Binary indicators, 1=missing; required by mask-aware arms.
            **kwargs: Unused PyHealth sample metadata.

        Returns:
            cif and y_prob [batch, risks], survival [batch]; labeled batches also
            return scalar loss and y_true [batch, 2] containing time/event.

        Raises:
            ValueError: Features, times, events, or masks are invalid.
            RuntimeError: Labeled inference_mode cannot differentiate time.
            FloatingPointError: Hazard, density, or likelihood is nonfinite.
        """
        if time is None:
            raise ValueError("forward requires time; use predict_cif for a grid")
        if event is not None and torch.is_inference_mode_enabled():
            raise RuntimeError("Labeled NLL requires no_grad, not inference_mode")
        # no_grad validation still needs local time derivatives.
        caller_graph = torch.is_grad_enabled()
        with torch.enable_grad() if event is not None else nullcontext():
            representation = self._embedding(features, missing_mask)
            times = self._tensor(time).reshape(-1)
            if (
                len(times) != len(representation)
                or not torch.isfinite(times).all()
                or (times < 0).any()
            ):
                raise ValueError("time must match batch and be finite/nonnegative")
            raw_events = None if event is None else self._tensor(event).reshape(-1)
            if raw_events is not None and (
                len(raw_events) != len(times)
                or not torch.isfinite(raw_events).all()
                or (raw_events != raw_events.round()).any()
                or (raw_events < 0).any()
                or (raw_events > self.num_risks).any()
            ):
                raise ValueError("event must match batch and contain integers 0..risks")
            log_beta = torch.log_softmax(self.balance(representation), dim=1)
            hazards, derivatives = [], []
            for head in self.outcome:
                # A separate time leaf keeps one head's derivative graph independent.
                tau = times.clone()
                if event is not None:
                    tau.requires_grad_(True)
                hazard = tau * head(
                    torch.cat((representation, tau[:, None]), dim=1)
                ).squeeze(1)
                hazards.append(hazard)
                if event is not None:
                    derivative = torch.autograd.grad(
                        hazard.sum(),
                        tau,
                        create_graph=caller_graph,
                        retain_graph=caller_graph,
                    )[0]
                    if not torch.isfinite(derivative).all() or (derivative < 0).any():
                        raise FloatingPointError("Invalid monotonic density derivative")
                    derivatives.append(derivative.clamp_min(1e-10).log())
            cumulative = torch.stack(hazards, dim=1)
            if not torch.isfinite(cumulative).all():
                raise FloatingPointError("Nonfinite cumulative hazard")
            cif = log_beta.exp() * (-torch.expm1(-cumulative))
            log_survival = torch.logsumexp(log_beta - cumulative, dim=1)
            result = {"cif": cif, "y_prob": cif, "survival": log_survival.exp()}
            if raw_events is not None:
                log_density = log_beta - cumulative + torch.stack(derivatives, dim=1)
                selected = log_density.gather(
                    1,
                    (raw_events.long() - 1).clamp_min(0)[:, None],
                ).squeeze(1)
                likelihood = torch.where(raw_events == 0, log_survival, selected)
                if not torch.isfinite(likelihood).all():
                    raise FloatingPointError("Nonfinite competing-risk likelihood")
                result["loss"] = -likelihood.mean()
                result["y_true"] = torch.stack((times, raw_events), dim=1)
        return (
            result
            if caller_graph
            else {key: value.detach() for key, value in result.items()}
        )

    def predict_cif(
        self,
        features: Any,
        horizons: Any,
        missing_mask: Any = None,
    ) -> Tensor:
        """Predict CIFs on a nonnegative time grid without time differentiation.

        Call eval() to disable dropout and training augmentation. Gradients
        follow the caller's context; use no_grad() for numerical inference.

        Args:
            features: Imputed/scaled clinical [batch, input_dim].
            horizons: Finite, nonnegative grid in preprocessing time units.
            missing_mask: Binary indicators matching clinical features.

        Returns:
            CIF tensor [batch, horizons, risks], in the supplied horizon order.
            Empty grids retain a zero-length horizon dimension.

        Raises:
            ValueError: Features, masks, or horizons are invalid.
        """
        features = self._tensor(features)
        self._embedding(features, missing_mask)
        horizons = self._tensor(horizons).reshape(-1)
        if not torch.isfinite(horizons).all() or (horizons < 0).any():
            raise ValueError("horizons must be finite and nonnegative")
        curves = [
            self(features, horizon.expand(len(features)), missing_mask=missing_mask)[
                "cif"
            ]
            for horizon in horizons
        ]
        return (
            torch.stack(curves, dim=1)
            if curves
            else features.new_empty((len(features), 0, self.num_risks))
        )
