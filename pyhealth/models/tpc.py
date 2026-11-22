"""TPC model for predicting time left in the ICU.

Contributor: Audrey Labrie.
Paper: Rocheteau, Lio, and Hyland (2021), Temporal Pointwise Convolutional
Networks for Length of Stay Prediction in the Intensive Care Unit.
Paper link: https://arxiv.org/abs/2007.09483
Reference code: https://github.com/EmmaRocheteau/TPC-LoS-prediction

MIT License
Copyright (c) 2020 Emma Rocheteau

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
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from pyhealth.datasets import SampleDataset

from .base_model import BaseModel


def masked_log_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Calculate squared log error for the hours marked True.

    Uses log(y), not log(1 + y). Values outside the mask are ignored.

    Args:
        predictions: Predicted days left, usually shaped (B, T).
        targets: Actual days left, with the same shape as predictions.
        valid_mask: Boolean mask with the same shape. True keeps a prediction.

    Returns:
        Mean squared log error over all kept predictions.

    Raises:
        ValueError: If shapes differ, the mask is empty, or kept values are
            zero, negative, NaN, or infinite.
    """
    if predictions.shape != targets.shape or targets.shape != valid_mask.shape:
        raise ValueError("Prediction, target and validity shapes must match")

    p, y = predictions[valid_mask], targets[valid_mask]
    if p.numel() == 0:
        raise ValueError("No valid prediction hours in this batch")

    if not bool(
        (torch.isfinite(p) & torch.isfinite(y) & (p > 0) & (y > 0)).all()
    ):
        raise ValueError(
            "Valid predictions and targets must be finite and positive"
        )
    return (p.log() - y.log()).square().mean()


class TPCLayer(nn.Module):
    """One TPC layer, using earlier hours and outputs from the previous layer.

    Args:
        groups: Number of feature groups.
        temporal_in: Temporal input channels per group.
        temporal_out: Temporal output channels per group.
        point_in: Number of pointwise inputs.
        point_out: Number of pointwise outputs per hour.
        kernel: Temporal kernel size.
        dilation: Temporal dilation rate.
        main_dropout: Pointwise dropout rate.
        temporal_dropout: Temporal dropout rate.
    """

    def __init__(
        self,
        groups: int,
        temporal_in: int,
        temporal_out: int,
        point_in: int,
        point_out: int,
        kernel: int,
        dilation: int,
        main_dropout: float,
        temporal_dropout: float,
    ) -> None:
        """Set up the temporal and pointwise layers."""
        super().__init__()
        self.groups = groups
        self.temporal_out = temporal_out
        self.point_out = point_out
        self.left_padding = (kernel - 1) * dilation

        self.temporal = nn.Conv1d(
            groups * temporal_in,
            groups * temporal_out,
            kernel_size=kernel,
            dilation=dilation,
            groups=groups,
        )
        self.point = nn.Linear(point_in, point_out)
        self.temporal_dropout = nn.Dropout(temporal_dropout)
        self.point_dropout = nn.Dropout(main_dropout)

    def forward(
        self,
        x: torch.Tensor,
        raw: torch.Tensor,
        repeated_static: torch.Tensor,
        point_skip: torch.Tensor,
        previous_temporal: Optional[torch.Tensor],
        previous_point: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one layer and keep the inputs needed by the next layer.

        Args:
            x: Temporal input shaped (B, groups * temporal_in, T).
            raw: Original inputs reshaped to (B * T, 2 * F + 2).
            repeated_static: Static features repeated each hour, (B * T, S).
            point_skip: Measurements and pointwise values saved from earlier layers.
            previous_temporal: Previous temporal output, or None for layer one.
            previous_point: Previous pointwise output, or None for layer one.

        Returns:
            The next input, saved values, temporal output, and pointwise output.
        """
        b, _, t = x.shape

        temporal = self.temporal_dropout(
            self.temporal(F.pad(x, (self.left_padding, 0)))
        )

        pieces = [
            value
            for value in (
                previous_temporal, previous_point, raw, repeated_static
            )
            if value is not None
        ]
        point = self.point_dropout(
            self.point(torch.cat(pieces, dim=1))
        )

        if previous_point is not None:
            point_skip = torch.cat([
                point_skip,
                previous_point.reshape(b, t, -1).transpose(1, 2),
            ], dim=1)

        temporal_and_skip = torch.cat([
            point_skip.unsqueeze(2),
            temporal.reshape(b, self.groups, self.temporal_out, t),
        ], dim=2)

        point_repeat = point.reshape(b, t, self.point_out).transpose(1, 2)
        point_repeat = point_repeat.unsqueeze(2).expand(
            -1, -1, self.temporal_out + 1, -1
        )

        combined = F.relu(
            torch.cat([temporal_and_skip, point_repeat], dim=1)
        )
        next_x = combined.reshape(b, -1, t)
        previous_temporal = temporal.transpose(1, 2).reshape(b * t, -1)

        return next_x, point_skip, previous_temporal, point


class TPC(BaseModel):
    """Predict time left in the ICU from hourly patient data.

    Uses temporal and pointwise layers. Normalization is off. Inputs must already
    be prepared and stored in a PyHealth dataset with raw processors.

    Input order: time since admission in days, F measurements, F indicators
    showing how recently each value was measured, and scaled hour of day.
    The eICU indicators are 1 for a new reading, 0 if never measured, and
    1 / (gap_hours * 4/3) otherwise. They are separate from valid_mask.

    Predictions range from about 1/48 to 100 days. Loss is mean squared log error.

    Args:
        dataset: PyHealth sample dataset with the fields used by forward.
        features: Number F of measurements, excluding time and recency indicators.
        static_dim: Number S of static features. Use 0 and an empty (B, 0)
            input tensor to leave them out.
        diagnosis_dim: Number D of diagnosis features. Use 0 and an empty
            (B, 0) input tensor to leave them out.
        kernel_size: Temporal kernel size for all layers.
        dilation_rates: One positive dilation rate per layer.
        temporal_channels: Temporal output channels per feature group.
        point_channels: New pointwise channels per layer.
        diagnosis_hidden: Hidden layer size for diagnosis features.
        final_hidden: Hidden layer size before the prediction.
        main_dropout: Dropout for the pointwise, diagnosis, and final layers.
        temporal_dropout: Dropout for temporal layers.

    Raises:
        ValueError: If sizes or dilation rates are invalid, or dropout is
            outside [0, 1).

    Examples:
        With a prepared dataset and padded batch::

            model = TPC(dataset, features=83, static_dim=64, diagnosis_dim=277)
            output = model(**batch)
            output["loss"].backward()

        See examples/eicu_remaining_los_tpc.py for a complete example.
    """

    @property
    def mode(self) -> str:
        """Tell PyHealth that this model predicts numbers.

        Returns:
            The string "regression".
        """
        return "regression"

    def __init__(
        self,
        dataset: SampleDataset,
        features: int,
        static_dim: int,
        diagnosis_dim: int,
        kernel_size: int = 4,
        dilation_rates: Sequence[int] = (1, 3, 6, 9, 12, 15, 18, 21, 24),
        temporal_channels: int = 12,
        point_channels: int = 13,
        diagnosis_hidden: int = 64,
        final_hidden: int = 17,
        main_dropout: float = 0.45,
        temporal_dropout: float = 0.05,
    ) -> None:
        """Set up the TPC layers and the final prediction layers."""
        super().__init__(dataset=dataset)

        sizes = (
            features, kernel_size, temporal_channels,
            point_channels, diagnosis_hidden, final_hidden,
        )
        if any(
            not isinstance(value, int) or value < 1 for value in sizes
        ):
            raise ValueError(
                "Feature/channel/kernel sizes must be positive integers"
            )
        if static_dim < 0 or diagnosis_dim < 0:
            raise ValueError(
                "Static and diagnosis dimensions must be nonnegative"
            )
        if not dilation_rates or any(
            not isinstance(d, int) or d < 1 for d in dilation_rates
        ):
            raise ValueError("Supply positive integer dilations explicitly")
        if not (
            0 <= main_dropout < 1 and 0 <= temporal_dropout < 1
        ):
            raise ValueError("Dropout probabilities must be in [0, 1)")

        self.features = features
        self.static_dim = static_dim
        self.diagnosis_dim = diagnosis_dim
        self.receptive_field = (
            1 + (kernel_size - 1) * sum(dilation_rates)
        )
        self.layers = nn.ModuleList()

        cumulative = previous_point = previous_temporal = 0
        for i, dilation in enumerate(dilation_rates):
            groups = features + cumulative
            point_in = (
                (groups - previous_point) * previous_temporal
                + previous_point + 2 * features + 2 + static_dim
            )
            self.layers.append(TPCLayer(
                groups,
                2 if i == 0 else 1 + previous_temporal,
                temporal_channels,
                point_in,
                point_channels,
                kernel_size,
                dilation,
                main_dropout,
                temporal_dropout,
            ))
            previous_temporal = temporal_channels
            previous_point = point_channels
            cumulative += point_channels

        output_channels = (
            (features + cumulative) * (1 + temporal_channels)
        )

        self.diagnosis_encoder = None
        if diagnosis_dim:
            self.diagnosis_encoder = nn.Sequential(
                nn.Linear(diagnosis_dim, diagnosis_hidden),
                nn.Dropout(main_dropout),
                nn.ReLU(),
            )

        head_in = (
            output_channels + static_dim
            + (diagnosis_hidden if diagnosis_dim else 0)
        )
        self.head = nn.Sequential(
            nn.Linear(head_in, final_hidden),
            nn.Dropout(main_dropout),
            nn.ReLU(),
            nn.Linear(final_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        static: torch.Tensor,
        diagnoses: torch.Tensor,
        valid_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Predict days left at each hour and calculate loss if targets are given.

        Args:
            x: Hourly data shaped (B, 2 * F + 2, T), padded at the end.
                B is batch size and T is the number of hours.
            static: Static features shaped (B, S), or (B, 0) if unused.
            diagnoses: Diagnosis features shaped (B, D), or (B, 0) if unused.
            valid_mask: Boolean (B, T) mask. True marks hours to score. Keep
                early hours as inputs, but mark their labels and padding False.
            targets: Optional actual days left, shaped (B, T). Kept targets
                must be finite and greater than zero.
            **kwargs: Extra batch fields, which are ignored.

        Returns:
            A dictionary with predictions (B, T), kept predictions y_prob
            (N_valid, 1), and the same values as logit. These values are days.
            With targets, also returns loss and kept targets y_true (N_valid, 1).
            Score only the kept predictions; predictions includes padded hours.

        Raises:
            ValueError: If input shapes or values used by masked_log_loss
                are invalid.
        """
        x = x.to(device=self.device, dtype=torch.float32)
        static = static.to(device=self.device, dtype=torch.float32)
        diagnoses = diagnoses.to(
            device=self.device, dtype=torch.float32
        )
        valid = valid_mask.to(device=self.device, dtype=torch.bool)

        if x.ndim != 3 or x.shape[1] != 2 * self.features + 2:
            raise ValueError(
                "Expected x shaped [batch, 2*features+2, time]"
            )

        b, channels, t = x.shape
        if (
            static.shape != (b, self.static_dim)
            or diagnoses.shape != (b, self.diagnosis_dim)
        ):
            raise ValueError("Incorrect static or diagnosis dimensions")
        if valid.shape != (b, t):
            raise ValueError("Incorrect prediction mask dimensions")

        values, masks = torch.split(
            x[:, 1:-1], self.features, dim=1
        )
        next_x = torch.stack(
            (values, masks), dim=2
        ).reshape(b, -1, t)

        raw = x.transpose(1, 2).reshape(b * t, channels)
        repeated_static = static.repeat_interleave(t, dim=0)
        point_skip = values
        previous_temporal = previous_point = None

        for layer in self.layers:
            next_x, point_skip, previous_temporal, previous_point = layer(
                next_x, raw, repeated_static, point_skip,
                previous_temporal, previous_point,
            )

        parts = [repeated_static]
        if self.diagnosis_encoder is not None:
            diagnosis_embedding = self.diagnosis_encoder(diagnoses)
            parts.append(
                diagnosis_embedding.repeat_interleave(t, dim=0)
            )
        parts.append(
            next_x.transpose(1, 2).reshape(b * t, -1)
        )

        log_prediction = self.head(
            torch.cat(parts, dim=1)
        ).reshape(b, t)

        predictions = log_prediction.clamp(
            math.log(1 / 48), math.log(100)
        ).exp()

        selected = predictions[valid].reshape(-1, 1)
        result = {
            "predictions": predictions,
            "y_prob": selected,
            "logit": selected,
        }

        if targets is not None:
            targets = targets.to(
                device=self.device, dtype=torch.float32
            )
            result["loss"] = masked_log_loss(
                predictions, targets, valid
            )
            result["y_true"] = targets[valid].reshape(-1, 1)

        return result
