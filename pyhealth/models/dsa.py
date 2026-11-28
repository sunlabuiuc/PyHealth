"""Dynamic Survival Analysis model for early event prediction.

Contributors:
    Zhipeng Zhou
    Xinyue Zuo

Related paper:
    Dynamic Survival Analysis for Early Event Prediction
    https://proceedings.mlr.press/v248/yeche24a.html

This module implements the DSA model and discrete-time survival outputs for
integration with PyHealth.
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class DSA(BaseModel):
    """Dynamic Survival Analysis model for early event prediction.

    DSA models the discrete-time conditional hazard of an event over a future
    prediction horizon. Each input timestep combines continuous measurements,
    categorical measurements, and observation masks. A linear timestep
    embedding and GRU encode the observed history, and a hazard head predicts
    one conditional hazard probability for each future time bin.

    The model computes its own masked discrete-time survival loss rather than
    using a standard PyHealth label-processor loss. It additionally returns the
    probability mass function, survival function, cumulative risk, and a
    fixed-horizon event probability.

    Args:
        dataset: Processed PyHealth sample dataset containing
            ``dynamic_continuous``, ``dynamic_categorical``, ``dynamic_mask``,
            ``hazard_label``, ``hazard_mask``, and ``y_true`` fields.
        hidden_dim: Size of the timestep embedding and GRU hidden state.
        max_horizon: Number of future discrete hazard bins predicted.
        prediction_horizon: Zero-based cumulative-risk index returned as
            ``y_prob``. For hourly bins, index 23 represents risk through the
            first 24 future hours.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability between GRU layers. PyTorch applies GRU
            dropout only when ``num_layers > 1``.
        hazard_prevalence: Optional per-bin hazard prevalence used to initialize
            the hazard-head bias logits.
        l1_reg_embedding: Weight applied to the mean absolute value of the
            timestep-embedding parameters.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import DSA
        >>> from pyhealth.tasks import DecompensationMIMIC3
        >>> task = DecompensationMIMIC3()
        >>> sample = {
        ...     "patient_id": "synthetic-0",
        ...     "dynamic_continuous": {"time": [1.0], "values": [[0.0] * 12]},
        ...     "dynamic_categorical": {"time": [1.0], "values": [[0.0] * 47]},
        ...     "dynamic_mask": {"time": [1.0], "values": [[1.0] * 17]},
        ...     "hazard_label": [0.0] * 24,
        ...     "hazard_mask": [1.0] * 24,
        ...     "y_true": 0,
        ... }
        >>> positive = dict(sample, patient_id="synthetic-1", y_true=1)
        >>> positive["hazard_label"] = [0.0] * 23 + [1.0]
        >>> dataset = create_sample_dataset(
        ...     samples=[sample, positive],
        ...     input_schema=task.input_schema,
        ...     output_schema=task.output_schema,
        ...     dataset_name="dsa_example",
        ... )
        >>> model = DSA(dataset, hidden_dim=8)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>> model(**batch)["hazard"].shape
        torch.Size([2, 24])

    Raises:
        ValueError: If a model hyperparameter is invalid or the dataset is
            missing required inputs or outputs.
    """

    @property
    def label_processors(self) -> dict[str, Any]:
        """Disable standard label-processor loss for the custom survival loss."""
        return {}

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 128,
        max_horizon: int = 24,
        prediction_horizon: int = 23,
        num_layers: int = 1,
        dropout: float = 0.0,
        hazard_prevalence: Sequence[float] | torch.Tensor | None = None,
        l1_reg_embedding: float = 0.0,
    ) -> None:
        super().__init__(dataset)

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if max_horizon <= 0:
            raise ValueError("max_horizon must be positive.")
        if not 0 <= prediction_horizon < max_horizon:
            raise ValueError(
                "prediction_horizon must satisfy "
                "0 <= prediction_horizon < max_horizon."
            )
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if l1_reg_embedding < 0.0:
            raise ValueError("l1_reg_embedding must be non-negative.")

        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.prediction_horizon = prediction_horizon
        self.l1_reg_embedding = l1_reg_embedding

        required_inputs = {
            "dynamic_continuous",
            "dynamic_categorical",
            "dynamic_mask",
        }
        missing = required_inputs - set(dataset.input_schema)
        if missing:
            raise ValueError(f"Missing DSA inputs: {sorted(missing)}")

        required_outputs = {"hazard_label", "hazard_mask", "y_true"}
        missing = required_outputs - set(dataset.output_schema)
        if missing:
            raise ValueError(f"Missing DSA outputs: {sorted(missing)}")

        sample = dataset[0]
        continuous_dim = int(sample["dynamic_continuous"][0].shape[-1])
        categorical_dim = int(sample["dynamic_categorical"][0].shape[-1])
        mask_dim = int(sample["dynamic_mask"][0].shape[-1])
        self.input_dim = continuous_dim + categorical_dim + mask_dim

        self.time_step_embedding = nn.Linear(self.input_dim, hidden_dim)

        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.hazard_head = nn.Linear(hidden_dim, max_horizon)

        if hazard_prevalence is not None:
            self.initialize_hazard_bias(hazard_prevalence)

    def initialize_hazard_bias(
        self,
        hazard_prevalence: Sequence[float] | torch.Tensor,
    ) -> None:
        """Initialize hazard-head biases from per-bin prevalence.

        Args:
            hazard_prevalence: Sequence containing one event prevalence value
                for each discrete future hazard bin.

        Raises:
            ValueError: If the number of prevalence values does not equal
                ``max_horizon``.
        """
        prevalence = torch.as_tensor(
            hazard_prevalence,
            dtype=self.hazard_head.bias.dtype,
            device=self.hazard_head.bias.device,
        ).flatten()

        if prevalence.numel() != self.max_horizon:
            raise ValueError(
                "hazard_prevalence must contain exactly "
                f"{self.max_horizon} values, got {prevalence.numel()}."
            )

        prevalence = prevalence.clamp(1e-7, 1.0 - 1e-7)

        with torch.no_grad():
            self.hazard_head.bias.copy_(torch.logit(prevalence))

    def embedding_l1_loss(self) -> torch.Tensor:
        """Compute L1 regularization for the timestep embedding.

        Returns:
            Mean absolute value across all timestep-embedding parameters.
        """
        params = list(self.time_step_embedding.parameters())
        n_params = sum(p.numel() for p in params)
        return sum(p.abs().sum() for p in params) / n_params

    @staticmethod
    def _unpack(
        field: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(field) != 3:
            raise ValueError(
                "DSA expects sequence_v2 fields as "
                "(values, time, offsets)."
            )
        return field

    @staticmethod
    def _pad(
        values: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = [
            values[offsets[i] : offsets[i + 1]]
            for i in range(len(offsets) - 1)
        ]

        if any(len(sequence) == 0 for sequence in sequences):
            raise ValueError(
                "DSA requires at least one observation per sample."
            )

        lengths = torch.tensor(
            [len(sequence) for sequence in sequences],
            dtype=torch.long,
            device=values.device,
        )

        return pad_sequence(sequences, batch_first=True), lengths

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        """Run a forward pass and compute the discrete survival loss.

        Args:
            **batch: Batched PyHealth fields. The three dynamic inputs must be
                ``sequence_v2`` tuples of ``(values, time, offsets)``.
                ``hazard_label`` and ``hazard_mask`` must have shape
                ``(batch_size, max_horizon)``.

        Returns:
            Dictionary containing ``loss``, ``survival_loss``, ``l1_loss``,
            ``y_prob``, ``y_true``, ``logit``, ``hazard_logit``, ``hazard``,
            ``pmf``, ``survival``, ``cumulative_risk``, and ``embed``.

        Raises:
            ValueError: If sequence metadata are inconsistent or survival
                targets do not match the configured horizon.
        """
        continuous, continuous_time, continuous_offsets = self._unpack(
            batch["dynamic_continuous"]
        )
        categorical, categorical_time, categorical_offsets = self._unpack(
            batch["dynamic_categorical"]
        )
        observation_mask, mask_time, mask_offsets = self._unpack(
            batch["dynamic_mask"]
        )

        continuous = continuous.to(self.device)
        categorical = categorical.to(self.device)
        observation_mask = observation_mask.to(self.device)

        continuous_time = continuous_time.to(self.device)
        categorical_time = categorical_time.to(self.device)
        mask_time = mask_time.to(self.device)

        continuous_offsets = continuous_offsets.to(self.device)
        categorical_offsets = categorical_offsets.to(self.device)
        mask_offsets = mask_offsets.to(self.device)

        if not torch.equal(continuous_offsets, categorical_offsets):
            raise ValueError(
                "Continuous and categorical offsets do not match."
            )
        if not torch.equal(continuous_offsets, mask_offsets):
            raise ValueError("Continuous and mask offsets do not match.")
        if not torch.equal(continuous_time, categorical_time):
            raise ValueError("Continuous and categorical times do not match.")
        if not torch.equal(continuous_time, mask_time):
            raise ValueError("Continuous and mask times do not match.")

        values = torch.cat(
            [continuous, categorical, observation_mask],
            dim=-1,
        )

        values = self.time_step_embedding(values)

        padded, lengths = self._pad(values, continuous_offsets)

        packed = pack_padded_sequence(
            padded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, hidden = self.gru(packed)
        representation = hidden[-1]

        hazard_logit = self.hazard_head(representation)
        hazard = torch.sigmoid(hazard_logit)

        survival = torch.cumprod(1.0 - hazard, dim=-1)

        survival_before = torch.cat(
            [
                torch.ones(
                    hazard.shape[0],
                    1,
                    dtype=hazard.dtype,
                    device=hazard.device,
                ),
                survival[:, :-1],
            ],
            dim=-1,
        )

        pmf = survival_before * hazard
        cumulative_risk = 1.0 - survival

        hazard_label = batch["hazard_label"].to(self.device).float()
        hazard_mask = batch["hazard_mask"].to(self.device).float()

        if hazard_label.shape != hazard_logit.shape:
            raise ValueError(
                f"hazard_label shape {hazard_label.shape} "
                f"does not match {hazard_logit.shape}."
            )

        if hazard_mask.shape != hazard_logit.shape:
            raise ValueError(
                f"hazard_mask shape {hazard_mask.shape} "
                f"does not match {hazard_logit.shape}."
            )

        survival_loss = F.binary_cross_entropy_with_logits(
            hazard_logit,
            hazard_label,
            reduction="none",
        )
        survival_loss = (
            survival_loss * hazard_mask
        ).sum(dim=-1).mean()

        l1_loss = (
            self.l1_reg_embedding * self.embedding_l1_loss()
            if self.l1_reg_embedding > 0.0
            else survival_loss.new_zeros(())
        )

        loss = survival_loss + l1_loss

        y_true = batch["y_true"].to(
            self.device
        ).float().reshape(-1, 1)

        y_prob = cumulative_risk[
            :,
            self.prediction_horizon : self.prediction_horizon + 1,
        ]

        logit = torch.logit(
            y_prob.clamp(1e-7, 1.0 - 1e-7)
        )

        return {
            "loss": loss,
            "survival_loss": survival_loss,
            "l1_loss": l1_loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logit,
            "hazard_logit": hazard_logit,
            "hazard": hazard,
            "pmf": pmf,
            "survival": survival,
            "cumulative_risk": cumulative_risk,
            "embed": representation,
        }
