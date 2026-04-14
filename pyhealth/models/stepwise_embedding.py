from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from .base_model import BaseModel


class StepWiseEmbeddingModel(BaseModel):
    """Step-wise embedding model for sparse heterogeneous ICU measurements.

    The model accepts raw step-wise inputs where each time step contains a list
    of observed variable identifiers and their numeric values. Each variable is
    embedded independently, combined with an observed-value projection, pooled
    into a step embedding, and then encoded by a recurrent sequence model.

    Args:
        dataset: ``SampleDataset`` produced by a task such as
            ``StepWiseMortalityPredictionMIMICExtract``.
        embedding_dim: Shared variable and step embedding dimension.
        hidden_dim: Hidden dimension of the recurrent encoder.
        rnn_type: One of ``"GRU"`` or ``"LSTM"``.
        num_layers: Number of recurrent layers.
        dropout: Dropout applied after the recurrent encoder.
        group_mode: ``"grouped"`` adds learned group embeddings to variables,
            while ``"flat"`` disables them for ablation.
        num_groups: Number of learned feature groups used when
            ``group_mode="grouped"``.
    """

    def __init__(
        self,
        dataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.1,
        group_mode: str = "grouped",
        num_groups: int = 8,
    ) -> None:
        super().__init__(dataset=dataset)

        if len(self.feature_keys) < 1:
            raise ValueError("StepWiseEmbeddingModel requires at least one input key.")
        if len(self.label_keys) != 1:
            raise ValueError("StepWiseEmbeddingModel supports a single label.")
        if rnn_type not in {"GRU", "LSTM"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")
        if group_mode not in {"grouped", "flat"}:
            raise ValueError(f"Unsupported group_mode: {group_mode}")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout_rate = dropout
        self.group_mode = group_mode
        self.num_groups = num_groups

        self.step_key = "step_wise_inputs"
        if self.step_key not in self.feature_keys:
            self.step_key = self.feature_keys[0]
        self.hours_key = "hours" if "hours" in self.feature_keys else None
        self.label_key = self.label_keys[0]

        self.code_to_index = self._build_code_vocab(dataset)
        self.vocab_size = len(self.code_to_index)

        self.code_embedding = nn.Embedding(self.vocab_size + 1, embedding_dim)
        self.value_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
        )
        if group_mode == "grouped":
            self.group_embedding = nn.Embedding(num_groups, embedding_dim)
        else:
            self.group_embedding = None

        self.time_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.Tanh(),
        )

        rnn_dropout = dropout if num_layers > 1 else 0.0
        rnn_cls = nn.GRU if rnn_type == "GRU" else nn.LSTM
        self.sequence_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, self.get_output_size())

    def _build_code_vocab(self, dataset) -> Dict[str, int]:
        """Infer a stable code vocabulary from raw step-wise samples."""
        codes: set[str] = set()
        for idx in range(len(dataset)):
            sample = dataset[idx]
            for step in sample.get("step_wise_inputs", []):
                for code in step.get("codes", []):
                    codes.add(str(code))

        if not codes:
            raise ValueError(
                "StepWiseEmbeddingModel could not infer any variable codes from "
                "the provided dataset."
            )
        return {code: i + 1 for i, code in enumerate(sorted(codes))}

    def _group_ids(self, code_indices: torch.Tensor) -> torch.Tensor:
        """Map variable ids onto deterministic feature groups."""
        return torch.remainder(code_indices - 1, self.num_groups).clamp(min=0)

    def _encode_steps(
        self,
        steps_batch: List[List[Dict[str, Any]]],
        hours_batch: List[List[float]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw batch input into padded step embeddings and lengths."""
        batch_size = len(steps_batch)
        lengths = torch.tensor(
            [len(steps) for steps in steps_batch],
            dtype=torch.long,
            device=self.device,
        )
        max_steps = int(lengths.max().item()) if batch_size > 0 else 0

        sequence = torch.zeros(
            batch_size,
            max_steps,
            self.embedding_dim,
            device=self.device,
        )

        for batch_idx, steps in enumerate(steps_batch):
            sample_hours = hours_batch[batch_idx] if hours_batch is not None else None
            for step_idx, step in enumerate(steps):
                paired_observations = [
                    (self.code_to_index[str(code)], float(value))
                    for code, value in zip(
                        step.get("codes", []),
                        step.get("values", []),
                    )
                    if str(code) in self.code_to_index
                ]
                if not paired_observations:
                    continue

                codes_tensor = torch.tensor(
                    [item[0] for item in paired_observations],
                    dtype=torch.long,
                    device=self.device,
                )
                values_tensor = torch.tensor(
                    [item[1] for item in paired_observations],
                    dtype=torch.float32,
                    device=self.device,
                ).unsqueeze(-1)

                code_embedding = self.code_embedding(codes_tensor)
                value_embedding = self.value_projection(values_tensor)
                step_embedding = code_embedding + value_embedding

                if self.group_embedding is not None:
                    group_ids = self._group_ids(codes_tensor)
                    step_embedding = step_embedding + self.group_embedding(group_ids)

                pooled = step_embedding.mean(dim=0)
                if sample_hours is not None and step_idx < len(sample_hours):
                    hour_value = torch.tensor(
                        [[float(sample_hours[step_idx])]],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    pooled = pooled + self.time_projection(hour_value).squeeze(0)

                sequence[batch_idx, step_idx] = pooled

        return sequence, lengths

    def forward(
        self,
        embed: bool = False,
        **kwargs: torch.Tensor | list[Any],
    ) -> Dict[str, torch.Tensor]:
        """Run the forward pass on a batch of raw step-wise samples."""
        steps_batch = kwargs[self.step_key]
        if not isinstance(steps_batch, list):
            raise TypeError(
                "StepWiseEmbeddingModel expects raw batched step inputs as lists. "
                "Use a task with input_schema={'step_wise_inputs': 'raw'}."
            )

        hours_batch = kwargs.get(self.hours_key) if self.hours_key is not None else None
        sequence, lengths = self._encode_steps(steps_batch, hours_batch)

        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.sequence_encoder(packed)
        if self.rnn_type == "LSTM":
            hidden_state = hidden[0][-1]
        else:
            hidden_state = hidden[-1]

        patient_embedding = self.dropout(hidden_state)
        logits = self.output_layer(patient_embedding)
        y_true = kwargs[self.label_key].to(self.device)

        outputs = {
            "loss": self.get_loss_function()(logits, y_true),
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if embed:
            outputs["embed"] = patient_embedding
        return outputs

    def forward_from_embedding(
        self,
        **kwargs: torch.Tensor | list[Any],
    ) -> Dict[str, torch.Tensor]:
        """Interpretability hook for dense step-wise inputs."""
        return self.forward(**kwargs)
