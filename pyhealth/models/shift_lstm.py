from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel
from .embedding import EmbeddingModel


class ShiftLSTMLayer(nn.Module):
    """Segment-wise LSTM layer with relaxed parameter sharing over time.

    The layer divides each sequence into ``num_segments`` temporal chunks. A
    dedicated ``nn.LSTMCell`` is used within each chunk, while the hidden and
    cell states are propagated through the entire sequence.

    This implements the core idea of shiftLSTM from
    "Relaxed Parameter Sharing: Effectively Modeling Time-Varying Relationships
    in Clinical Time-Series".

    Args:
        input_size: Input feature size.
        hidden_size: Hidden state size.
        num_segments: Number of temporal segments. ``1`` reduces to a standard
            shared-parameter LSTM.
        dropout: Dropout applied to the input sequence before recurrence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_segments: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments must be >= 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_segments = num_segments
        self.dropout = dropout

        self.dropout_layer = nn.Dropout(dropout)
        self.cells = nn.ModuleList(
            [nn.LSTMCell(input_size, hidden_size) for _ in range(num_segments)]
        )

    def _compute_lengths(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                (batch_size,),
                fill_value=x.size(1),
                dtype=torch.long,
                device=x.device,
            )
        else:
            lengths = mask.long().sum(dim=-1).clamp_min(1)
        return lengths

    def _segment_index(self, step: int, lengths: torch.Tensor) -> torch.Tensor:
        # Relative segment assignment per sample:
        # floor(step / length * K), clamped to [0, K - 1].
        seg = torch.div(
            step * self.num_segments,
            lengths,
            rounding_mode="floor",
        )
        return seg.clamp(max=self.num_segments - 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: Tensor of shape ``[batch_size, seq_len, input_size]``.
            mask: Optional tensor of shape ``[batch_size, seq_len]`` with 1 for
                valid steps and 0 for padding.

        Returns:
            outputs: Tensor of shape ``[batch_size, seq_len, hidden_size]``.
            last_outputs: Tensor of shape ``[batch_size, hidden_size]``.
        """
        x = self.dropout_layer(x)
        batch_size, seq_len, _ = x.shape

        if mask is not None:
            mask = mask.to(x.device).bool()

        lengths = self._compute_lengths(x, mask)
        outputs = torch.zeros(
            batch_size, seq_len, self.hidden_size, device=x.device, dtype=x.dtype
        )
        last_outputs = torch.zeros(
            batch_size, self.hidden_size, device=x.device, dtype=x.dtype
        )

        h_t = torch.zeros(
            batch_size, self.hidden_size, device=x.device, dtype=x.dtype
        )
        c_t = torch.zeros(
            batch_size, self.hidden_size, device=x.device, dtype=x.dtype
        )

        for step in range(seq_len):
            if mask is None:
                valid = step < lengths
            else:
                valid = mask[:, step]

            if not valid.any():
                continue

            step_segment = self._segment_index(step, lengths)

            next_h = h_t.clone()
            next_c = c_t.clone()

            for segment_idx, cell in enumerate(self.cells):
                select = valid & (step_segment == segment_idx)
                if not select.any():
                    continue
                h_sel, c_sel = cell(x[select, step, :], (h_t[select], c_t[select]))
                next_h[select] = h_sel
                next_c[select] = c_sel

            h_t = next_h
            c_t = next_c
            outputs[valid, step, :] = h_t[valid]
            last_outputs[valid] = h_t[valid]

        return outputs, last_outputs


class ShiftLSTM(BaseModel):
    """PyHealth model wrapper for shiftLSTM.

    This model mirrors the high-level structure of :class:`pyhealth.models.RNN`,
    but replaces the shared-parameter recurrent layer with ``ShiftLSTMLayer`` to
    better model time-varying input-label relationships.

    Args:
        dataset: Dataset used to infer feature and label schemas.
        embedding_dim: Shared embedding size for all input features.
        hidden_dim: Hidden state size of each shiftLSTM layer.
        num_segments: Number of temporal segments per feature sequence.
        dropout: Input dropout applied before recurrence.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_segments: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__(dataset=dataset)
        assert len(self.label_keys) == 1, (
            "Only one label key is supported if ShiftLSTM is initialized"
        )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.dropout = dropout
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.shift_lstm = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.shift_lstm[feature_key] = ShiftLSTMLayer(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_segments=num_segments,
                dropout=dropout,
            )

        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def _extract_inputs_and_masks(self, **kwargs):
        inputs = {}
        masks = {}

        for feature_key in self.feature_keys:
            feature = kwargs[feature_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            schema = self.dataset.input_processors[feature_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{feature_key}' must contain 'value' in the schema."
                )

            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask

        return inputs, masks

    def _prepare_sequence_feature(
        self,
        feature_key: str,
        embedded_feature: torch.Tensor,
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = embedded_feature
        x_dim_orig = x.dim()

        if x_dim_orig == 4:
            x = x.sum(dim=2)
            if feature_key in masks:
                mask = (masks[feature_key].to(self.device).sum(dim=-1) > 0).int()
            else:
                mask = (torch.abs(x).sum(dim=-1) != 0).int()
        elif x_dim_orig == 2:
            x = x.unsqueeze(1)
            mask = None
        else:
            if feature_key in masks:
                mask = masks[feature_key].to(self.device).int()
                if mask.dim() == 3:
                    mask = (mask.sum(dim=-1) > 0).int()
            else:
                mask = (torch.abs(x).sum(dim=-1) != 0).int()

        return x, mask

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward propagation."""
        inputs, masks = self._extract_inputs_and_masks(**kwargs)
        embedded = self.embedding_model(inputs, masks=masks)

        patient_emb = []
        for feature_key in self.feature_keys:
            x, mask = self._prepare_sequence_feature(
                feature_key, embedded[feature_key], masks
            )
            _, last_output = self.shift_lstm[feature_key](x, mask)
            patient_emb.append(last_output)

        patient_emb = torch.cat(patient_emb, dim=1)
        logits = self.fc(patient_emb)

        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
