from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Optional, cast, get_args

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import (
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
    MultiHotProcessor,
    NestedFloatsProcessor,
    NestedSequenceProcessor,
    SequenceProcessor,
    SequenceProcessorV2,
    StageNetProcessor,
    StageNetTensorProcessor,
    TensorProcessor,
    TimeseriesProcessor,
)

from .embedding import EmbeddingModel
from .sequence_embedding import (
    LevelEmbedding,
    MaskMode,
    PretrainedEmbedding,
    SequenceEmbeddingModel,
)

# Label modes with one prediction per label record (TimeseriesLabelProcessor).
_PER_STEP_MODES = ("timeseries_binary", "timeseries_regression")


class RNNLayer(nn.Module):
    """Recurrent neural network layer.

    This layer wraps the PyTorch RNN layer with masking and dropout support. It is
    used in the RNN model. But it can also be used as a standalone layer.

    Args:
        input_size: input feature size.
        hidden_size: hidden feature size.
        rnn_type: type of rnn, one of "RNN", "LSTM", "GRU". Default is "GRU".
        num_layers: number of recurrent layers. Default is 1.
        dropout: dropout rate. If non-zero, introduces a Dropout layer before each
            RNN layer. Default is 0.5.
        bidirectional: whether to use bidirectional recurrent layers. If True,
            a fully-connected layer is applied to the concatenation of the forward
            and backward hidden states to reduce the dimension to hidden_size.
            Default is False.

    Examples:
        >>> from pyhealth.models import RNNLayer
        >>> input = torch.randn(3, 128, 5)  # [batch size, sequence len, input_size]
        >>> layer = RNNLayer(5, 64)
        >>> outputs, last_outputs = layer(input)
        >>> outputs.shape
        torch.Size([3, 128, 64])
        >>> last_outputs.shape
        torch.Size([3, 64])
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rnn_type: str = "GRU",
        num_layers: int = 1,
        dropout: float = 0.5,
        bidirectional: bool = False,
    ):
        super(RNNLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.dropout_layer = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        rnn_module = getattr(nn, rnn_type)
        self.rnn = rnn_module(
            input_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        if bidirectional:
            self.down_projection = nn.Linear(hidden_size * 2, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward propagation.

        Args:
            x: a tensor of shape [batch size, sequence len, input size].
            mask: an optional tensor of shape [batch size, sequence len], where
                1 indicates valid and 0 indicates invalid. Samples with all-zero
                masks are clamped to length 1 to prevent pack_padded_sequence
                from receiving zero-length sequences.

        Returns:
            outputs: a tensor of shape [batch size, sequence len, hidden size],
                containing the output features for each time step.
            last_outputs: a tensor of shape [batch size, hidden size], containing
                the output features for the last time step.
        """
        # pytorch's rnn will only apply dropout between layers
        x = self.dropout_layer(x)
        batch_size = x.size(0)
        if mask is None:
            lengths = torch.full(
                size=(batch_size,), fill_value=x.size(1), dtype=torch.int64
            )
        else:
            lengths = torch.sum(mask.int(), dim=-1).cpu()
        # Clamp lengths to at least 1 to handle empty sequences,
        # matching TCNLayer (tcn.py:186).
        lengths = torch.clamp(lengths, min=1)
        # Ensure tensor is contiguous for cuDNN compatibility
        x = x.contiguous()
        x = rnn_utils.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        outputs, _ = self.rnn(x)
        outputs, _ = rnn_utils.pad_packed_sequence(outputs, batch_first=True)
        # Ensure outputs are contiguous after unpacking
        outputs = outputs.contiguous()

        if not self.bidirectional:
            last_outputs = outputs[torch.arange(batch_size), (lengths - 1), :]
            return outputs, last_outputs
        else:
            outputs = outputs.view(batch_size, outputs.shape[1], 2, -1)
            f_last_outputs = outputs[torch.arange(batch_size), (lengths - 1), 0, :]
            b_last_outputs = outputs[:, 0, 1, :]
            last_outputs = torch.cat([f_last_outputs, b_last_outputs], dim=-1)
            # Ensure view result is contiguous for cuDNN
            outputs = outputs.view(batch_size, outputs.shape[1], -1).contiguous()
            last_outputs = self.down_projection(last_outputs)
            outputs = self.down_projection(outputs)
            return outputs, last_outputs


class RNN(BaseModel):
    """Recurrent neural network model.

    This model applies a separate RNN layer for each feature, and then concatenates
    the final hidden states of each RNN layer. The concatenated hidden states are
    then fed into a fully connected layer to make predictions.

    Note:
        We use separate RNN layers for different feature_keys.
        Currently, we support two types of input formats:
            - Sequence of codes (e.g., diagnosis codes, procedure codes)
                - Input format: (batch_size, sequence_length)
                - Each code is embedded into a vector and RNN is applied on the sequence
            - Timeseries values (e.g., lab tests, vital signs)
                - Input format: (batch_size, sequence_length, num_features)
                - Each timestep contains a fixed number of measurements
                - RNN is applied directly on the timeseries data

    Args:
        dataset (SampleDataset): the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim (int): the embedding dimension. Default is 128.
        hidden_dim (int): the hidden dimension. Default is 128.
        **kwargs: other parameters for the RNN layer (e.g., rnn_type, num_layers, dropout, bidirectional).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86", "cond-80"],
        ...         "procedures": ["proc-12", "proc-45"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-12", "cond-52"],
        ...         "procedures": ["proc-23"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = RNN(dataset=dataset, embedding_dim=128, hidden_dim=64)
        >>>
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(RNN, self).__init__(
            dataset=dataset,
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        assert len(self.label_keys) == 1, "Only one label key is supported if RNN is initialized"
        self.label_key = self.label_keys[0]
        if self.mode in _PER_STEP_MODES:
            raise ValueError(
                "RNN does not support per-step (timeseries_*) labels; use "
                "RNN_V2 with sequence_v2 inputs."
            )

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        self.rnn = nn.ModuleDict()
        for feature_key in self.dataset.input_processors.keys():
            self.rnn[feature_key] = RNNLayer(
                input_size=embedding_dim, hidden_size=hidden_dim, **kwargs
            )
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.hidden_dim, output_size)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient embeddings if requested.
        """
        patient_emb = []
        
        # We need to preprocess kwargs to extract values and masks for EmbeddingModel
        # because EmbeddingModel expects dict of tensors
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
                raise ValueError(f"Feature '{feature_key}' must contain 'value' in the schema.")
            
            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask

        embedded = self.embedding_model(inputs, masks=masks)
        
        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            x_dim_orig = x.dim()
            if x_dim_orig == 4:
                # nested_sequence: (B, num_visits, num_codes, D)
                # @TODO: sum-pooling across codes is a simple baseline. May need to investigate better embeddings for nested codes.
                x = x.sum(dim=2)  # (B, num_visits, D)
                if feature_key in masks:
                    mask = (masks[feature_key].to(self.device).sum(dim=-1) > 0).int()  # (B, V)
                else:
                    mask = (torch.abs(x).sum(dim=-1) != 0).int()
            elif x_dim_orig == 2:
                x = x.unsqueeze(1)
                mask = None
            else:
                # 3D: already (B, T, D)
                if feature_key in masks:
                    mask = masks[feature_key].to(self.device).int()
                    if mask.dim() == 3:
                        mask = (mask.sum(dim=-1) > 0).int()
                else:
                    mask = (torch.abs(x).sum(dim=-1) != 0).int()

            _, x = self.rnn[feature_key](x, mask)
            patient_emb.append(x)

        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)
        # obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        results = {"loss": loss, "y_prob": y_prob, "y_true": y_true, "logit": logits}
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


class MultimodalRNN(BaseModel):
    """Multimodal RNN model that handles both sequential and non-sequential features.

    This model extends the vanilla RNN to support mixed input modalities:
    - Sequential features (sequences, timeseries) go through RNN layers
    - Non-sequential features (multi-hot, tensor) bypass RNN and use embeddings directly

    The model automatically classifies input features based on their processor types:
    - Sequential processors (apply RNN): SequenceProcessor, NestedSequenceProcessor,
        DeepNestedSequenceProcessor, NestedFloatsProcessor, DeepNestedFloatsProcessor,
        TimeseriesProcessor
    - Non-sequential processors (embeddings only): MultiHotProcessor, TensorProcessor,
        StageNetProcessor, StageNetTensorProcessor

    For sequential features, the model:
    1. Embeds the input using EmbeddingModel
    2. Applies RNNLayer to get sequential representations
    3. Extracts the last hidden state

    For non-sequential features, the model:
    1. Embeds the input using EmbeddingModel
    2. Applies mean pooling if needed to reduce to 2D
    3. Uses the embedding directly

    All feature representations are concatenated and passed through a final
    fully connected layer for predictions.

    Args:
        dataset (SampleDataset): the dataset to train the model. It is used to query
            certain information such as the set of all tokens and processor types.
        embedding_dim (int): the embedding dimension. Default is 128.
        hidden_dim (int): the hidden dimension for RNN layers. Default is 128.
        **kwargs: other parameters for the RNN layer (e.g., rnn_type, num_layers,
            dropout, bidirectional).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["cond-33", "cond-86"],  # sequential
        ...         "demographics": ["asian", "male"],      # multi-hot
        ...         "vitals": [120.0, 80.0, 98.6],        # tensor
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["cond-12", "cond-52"],  # sequential
        ...         "demographics": ["white", "female"],    # multi-hot
        ...         "vitals": [110.0, 75.0, 98.2],        # tensor
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "sequence",
        ...         "demographics": "multi_hot",
        ...         "vitals": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="test"
        ... )
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>>
        >>> model = MultimodalRNN(dataset=dataset, embedding_dim=128, hidden_dim=64)
        >>>
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(...),
            'y_prob': tensor(...),
            'y_true': tensor(...),
            'logit': tensor(...)
        }
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        **kwargs
    ):
        super(MultimodalRNN, self).__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # validate kwargs for RNN layer
        if "input_size" in kwargs:
            raise ValueError("input_size is determined by embedding_dim")
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]
        if self.mode in _PER_STEP_MODES:
            raise ValueError(
                "MultimodalRNN does not support per-step (timeseries_*) labels; "
                "use RNN_V2 with sequence_v2 inputs."
            )

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Classify features as sequential or non-sequential
        self.sequential_features = []
        self.non_sequential_features = []

        self.rnn = nn.ModuleDict()
        for feature_key in self.feature_keys:
            processor = dataset.input_processors[feature_key]
            if self._is_sequential_processor(processor):
                self.sequential_features.append(feature_key)
                # Create RNN for this feature
                self.rnn[feature_key] = RNNLayer(
                    input_size=embedding_dim,
                    hidden_size=hidden_dim,
                    **kwargs
                )
            else:
                self.non_sequential_features.append(feature_key)

        # Calculate final concatenated dimension
        final_dim = (len(self.sequential_features) * hidden_dim +
                     len(self.non_sequential_features) * embedding_dim)
        output_size = self.get_output_size()
        self.fc = nn.Linear(final_dim, output_size)

    def _is_sequential_processor(self, processor) -> bool:
        """Check if processor represents sequential data.

        Sequential processors are those that benefit from RNN processing,
        including sequences of codes and timeseries data.

        Note:
            StageNetProcessor and StageNetTensorProcessor are excluded as they
            are specialized for the StageNet model architecture and should be
            treated as non-sequential for standard RNN processing.

        Args:
            processor: The processor instance to check.

        Returns:
            bool: True if processor is sequential, False otherwise.
        """
        return isinstance(processor, (
            SequenceProcessor,
            NestedSequenceProcessor,
            DeepNestedSequenceProcessor,
            NestedFloatsProcessor,
            DeepNestedFloatsProcessor,
            TimeseriesProcessor,
        ))

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation handling mixed modalities.

        The label `kwargs[self.label_key]` is a list of labels for each patient.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient embeddings if requested.
        """
        # Preprocess features
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
                raise ValueError(f"Feature '{feature_key}' must contain 'value' in the schema.")
            
            inputs[feature_key] = value
            if mask is not None:
                masks[feature_key] = mask

        patient_emb = []
        embedded, mask = self.embedding_model(inputs, masks=masks, output_mask=True)

        # Process sequential features through RNN
        for feature_key in self.sequential_features:
            x = embedded[feature_key]
            m = mask[feature_key]

            x_dim_orig = x.dim()
            if x_dim_orig == 4:
                # nested_sequence: (B, num_visits, num_codes, D)
                # Pool codes within each visit, then run RNN over visits.
                # Flattening visits*codes would produce length=0 when inner lists are empty.
                x = x.sum(dim=2)  # (B, num_visits, D)
                if feature_key in masks:
                    m = (masks[feature_key].to(self.device).sum(dim=-1) > 0).int()  # (B, V)
                else:
                    m = (torch.abs(x).sum(dim=-1) != 0).int()
            elif x_dim_orig == 2:
                x = x.unsqueeze(1)
                m = None
            else:
                # 3D: already (B, T, D)
                if m is not None and m.dim() == 3:
                    m = (m.sum(dim=-1) > 0).int()
                elif m is not None and m.dim() == 1:
                    m = m.unsqueeze(1)

            _, last_hidden = self.rnn[feature_key](x, m)
            patient_emb.append(last_hidden)

        # Process non-sequential features (use embeddings directly)
        for feature_key in self.non_sequential_features:
            x = embedded[feature_key]
            # If multi-dimensional, aggregate (mean pooling)
            while x.dim() > 2:
                x = x.mean(dim=1)
            patient_emb.append(x)

        # Concatenate all representations
        patient_emb = torch.cat(patient_emb, dim=1)
        # (patient, label_size)
        logits = self.fc(patient_emb)

        # Calculate loss and predictions
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)

        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


# ----------------------------------------------------------------------------
# RNN_V2: RNNs over the packed records of SequenceProcessorV2 fields
# ----------------------------------------------------------------------------

RNNType = Literal["RNN", "LSTM", "GRU"]


def _per_field[T](
    name: str,
    value: T | Mapping[str, T],
    fields: Sequence[str],
    default: T,
    valid: Callable[[T], bool],
) -> dict[str, T]:
    """One value per field, from a single value or a ``{field: value}`` dict.

    Fields a dict leaves out get ``default``.
    """
    if isinstance(value, Mapping):
        unknown = set(value) - set(fields)
        if unknown:
            raise ValueError(
                f"{name} names {sorted(unknown)}, which are not among {sorted(fields)}."
            )
        # T is never itself a Mapping here, so the dict holds T values.
        values = {field: cast(T, value.get(field, default)) for field in fields}
    else:
        values = {field: cast(T, value) for field in fields}
    for field, item in values.items():
        if not valid(item):
            raise ValueError(f"{name} for '{field}' is invalid: {item!r}.")
    return values


def _groups(offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The group of each row and its position in the group, from CSR offsets."""
    counts = offsets.diff()
    group = torch.repeat_interleave(
        torch.arange(len(counts), device=offsets.device), counts
    )
    return group, torch.arange(len(group), device=offsets.device) - offsets[:-1][group]


def _pool(x: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Mean-pools the rows of each group; an empty group gives zeros."""
    counts = offsets.diff()
    group, _ = _groups(offsets)
    out = x.new_zeros(len(counts), x.shape[1])
    out = out.index_add(0, group, x)
    return out / counts.clamp(min=1).unsqueeze(1).to(x.dtype)


def _gaps(time: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Time since the previous row of the same group (0 for a group's first).

    ``time`` is ``[N]`` or ``[N, k]`` (one column per timed level).
    """
    _, position = _groups(offsets)
    gaps = torch.diff(time, dim=0, prepend=time[:1])
    first = position == 0
    return gaps.masked_fill(first.unsqueeze(1) if time.dim() > 1 else first, 0.0)


def _rnn_over_groups(
    layer: RNNLayer, x: torch.Tensor, offsets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs ``layer`` over each group of rows.

    Returns each group's last output ``[G, H]`` (zeros for an empty group,
    which is skipped) and every row's output ``[N, H]``.
    """
    counts = offsets.diff()
    hidden = layer.hidden_size
    summary = x.new_zeros(len(counts), hidden)
    filled = counts > 0
    if not bool(filled.any()):
        return summary, x.new_zeros(len(x), hidden)
    group, position = _groups(offsets)
    rank = torch.cumsum(filled.long(), dim=0) - 1  # index among non-empty groups
    lengths = counts[filled]
    padded = x.new_zeros(len(lengths), int(lengths.max()), x.shape[1])
    padded[rank[group], position] = x
    steps = torch.arange(padded.shape[1], device=x.device)
    outputs, last = layer(padded, (steps < lengths.unsqueeze(1)).int())
    summary[filled] = last
    return summary, outputs[rank[group], position]


class _FieldRNN(nn.Module):
    """Encodes one V2 field's level embeddings into one summary per sample."""

    def __init__(
        self,
        field: str,
        processor: SequenceProcessorV2,
        level_dims: Sequence[int],
        hidden_dim: int,
        rnn_type: RNNType,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        time_feature: bool,
    ):
        super().__init__()
        self.depth = processor.depth
        orderings = processor.ordering()
        # A level runs an RNN unless it is a set (a resampled series is a sequence).
        self.is_rnn = [orderings[level] != "set" for level in range(self.depth + 1)]
        timed = {level for level, role in processor.schema() if role == "time"}
        self.gap_levels = sorted(timed) if time_feature else []

        def rnn(input_size: int) -> nn.Module:
            return RNNLayer(
                input_size,
                hidden_dim,
                rnn_type=rnn_type,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
            )

        self.encoders = nn.ModuleList()
        below = 0  # the summary size of the level below
        for level in range(self.depth + 1):
            width = level_dims[level] + below + (level in self.gap_levels)
            if width == 0:
                raise ValueError(
                    f"'{field}' level {level} has nothing to encode: no embedding, "
                    "and nothing below it."
                )
            self.encoders.append(rnn(width) if self.is_rnn[level] else nn.Identity())
            below = hidden_dim if self.is_rnn[level] else width
        self.output_dim = below

    # -- inputs --------------------------------------------------------------

    def _inputs(
        self, level: int, out: dict[Any, Any], parts: list[torch.Tensor]
    ) -> torch.Tensor:
        """A level's input rows: its embedding, the given parts, and its gaps."""
        embedding = out[level]["embedding"]
        rows = [embedding] if embedding is not None else []
        rows += parts
        if level in self.gap_levels:
            rows.append(_gaps(out[level]["time"], out[level]["offsets"]).unsqueeze(1))
        return torch.cat(rows, dim=1)

    # -- bottom-up -------------------------------------------------------------

    def _bottom_up(
        self, out: dict[Any, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        summary: torch.Tensor | None = None
        states = None
        for level in range(self.depth + 1):
            x = self._inputs(level, out, [summary] if summary is not None else [])
            offsets = out[level]["offsets"]
            if self.is_rnn[level]:
                encoder = cast(RNNLayer, self.encoders[level])
                summary, states = _rnn_over_groups(encoder, x, offsets)
            else:
                summary, states = _pool(x, offsets), None
        assert summary is not None
        return summary, states  # states: the top level's per-record outputs

    # -- forward -----------------------------------------------------------------

    def forward(
        self, out: dict[Any, Any]
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Encodes a batch of the field.

        Returns the summary per sample ``[B, S]``, the states of the
        top-level records (None if the top level is pooled) and their times
        (None if the top level has no time column).
        """
        summary, states = self._bottom_up(out)
        return summary, states, out[self.depth]["time"]


class RNN_V2(BaseModel):
    """RNN model over :class:`~pyhealth.processors.SequenceProcessorV2` fields.

    Each V2 field is embedded by :class:`SequenceEmbeddingModel` and encoded
    into one summary per sample, independently of the other fields. Static
    fields (:class:`~pyhealth.processors.TensorProcessor`) are used directly,
    or through a projection. The summaries and statics are concatenated and a
    linear layer predicts the label. Other processors are rejected: sequences,
    sets and token fields are expressed with ``"sequence_v2"``.

    Each V2 field is encoded bottom-up, level by level, from level 0 up.
    A level's input is its records' embeddings concatenated with the summary
    of each record's children. A ``set`` level mean-pools each group; a
    ``sequence`` / ``series`` level runs its RNN over each group and keeps
    the last output, giving one summary per parent record, up to one per sample.

    A group without records (e.g. a stay without labs) is skipped by the RNN
    and summarized as zeros. With ``time_feature``, each record of a timed
    (``series``) level also gets the time since the previous record of its
    group.

    Per-step labels (``"timeseries_binary"`` / ``"timeseries_regression"``,
    a :class:`~pyhealth.processors.TimeseriesLabelProcessor`): the model
    predicts once per label record. The label's times are in the unit of the
    fields' top level, with the same origin, and a label at time ``t`` uses
    each field's top-level state after its last record with time ``<= t``
    (zeros before the first) and the statics, so it never sees later records.
    Every V2 field must have a ``series`` top level; a resampled one counts,
    with record ``i`` at time ``i * step``. Wrap a ``set`` or ``sequence``
    top level in a ``series`` with ``t = 0``. Logits are ``[label records,
    1]``, packed like the label.

    Every per-field argument takes one value for all V2 fields, or a
    ``{field: value}`` dict (unlisted fields get the default).

    Args:
        dataset: The dataset, with fitted processors.
        embedding_dim: Per V2 field, one
            :class:`~pyhealth.models.LevelEmbedding` per level.
        hidden_dim: The hidden size of every RNN of a field.
        rnn_type: ``"RNN"``, ``"LSTM"`` or ``"GRU"`` (default), per field.
        num_layers: The layers of each level's RNN, per field.
        dropout: Dropout before each RNN, per field.
        bidirectional: Bidirectional RNNs, per field. Not with per-step labels.
        time_feature: Whether timed levels get the time since the previous
            record, per field.
        mask: How :class:`SequenceEmbeddingModel` uses missing-value masks:
            ``"ignore"`` (default) or ``"concat"``.
        pretrain: :class:`~pyhealth.models.PretrainedEmbedding` entries.
        static_dim: Per static field, the size of its projection, or None
            (default) to use it directly.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {"patient_id": f"p{i}", "age": [50.0 + i],
        ...      "visits": {"time": [0.0, 1.0], "children": [{"tokens": ["A", "B"]}, {"tokens": ["C"] * i}]},
        ...      "label": i % 2}
        ...     for i in range(4)
        ... ]
        >>> levels = [{"ordering": "set", "tokens": 1}, {"ordering": "series"}]
        >>> dataset = create_sample_dataset(
        ...     samples,
        ...     {"age": "tensor", "visits": ("sequence_v2", {"levels": levels})},
        ...     {"label": "binary"},
        ... )
        >>> model = RNN_V2(
        ...     dataset, embedding_dim={"visits": [{"tokens": (8,)}, {}]}, hidden_dim=16
        ... )
        >>> out = model(**next(iter(get_dataloader(dataset, batch_size=4))))
        >>> out["logit"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: Mapping[str, Sequence[LevelEmbedding]],
        hidden_dim: int | Mapping[str, int] = 128,
        rnn_type: RNNType | Mapping[str, RNNType] = "GRU",
        num_layers: int | Mapping[str, int] = 1,
        dropout: float | Mapping[str, float] = 0.5,
        bidirectional: bool | Mapping[str, bool] = False,
        time_feature: bool | Mapping[str, bool] = False,
        mask: MaskMode = "ignore",
        pretrain: Sequence[PretrainedEmbedding] | None = None,
        static_dim: int | Mapping[str, int | None] | None = None,
    ):
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("RNN_V2 supports exactly one label.")
        self.label_key = self.label_keys[0]
        self.per_step = self.mode in _PER_STEP_MODES

        self.sequence_fields: list[str] = []
        self.static_fields: list[str] = []
        for field in self.feature_keys:
            processor = dataset.input_processors[field]
            if isinstance(processor, SequenceProcessorV2):
                self.sequence_fields.append(field)
            elif isinstance(processor, TensorProcessor):
                self.static_fields.append(field)
            else:
                raise TypeError(
                    f"Field '{field}': {type(processor).__name__} is not supported. "
                    "RNN_V2 takes 'sequence_v2' fields and static 'tensor' "
                    "fields; express sequences, sets and tokens with 'sequence_v2'."
                )

        fields = self.sequence_fields
        hidden = _per_field(
            "hidden_dim",
            hidden_dim,
            fields,
            128,
            lambda v: isinstance(v, int) and v > 0,
        )
        rnn_types: dict[str, RNNType] = _per_field(
            "rnn_type", rnn_type, fields, "GRU", lambda v: v in get_args(RNNType)
        )
        layers = _per_field(
            "num_layers", num_layers, fields, 1, lambda v: isinstance(v, int) and v > 0
        )
        dropouts = _per_field(
            "dropout",
            dropout,
            fields,
            0.5,
            lambda v: isinstance(v, (int, float)) and 0 <= v < 1,
        )
        bidirectionals = _per_field(
            "bidirectional", bidirectional, fields, False, lambda v: isinstance(v, bool)
        )
        time_features = _per_field(
            "time_feature", time_feature, fields, False, lambda v: isinstance(v, bool)
        )
        static_dims: dict[str, int | None] = _per_field(
            "static_dim",
            static_dim,
            self.static_fields,
            None,
            lambda v: v is None or (isinstance(v, int) and v > 0),
        )

        for field in fields:
            if bidirectionals[field] and self.per_step:
                raise ValueError(
                    "A per-step label needs a causal model; bidirectional=True "
                    "would use future time steps."
                )
        self.processors = {
            field: cast(SequenceProcessorV2, dataset.input_processors[field])
            for field in fields
        }
        if self.per_step:
            for field, processor in self.processors.items():
                top = processor.levels[processor.depth]
                if top.get("ordering", "sequence") != "series":
                    raise ValueError(
                        f"'{field}': per-step labels need a 'series' top level, so each "
                        "step can use the records before it. Wrap a 'set' or "
                        "'sequence' top level in a 'series' with t = 0."
                    )

        self.embedding = SequenceEmbeddingModel(
            dataset,
            embedding_dim=embedding_dim,
            pretrain=pretrain,
            mask=mask,
        )
        self.encoders = nn.ModuleDict(
            {
                field: _FieldRNN(
                    field,
                    self.processors[field],
                    self.embedding.embedding_dim_per_level(field),
                    hidden[field],
                    rnn_types[field],
                    layers[field],
                    dropouts[field],
                    bidirectionals[field],
                    time_features[field],
                )
                for field in fields
            }
        )
        self.static_projections = nn.ModuleDict()
        width = 0
        for field in self.static_fields:
            size = int(torch.as_tensor(dataset[0][field]).numel())
            projected = static_dims[field]
            if projected is None:
                width += size
            else:
                self.static_projections[field] = nn.Linear(size, projected)
                width += projected
        for field in fields:
            width += cast(_FieldRNN, self.encoders[field]).output_dim
        self.fc = nn.Linear(width, self.get_output_size())

    def _at_labels(
        self,
        field: str,
        states: torch.Tensor,
        time: torch.Tensor | None,
        batch: torch.Tensor,
        label_time: torch.Tensor,
        label_sample: torch.Tensor,
    ) -> torch.Tensor:
        """The top-level state of ``field`` at each label record ``[ΣN, H]``.

        That is the state after the sample's last top-level record with time
        ``<=`` the label's time, or zeros before the first.
        """
        if time is None:  # resampled top level: record i is at time i * step
            processor = self.processors[field]
            resample = processor.levels[processor.depth].get("resample")
            assert resample is not None  # __init__ allows no other untimed top level
            _, position = _groups(batch)
            time = position.to(states.dtype) * float(resample["step"])
        counts = batch.diff()
        n, width = len(counts), int(counts.max()) if len(counts) else 0
        padded = time.new_full((n, max(width, 1)), float("inf"))
        group, position = _groups(batch)
        padded[group, position] = time.to(padded.dtype)
        seen = torch.searchsorted(
            padded[label_sample],
            label_time.to(padded.dtype).unsqueeze(1),
            right=True,
        )[:, 0]
        index = (batch[:-1][label_sample] + seen - 1).clamp(min=0)
        if len(states) == 0:
            return states.new_zeros(len(label_time), states.shape[1])
        picked = states[index]
        return torch.where((seen > 0).unsqueeze(1), picked, torch.zeros_like(picked))

    def forward(self, **batch: Any) -> dict[str, torch.Tensor]:
        """Predicts the label of a batch of :func:`~pyhealth.datasets.get_dataloader`.

        Returns:
            ``loss``, ``y_prob``, ``y_true`` and ``logit``; also ``embed``, the
            concatenated summaries and statics, if the batch has ``embed=True``.
        """
        embedded = self.embedding(
            {field: batch[field] for field in self.sequence_fields}
        )
        label: tuple[torch.Tensor, ...] = ()
        label_time = label_sample = torch.zeros(0, dtype=torch.long)
        if self.per_step:
            # Packed label records: (values [ΣN, 1], time [ΣN], offsets [B + 1]).
            label = tuple(t.to(self.device) for t in batch[self.label_key])
            label_time = label[1]
            label_sample, _ = _groups(label[2])
        parts = []
        for field in self.sequence_fields:
            encoder = cast(_FieldRNN, self.encoders[field])
            summary, states, time = encoder(embedded[field])
            if not self.per_step:
                parts.append(summary)
                continue
            assert states is not None  # a series top level has states
            out = embedded[field]
            offsets = out[encoder.depth]["offsets"]
            parts.append(
                self._at_labels(field, states, time, offsets, label_time, label_sample)
            )
        for field in self.static_fields:
            x = batch[field].to(self.device).float().flatten(1)
            if field in self.static_projections:
                x = self.static_projections[field](x)
            parts.append(x[label_sample] if self.per_step else x)
        embed = torch.cat(parts, dim=-1)
        logits = self.fc(embed)
        y_true = label if self.per_step else batch[self.label_key].to(self.device)
        results = {
            "loss": self.get_loss_function()(logits, y_true),
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if batch.get("embed", False):
            results["embed"] = embed
        return results
