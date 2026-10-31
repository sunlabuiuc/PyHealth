from collections.abc import Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel
from pyhealth.processors import SequenceProcessorV2, TensorProcessor
from pyhealth.processors.sequence_processor_v2 import Resample

from .embedding import EmbeddingModel
from .rnn import _groups, _pool
from .sequence_embedding import (
    LevelEmbedding,
    MaskMode,
    PretrainedEmbedding,
    SequenceEmbeddingModel,
)


class LogisticRegression(BaseModel):
    """Logistic/Linear regression baseline model.

    This model uses embeddings from different input features and applies a single
    linear transformation (no hidden layers or non-linearity) to produce predictions.
    
    - For classification tasks: acts as logistic regression
    - For regression tasks: acts as linear regression
    
    The model automatically handles different input types through the EmbeddingModel,
    pools sequence dimensions, concatenates all feature embeddings, and applies a
    final linear layer.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        **kwargs: other parameters (for compatibility).

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...         {
        ...             "patient_id": "patient-0",
        ...             "visit_id": "visit-0",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [1.0, 2.0, 3.5, 4],
        ...             "label": 0,
        ...         },
        ...         {
        ...             "patient_id": "patient-1",
        ...             "visit_id": "visit-1",
        ...             "conditions": ["cond-33", "cond-86", "cond-80"],
        ...             "procedures": [5.0, 2.0, 3.5, 4],
        ...             "label": 1,
        ...         },
        ...     ]
        >>> input_schema = {"conditions": "sequence",
        ...                 "procedures": "tensor"}
        >>> output_schema = {"label": "binary"}
        >>> dataset = create_sample_dataset(samples=samples,
        ...                        input_schema=input_schema,
        ...                        output_schema=output_schema,
        ...                        dataset_name="test")
        >>>
        >>> from pyhealth.models import LogisticRegression
        >>> model = LogisticRegression(dataset=dataset)
        >>>
        >>> from pyhealth.datasets import get_dataloader
        >>> train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)
        >>> data_batch = next(iter(train_loader))
        >>>
        >>> ret = model(**data_batch)
        >>> print(ret)
        {
            'loss': tensor(0.6931, grad_fn=<BinaryCrossEntropyWithLogitsBackward0>),
            'y_prob': tensor([[0.5123],
                            [0.4987]], grad_fn=<SigmoidBackward0>),
            'y_true': tensor([[1.],
                            [0.]]),
            'logit': tensor([[0.0492],
                            [-0.0052]], grad_fn=<AddmmBackward0>)
        }
        >>>

    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        **kwargs,
    ):
        super(LogisticRegression, self).__init__(dataset)
        self.embedding_dim = embedding_dim

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        # Use the EmbeddingModel to handle embedding logic
        self.embedding_model = EmbeddingModel(dataset, embedding_dim)

        # Single linear layer (no hidden layers, no activation)
        output_size = self.get_output_size()
        self.fc = nn.Linear(len(self.feature_keys) * self.embedding_dim, output_size)

    @staticmethod
    def mean_pooling(x, mask):
        """Mean pooling over the middle dimension of the tensor.

        Args:
            x: tensor of shape (batch_size, seq_len, embedding_dim)
            mask: tensor of shape (batch_size, seq_len)

        Returns:
            x: tensor of shape (batch_size, embedding_dim)

        Examples:
            >>> x.shape
            [128, 5, 32]
            >>> mean_pooling(x, mask).shape
            [128, 32]
        """
        return x.sum(dim=1) / mask.sum(dim=1, keepdim=True)

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: keyword arguments for the model. The keys must contain
                all the feature keys and the label key.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with the following keys:
                - loss: a scalar tensor representing the loss.
                - y_prob: a tensor representing the predicted probabilities.
                - y_true: a tensor representing the true labels.
                - logit: a tensor representing the logits.
                - embed (optional): a tensor representing the patient
                    embeddings if requested.
        """
        patient_emb = []

        # Preprocess inputs for EmbeddingModel
        processed_inputs = {}
        reshape_info = {}  # Track which inputs were reshaped

        for feature_key in self.feature_keys:
            x = kwargs[feature_key]

            # Convert to tensor if not already
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, device=self.device)
            else:
                x = x.to(self.device)

            # Handle 3D input: (patient, event, # of codes) -> flatten to 2D
            if x.dim() == 3:
                batch_size, seq_len, inner_len = x.shape
                x = x.view(batch_size, seq_len * inner_len)
                reshape_info[feature_key] = {
                    "original_shape": (batch_size, seq_len, inner_len),
                    "was_3d": True,
                    "expanded": False,
                }
            elif x.dim() == 1:
                x = x.unsqueeze(0)
                reshape_info[feature_key] = {"was_3d": False, "expanded": True}
            else:
                reshape_info[feature_key] = {"was_3d": False, "expanded": False}

            processed_inputs[feature_key] = x

        # Pass through EmbeddingModel
        embedded = self.embedding_model(processed_inputs)

        for feature_key in self.feature_keys:
            x = embedded[feature_key]

            info = reshape_info[feature_key]
            if info.get("expanded") and x.dim() > 1:
                x = x.squeeze(0)

            # Handle different tensor dimensions for pooling
            if x.dim() == 3:
                # Case: (batch, seq_len, embedding_dim) - apply mean pooling
                mask = (x.sum(dim=-1) != 0).float()
                if mask.sum(dim=-1, keepdim=True).any():
                    x = self.mean_pooling(x, mask)
                else:
                    x = x.mean(dim=1)
            elif x.dim() == 2:
                # Case: (batch, embedding_dim) - already pooled, use as is
                pass
            else:
                raise ValueError(f"Unsupported tensor dimension: {x.dim()}")

            patient_emb.append(x)

        # Concatenate all feature embeddings
        patient_emb = torch.cat(patient_emb, dim=1)

        # Apply single linear layer (no activation)
        logits = self.fc(patient_emb)
        
        # Obtain y_true, loss, y_prob
        y_true = kwargs[self.label_key].to(self.device)
        loss = self.get_loss_function()(logits, y_true)
        y_prob = self.prepare_y_prob(logits)
        
        results = {
            "loss": loss,
            "y_prob": y_prob,
            "y_true": y_true,
            "logit": logits,
        }
        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results


class Unified_LogisticRegression(BaseModel):
    """Linear prediction over bottom-up means of learned record embeddings.

    Accepts ``sequence_v2`` fields and static ``tensor`` fields. At each level,
    a record's embedding is concatenated with its children's mean summary,
    then records are mean-pooled within their parent. Levels may contain
    multiple token namespaces, numeric values, both, or only child groups.
    Spatial value axes are flattened and projected within each record by
    :class:`SequenceEmbeddingModel`. Sequence order and timestamps do not
    become input features. Empty groups give zero summaries; parents with
    empty children still contribute their own embeddings.

    Field summaries and flattened static tensors are concatenated and passed
    to a single linear head. This is a linear head over learned embeddings,
    rather than classical logistic regression on raw inputs. Nested means
    weight parents equally, not individual leaves across different parents.

    Per-step binary/regression labels use the cumulative mean of top-level
    records with time ``<=`` the label time, or zeros before the first record.
    Every sequence field must then have a ``series`` top level. Resampled
    record ``i`` has time ``i * step``. Each top-level timestamp must indicate
    when the entire record, including its children, is available. Static
    vectors are repeated for each label record of their sample.

    Args:
        dataset: Dataset with fitted processors and exactly one label.
        embedding_dim: One :class:`LevelEmbedding` per level of each V2
            field, innermost first. Use ``{}`` for static-only datasets.
        mask: ``"ignore"`` or ``"concat"`` for numeric missing indicators.
        pretrain: Pretrained token namespace embeddings.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> dataset = create_sample_dataset(
        ...     [{"patient_id": str(i), "age": [50.0 + i], "y": i}
        ...      for i in range(2)],
        ...     {"age": "tensor"}, {"y": "binary"},
        ... )
        >>> model = Unified_LogisticRegression(dataset, embedding_dim={})
        >>> model(**next(iter(get_dataloader(dataset, batch_size=2))))["logit"].shape
        torch.Size([2, 1])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: Mapping[str, Sequence[LevelEmbedding]],
        mask: MaskMode = "ignore",
        pretrain: Sequence[PretrainedEmbedding] | None = None,
    ) -> None:
        super().__init__(dataset)
        if len(self.label_keys) != 1:
            raise ValueError("Unified pooling models support exactly one label.")
        if not self.feature_keys:
            raise ValueError("At least one input field is required.")
        self.label_key = self.label_keys[0]
        self.per_step = self.mode in ("timeseries_binary", "timeseries_regression")
        self.processors: dict[str, SequenceProcessorV2] = {}
        self.static_fields: list[str] = []
        for field in self.feature_keys:
            if isinstance(
                processor := dataset.input_processors[field], SequenceProcessorV2
            ):
                self.processors[field] = processor
                if (
                    self.per_step
                    and self.processors[field].levels[-1].get("ordering", "sequence")
                    != "series"
                ):
                    raise ValueError(
                        f"'{field}': per-step labels require a 'series' top level."
                    )
            elif isinstance(dataset.input_processors[field], TensorProcessor):
                self.static_fields.append(field)
            else:
                raise TypeError(
                    f"'{field}': use 'sequence_v2' or static 'tensor' inputs."
                )
        self.embedding = SequenceEmbeddingModel(
            dataset, embedding_dim=embedding_dim, mask=mask, pretrain=pretrain
        )
        self.encoders = nn.ModuleDict()
        self.static_encoders = nn.ModuleDict()
        self.field_dims: dict[str, int] = {}
        for field, processor in self.processors.items():
            self.encoders[field] = nn.ModuleList(
                nn.Identity() for _ in range(processor.depth + 1)
            )
            self.field_dims[field] = sum(self.embedding.embedding_dim_per_level(field))
            if self.embedding.embedding_dim_per_level(field)[0] == 0:
                raise ValueError(
                    f"'{field}': the innermost level has nothing to encode."
                )
        for field in self.static_fields:
            self.static_encoders[field] = nn.Identity()
            self.field_dims[field] = int(torch.as_tensor(dataset[0][field]).numel())
        self.fc = nn.Linear(sum(self.field_dims.values()), self.get_output_size())

    def forward(
        self, **batch: torch.Tensor | tuple[torch.Tensor, ...] | bool
    ) -> dict[str, Any]:
        """Return loss, probabilities, targets, logits, and optionally ``embed``."""
        embedded = self.embedding({field: batch[field] for field in self.processors})
        label: tuple[torch.Tensor, ...] = ()
        label_sample = torch.empty(0, dtype=torch.long, device=self.device)
        if self.per_step:
            label = tuple(
                t.to(self.device)
                for t in cast(tuple[torch.Tensor, ...], batch[self.label_key])
            )
            label_sample, _ = _groups(label[2])
        parts: list[torch.Tensor] = []
        for field, processor in self.processors.items():
            summary: torch.Tensor | None = None
            for level in range(processor.depth + 1):
                rows: list[torch.Tensor] = []
                if embedded[field][level]["embedding"] is not None:
                    rows.append(embedded[field][level]["embedding"])
                if summary is not None:
                    rows.append(summary)
                x = cast(nn.ModuleList, self.encoders[field])[level](
                    torch.cat(rows, dim=-1)
                )
                offsets = embedded[field][level]["offsets"]
                if self.per_step and level == processor.depth:
                    # Search each sample's times, then subtract cumulative sums
                    # at its start to obtain an independent causal prefix mean.
                    group, position = _groups(offsets)
                    time = embedded[field][level]["time"]
                    if time is None:
                        time = position.to(x.dtype) * float(
                            cast(Resample, processor.levels[level].get("resample"))[
                                "step"
                            ]
                        )
                    padded = time.new_full(
                        (len(offsets) - 1, max(int(offsets.diff().max()), 1)),
                        float("inf"),
                    )
                    padded[group, position] = time
                    seen = torch.searchsorted(
                        padded[label_sample],
                        cast(tuple[torch.Tensor, ...], label)[1]
                        .to(time.dtype)
                        .unsqueeze(1),
                        right=True,
                    )[:, 0]
                    prefix = torch.cat([x.new_zeros(1, x.shape[1]), x.cumsum(0)])
                    summary = (
                        prefix[offsets[:-1][label_sample] + seen]
                        - prefix[offsets[:-1][label_sample]]
                    ) / seen.clamp(min=1).unsqueeze(1).to(x.dtype)
                else:
                    summary = _pool(x, offsets)
            parts.append(cast(torch.Tensor, summary))
        for field in self.static_fields:
            x = self.static_encoders[field](
                cast(torch.Tensor, batch[field]).to(self.device).float().flatten(1)
            )
            parts.append(x[label_sample] if self.per_step else x)
        embed = torch.cat(parts, dim=-1)
        logits = self.fc(embed)
        y_true = (
            label
            if self.per_step
            else cast(torch.Tensor, batch[self.label_key]).to(self.device)
        )
        results: dict[str, torch.Tensor | tuple[torch.Tensor, ...]] = {
            "loss": self.get_loss_function()(logits, y_true),
            "y_prob": self.prepare_y_prob(logits),
            "y_true": y_true,
            "logit": logits,
        }
        if batch.get("embed", False):
            results["embed"] = embed
        return results


if __name__ == "__main__":
    from pyhealth.datasets import create_sample_dataset

    samples = [
        {
            "patient_id": "patient-0",
            "visit_id": "visit-0",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [1.0, 2.0, 3.5, 4],
            "label": 0,
        },
        {
            "patient_id": "patient-1",
            "visit_id": "visit-1",
            "conditions": ["cond-33", "cond-86", "cond-80"],
            "procedures": [5.0, 2.0, 3.5, 4],
            "label": 1,
        },
    ]

    # Define input and output schemas
    input_schema = {
        "conditions": "sequence",  # sequence of condition codes
        "procedures": "tensor",  # tensor of procedure values
    }
    output_schema = {"label": "binary"}  # binary classification

    # dataset
    dataset = create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema=output_schema,
        dataset_name="test",
    )

    # data loader
    from pyhealth.datasets import get_dataloader

    train_loader = get_dataloader(dataset, batch_size=2, shuffle=True)

    # model
    model = LogisticRegression(dataset=dataset)

    # data batch
    data_batch = next(iter(train_loader))

    # try the model
    ret = model(**data_batch)
    print(ret)

    # try loss backward
    ret["loss"].backward()
