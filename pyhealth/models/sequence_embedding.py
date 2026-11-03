"""Embeddings for the packed records of :class:`SequenceProcessorV2` fields."""

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Required, TypedDict, cast

import torch
from torch import nn

from ..datasets import SampleDataset
from ..processors import SequenceProcessorV2
from ..processors.sequence_processor_v2 import FieldKey
from .base_model import BaseModel
from .embedding import init_embedding_with_pretrained

MaskMode = Literal["ignore", "concat"]


class LevelEmbedding(TypedDict, total=False):
    """Embedding sizes of one level; a plain ``dict`` with type hints.

    A key is required exactly when the level has that role, and ``{}`` is a
    level without tokens or values.

    Keys:
        tokens: One size per token namespace (``K`` sizes). Each namespace has
            its own ``nn.Embedding``; the record's token embedding is their
            concatenation.
        values: The output size of the linear layer over the values.

    Examples:
        >>> labs: LevelEmbedding = {"tokens": (8, 8), "values": 16}
        >>> sum(labs["tokens"]) + labs["values"]  # the record's embedding size
        32
    """

    tokens: Sequence[int]
    values: int


class PretrainedEmbedding(TypedDict, total=False):
    """A pretrained text-vector file for one token namespace; a plain ``dict``.

    Keys:
        field: The V2 field.
        level: The level of the field.
        namespace: The token namespace of the level.
        path: A text file with one token and its vector per line (e.g. GloVe);
            the vector size must be the namespace's embedding size. Tokens
            not in the file keep their random initialization.
        freeze: Whether to freeze the embedding. Default False.
        normalize: Reserved for future work; only None.

    Examples:
        >>> glove: PretrainedEmbedding = {
        ...     "field": "notes", "level": 0, "namespace": 0,
        ...     "path": "glove.6B.100d.txt", "freeze": True,
        ... }
        >>> glove["freeze"]
        True
    """

    field: Required[str]
    level: Required[int]
    namespace: Required[int]
    path: Required[str]
    freeze: bool
    normalize: None


def _broadcast(x: torch.Tensor, spatial: tuple[int, ...]) -> torch.Tensor:
    """``[N, D]`` -> ``[N, *spatial, D]``; an ``x`` with spatial axes is kept."""
    if x.dim() == 2 + len(spatial):
        return x
    return x.reshape(x.shape[0], *([1] * len(spatial)), x.shape[-1]).expand(
        x.shape[0], *spatial, x.shape[-1]
    )


class _FieldEmbedding(nn.Module):
    """The embedding layers of one V2 field, one level after another."""

    def __init__(
        self,
        field: str,
        processor: SequenceProcessorV2,
        dims: Sequence[LevelEmbedding],
        flatten_spatial: Sequence[bool],
        mask: MaskMode,
    ):
        super().__init__()
        self.batch_schema = processor.batch_schema()
        self.mask = mask
        masked = {level for level, role in self.batch_schema if role == "mask"}
        self.depth = processor.depth
        vocab_size, value_dim, spatial = (
            processor.vocab_size(),
            processor.value_dim(),
            processor.spatial(),
        )
        if not isinstance(dims, (list, tuple)) or len(dims) != self.depth + 1:
            raise ValueError(
                f"embedding_dim['{field}'] must have one dict per level "
                f"({self.depth + 1}), got {dims!r}."
            )
        if len(flatten_spatial) != self.depth + 1:
            raise ValueError(
                f"flatten_spatial['{field}'] must have one bool per level "
                f"({self.depth + 1}), got {flatten_spatial!r}."
            )

        # Per level: the token embeddings ("token_<k>"), the values projection
        # ("value_proj"), and how the values are laid out before it. (Not
        # "values": that would shadow ModuleDict.values().)
        self.levels = nn.ModuleList()
        self.sizes: list[int] = []  # the record embedding size per level
        self.spatial_axes: list[tuple[int, ...]] = []  # value axes kept apart
        for level, (dim, flat) in enumerate(zip(dims, flatten_spatial)):
            where = f"embedding_dim['{field}'][{level}]"
            if not isinstance(dim, dict) or not dim.keys() <= {"tokens", "values"}:
                raise ValueError(f"{where} must be a dict of 'tokens' / 'values'.")
            layers = nn.ModuleDict()
            size = 0
            namespaces = vocab_size[level]
            if namespaces is None:
                if "tokens" in dim:
                    raise ValueError(f"{where}: level {level} has no tokens.")
            else:
                sizes = dim.get("tokens")
                if (
                    not isinstance(sizes, (list, tuple))
                    or len(sizes) != len(namespaces)
                    or not all(isinstance(s, int) and s > 0 for s in sizes)
                ):
                    raise ValueError(
                        f"{where}['tokens'] must be {len(namespaces)} positive "
                        f"sizes, one per token namespace, got {sizes!r}."
                    )
                for k, (vocab, width) in enumerate(zip(namespaces, sizes)):
                    layers[f"token_{k}"] = nn.Embedding(vocab, width, padding_idx=0)
                size += sum(sizes)

            shape, flags = value_dim[level], spatial[level]
            kept: tuple[int, ...] = ()
            if shape is None:
                if "values" in dim:
                    raise ValueError(f"{where}: level {level} has no values.")
                if not flat:
                    raise ValueError(
                        f"flatten_spatial['{field}'][{level}]: level {level} has "
                        "no values."
                    )
            else:
                width = dim.get("values")
                if not isinstance(width, int) or width <= 0:
                    raise ValueError(
                        f"{where}['values'] must be a positive size, got {width!r}."
                    )
                assert flags is not None  # a level with values has spatial flags
                if not flat:
                    kept = tuple(axis for axis, on in enumerate(flags) if on)
                    if not kept:
                        raise ValueError(
                            f"flatten_spatial['{field}'][{level}] is False, but no "
                            "value axis of the level is spatial."
                        )
                features = math.prod(
                    size for axis, size in enumerate(shape) if axis not in kept
                )
                # With "concat", the mask's features join the values'.
                concat = level in masked and mask == "concat"
                layers["value_proj"] = nn.Linear((2 if concat else 1) * features, width)
                size += width
            self.levels.append(layers)
            self.sizes.append(size)
            self.spatial_axes.append(kept)

        # Spatial shape of each level's embeddings, e.g. (16,) for kept ECG
        # samples; a level's token embedding is broadcast over it.
        self.spatial_shapes = [
            tuple(value_dim[level][axis] for axis in kept)  # type: ignore[index]
            for level, kept in enumerate(self.spatial_axes)
        ]

    def layer(self, level: int, name: str) -> nn.Module:
        """One layer of a level: ``token_<k>`` or ``value_proj``."""
        return cast(nn.ModuleDict, self.levels[level])[name]

    def embed_level(
        self, level: int, fields: Mapping[FieldKey, torch.Tensor]
    ) -> torch.Tensor | None:
        """The ``[N, *spatial, D]`` embeddings of one level's records, or None."""
        parts = []
        tokens = fields.get((level, "tokens"))
        if tokens is not None:
            tokens = tokens.unsqueeze(1) if tokens.dim() == 1 else tokens
            parts.append(
                torch.cat(
                    [
                        self.layer(level, f"token_{k}")(tokens[:, k])
                        for k in range(tokens.shape[1])
                    ],
                    dim=-1,
                )
            )
        values = fields.get((level, "values"))
        if values is not None:
            kept = self.spatial_axes[level]
            others = [1 + a for a in range(values.dim() - 1) if a not in kept]

            def features(x: torch.Tensor) -> torch.Tensor:
                # [N, *S] -> [N, *S_kept, prod(S_others)], to project per position.
                x = x.permute(0, *(1 + a for a in kept), *others)
                return x.reshape(
                    *x.shape[: 1 + len(kept)], math.prod(x.shape[1 + len(kept) :])
                )

            x = features(values)
            mask = fields.get((level, "mask"))
            if mask is not None and self.mask == "concat":
                x = torch.cat([x, features(mask.float())], dim=-1)
            parts.append(self.layer(level, "value_proj")(x))
        if not parts:
            return None
        spatial = self.spatial_shapes[level]
        return torch.cat([_broadcast(p, spatial) for p in parts], dim=-1)

    def forward(self, inputs: Sequence[torch.Tensor]) -> dict[Any, Any]:
        fields = dict(zip(self.batch_schema, inputs))
        levels = {
            level: {
                "embedding": self.embed_level(level, fields),
                "offsets": fields[(level, "offsets")],
                "time": fields.get((level, "time")),
            }
            for level in range(self.depth + 1)
        }
        return levels


class SequenceEmbeddingModel(BaseModel):
    """Embeds the packed records of every :class:`SequenceProcessorV2` field.

    Like :class:`~pyhealth.models.EmbeddingModel`, it is a building block for
    other models: :meth:`forward` maps each V2 field of a batch (the tuple of
    :meth:`SequenceProcessorV2.collate`) to embeddings. Fields of other
    processors are ignored and left out of the output.

    A record's embedding at level ``l`` is the concatenation of one
    ``nn.Embedding`` per token namespace (``<pad>`` embeds to zeros) and a
    linear projection of its values, so its size is
    ``D_l = sum(tokens) + values``. ``time`` is passed through unchanged.

    Args:
        dataset: The dataset, with fitted processors.
        embedding_dim: Per V2 field, one :class:`LevelEmbedding` per level,
            innermost first. Every V2 field must be listed.
        flatten_spatial: Per V2 field, one bool per level (default all True).
            True flattens all value axes before the projection, giving
            ``[N, values]``. False flattens only the non-spatial axes and
            projects at each position of the spatial ones, giving
            ``[N, *spatial, values]``; the level's token embedding is
            broadcast over those positions.
        pretrain: :class:`PretrainedEmbedding` entries that initialize token
            namespaces from text-vector files.
        mask: How levels with a ``mask`` (``missing_indicator``) use it.

            - ``"ignore"`` (default): the mask is not used.
            - ``"concat"``: the values and the mask are projected together,
              ``Linear(cat(values, mask))``.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [
        ...     {"patient_id": "p0", "visits": {"children": [{"tokens": ["A", "B"]}, {"tokens": ["C"]}]}, "label": 0},
        ...     {"patient_id": "p1", "visits": {"children": [{"tokens": ["C"]}]}, "label": 1},
        ... ]
        >>> levels = [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}]
        >>> dataset = create_sample_dataset(
        ...     samples, {"visits": ("sequence_v2", {"levels": levels})}, {"label": "binary"}
        ... )
        >>> model = SequenceEmbeddingModel(dataset, embedding_dim={"visits": [{"tokens": (8,)}, {}]})
        >>> proc = dataset.input_processors["visits"]
        >>> batch = proc.collate([dataset[0]["visits"], dataset[1]["visits"]])
        >>> out = model({"visits": batch})["visits"]
        >>> out[0]["embedding"].shape, out[0]["offsets"].tolist(), out[1]["offsets"].tolist()
        (torch.Size([4, 8]), [0, 2, 3, 4], [0, 2, 3])
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: Mapping[str, Sequence[LevelEmbedding]],
        flatten_spatial: Mapping[str, Sequence[bool]] | None = None,
        pretrain: Sequence[PretrainedEmbedding] | None = None,
        mask: MaskMode = "ignore",
    ):
        super().__init__(dataset)
        if mask not in ("ignore", "concat"):
            raise ValueError(f"mask must be 'ignore' or 'concat', got {mask!r}.")
        processors = {
            field: processor
            for field, processor in dataset.input_processors.items()
            if isinstance(processor, SequenceProcessorV2)
        }
        flatten_spatial = dict(flatten_spatial or {})
        for name, config in (
            ("embedding_dim", embedding_dim),
            ("flatten_spatial", flatten_spatial),
        ):
            unknown = set(config) - processors.keys()
            if unknown:
                raise ValueError(
                    f"{name} names {sorted(unknown)}, which are not "
                    f"SequenceProcessorV2 fields ({sorted(processors)})."
                )
        missing = processors.keys() - set(embedding_dim)
        if missing:
            raise ValueError(f"embedding_dim has no entry for {sorted(missing)}.")
        self.embedding_layers = nn.ModuleDict(
            {
                field: _FieldEmbedding(
                    field,
                    processor,
                    embedding_dim[field],
                    flatten_spatial.get(field, [True] * (processor.depth + 1)),
                    mask,
                )
                for field, processor in processors.items()
            }
        )
        seen = set()
        for entry in pretrain or ():
            self._load_pretrained(entry, processors, seen)

    def embedding_dim_per_level(self, field: str) -> tuple[int, ...]:
        """The record embedding size ``D_l`` of each level of a V2 field.

        Examples:
            >>> from pyhealth.datasets import create_sample_dataset
            >>> labs = {"tokens": [("Glucose", "mg/dL")], "values": [[1.0]]}
            >>> samples = [{"patient_id": f"p{i}", "labs": labs, "y": i} for i in range(2)]
            >>> dataset = create_sample_dataset(
            ...     samples, {"labs": ("sequence_v2", {"levels": [{"tokens": 2, "values": [1]}]})}, {"y": "binary"}
            ... )
            >>> model = SequenceEmbeddingModel(dataset, embedding_dim={"labs": [{"tokens": (8, 4), "values": 16}]})
            >>> model.embedding_dim_per_level("labs")
            (28,)
        """
        return tuple(cast(_FieldEmbedding, self.embedding_layers[field]).sizes)

    def _load_pretrained(
        self,
        entry: PretrainedEmbedding,
        processors: Mapping[str, SequenceProcessorV2],
        seen: set[tuple[str, int, int]],
    ) -> None:
        if (
            not isinstance(entry, dict)
            or not {"field", "level", "namespace", "path"} <= entry.keys()
        ):
            raise ValueError(
                f"pretrain entries need 'field', 'level', 'namespace' and 'path', "
                f"got {entry!r}."
            )
        if entry.get("normalize") is not None:
            raise NotImplementedError(
                "pretrain 'normalize' is reserved for future work and only "
                "accepts None for now."
            )
        field, level, namespace = entry["field"], entry["level"], entry["namespace"]
        if field not in processors:
            raise ValueError(f"pretrain: '{field}' is not a SequenceProcessorV2 field.")
        vocab_size = processors[field].vocab_size()
        if not (
            isinstance(level, int)
            and 0 <= level < len(vocab_size)
            and vocab_size[level] is not None
            and isinstance(namespace, int)
            and 0 <= namespace < len(vocab_size[level])  # type: ignore[arg-type]
        ):
            raise ValueError(
                f"pretrain: '{field}' has no token namespace {namespace} at level "
                f"{level}."
            )
        if (field, level, namespace) in seen:
            raise ValueError(
                f"pretrain: '{field}' level {level} namespace {namespace} is listed "
                "twice."
            )
        seen.add((field, level, namespace))
        layers = cast(_FieldEmbedding, self.embedding_layers[field])
        embedding = layers.layer(level, f"token_{namespace}")
        assert isinstance(embedding, nn.Embedding)
        init_embedding_with_pretrained(
            embedding,
            processors[field].vocab(level, namespace),
            entry["path"],
            embedding.embedding_dim,
            freeze=entry.get("freeze", False),
        )

    # BaseModel.forward(**kwargs) is the predictor contract (loss, y_prob, ...);
    # like EmbeddingModel, this building block embeds given inputs instead.
    def forward(  # type: ignore[override]
        self, inputs: Mapping[str, Any]
    ) -> dict[str, dict[Any, Any]]:
        """Embeds the V2 fields of a batch.

        Args:
            inputs: Batch fields, e.g. a batch of
                :func:`~pyhealth.datasets.get_dataloader`. Only the V2 fields
                are read; others are ignored.

        Returns:
            Per V2 field, per level: ``{"embedding": [N_l, *spatial, D_l]
            or None, "offsets": the level's offsets (the batch offsets at
            the top level), "time": [N_l] or None}``.
        """
        device = self.device
        return {
            field: layers(tuple(tensor.to(device) for tensor in inputs[field]))
            for field, layers in self.embedding_layers.items()
            if field in inputs
        }
