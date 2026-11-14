"""Bridges from legacy feature processors to the unified packed-record format.

A :class:`UnifiedBridge` wraps a fitted
:class:`~pyhealth.processors.LegacyFeatureProcessor` and describes its output
the way :class:`~pyhealth.processors.JaggedFeatureProcessor` does
(:meth:`~JaggedFeatureProcessor.schema`,
:meth:`~JaggedFeatureProcessor.vocab_size`, ...), so models built for the
unified format can read legacy fields. The legacy processor still processes
and collates the data; the bridge only converts each collated batch with
:meth:`UnifiedBridge.transform_batch`.

:func:`transform_processor` replaces every legacy processor of a dict with
its bridge, and :func:`transform_batch` converts the matching fields of a
batch.
"""

import copy
from abc import abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, ClassVar

import torch

from .audio_processor import AudioProcessor
from .base_processor import (
    FeatureProcessor,
    LegacyFeatureProcessor,
    TokenProcessorInterface,
)
from .cehr_processor import CehrProcessor
from .deep_nested_sequence_processor import (
    DeepNestedFloatsProcessor,
    DeepNestedSequenceProcessor,
)
from .image_processor import ImageProcessor
from .jagged_processor import JaggedFeatureProcessor, LevelConfig
from .multi_hot_processor import MultiHotProcessor
from .nested_multihot_processor import NestedMultiHotProcessor
from .nested_sequence_processor import NestedFloatsProcessor, NestedSequenceProcessor
from .sequence_processor import SequenceProcessor
from .stagenet_processor import StageNetProcessor, StageNetTensorProcessor
from .temporal_timeseries_processor import TemporalTimeseriesProcessor
from .tensor_processor import TensorProcessor
from .time_image_processor import TimeImageProcessor
from .timeseries_processor import TimeseriesProcessor

_PAD = TokenProcessorInterface.PAD
_UNK = TokenProcessorInterface.UNK

_BRIDGES: dict[type, type["UnifiedBridge"]] = {}


def _bridges[B: "UnifiedBridge"](legacy_cls: type) -> Callable[[type[B]], type[B]]:
    """Registers a bridge class for ``legacy_cls`` (and its subclasses)."""

    def register(bridge_cls: type[B]) -> type[B]:
        bridge_cls.legacy_cls = legacy_cls
        _BRIDGES[legacy_cls] = bridge_cls
        return bridge_cls

    return register


class UnifiedBridge(JaggedFeatureProcessor):
    """A fitted legacy processor, described in the unified packed-record format.

    The bridge has every metadata method of
    :class:`~pyhealth.processors.JaggedFeatureProcessor` (``depth``,
    ``levels``, :meth:`schema`, :meth:`batch_schema`, :meth:`ordering`,
    :meth:`vocab_size`, :meth:`value_dim`, :meth:`spatial`, :meth:`vocab`
    and :meth:`validate`), built from the legacy processor's fitted state.
    It does no processing: :meth:`fit`, :meth:`process` and :meth:`collate`
    raise, as the legacy processor still does that work.
    :meth:`transform_batch` converts one field's legacy batch (the output of
    :func:`~pyhealth.datasets.collate_fn_dict_with_padding`) into the tuple
    :meth:`collate` would give, in :meth:`batch_schema` order.

    :attr:`converts` is set once, when the bridge is made: a bridge made from
    a legacy processor converts, and one made from another bridge (by
    :func:`transform_processor`, for a model nested in one that already
    converts) passes batches through. Only converting bridges are applied
    by the module-level :func:`transform_batch`.

    The levels never configure imputation, normalization, resampling or code
    mapping: the legacy processor has already applied them. A level with
    ``missing_indicator`` names the imputation the legacy processor used, as
    the unified configuration requires one.

    Limitations:
        The legacy collate pads samples with 0 and keeps no lengths, so
        each sample's *trailing* padding is dropped and everything before it
        is kept. For a token field, that is its trailing ``<pad>`` ids (index
        0): a ``<pad>`` in between, such as the left padding of CEHR tasks,
        stays a ``<pad>`` record. A float field without a mask cannot tell
        padding from data, so its trailing rows whose values are all 0 (or
        NaN) are dropped: a sample whose real last rows are all exactly 0
        loses them. The same rule drops trailing empty visits and groups of
        nested token fields, while empty ones in between are kept.

    Args:
        legacy: The fitted legacy processor.
        levels: The unified configuration of its output.
        vocabs: Each token level's vocabulary (one namespace), with
            ``<pad>`` at 0, ``<unk>`` at 1 and the other ids contiguous.

    Examples:
        >>> from pyhealth.processors import SequenceProcessor
        >>> from pyhealth.processors.unified_bridge import SequenceToUnifiedBridge
        >>> legacy = SequenceProcessor()
        >>> legacy.fit([{"x": ["A", "B"]}], "x")
        >>> bridge = SequenceToUnifiedBridge(legacy)
        >>> bridge.schema(), bridge.vocab_size()
        (((0, 'tokens'),), {0: (4,)})
        >>> bridge.process(["A"])
        Traceback (most recent call last):
        ...
        NotImplementedError: SequenceToUnifiedBridge does not process data: SequenceProcessor processes it, and transform_batch() converts its batches.
    """

    legacy_cls: ClassVar[type]
    """The legacy processor class this bridge converts."""

    def __init__(
        self,
        legacy: LegacyFeatureProcessor,
        levels: Sequence[LevelConfig],
        vocabs: Mapping[int, Mapping[Any, int]] | None = None,
    ) -> None:
        super().__init__(levels)
        self.legacy = legacy
        """The wrapped legacy processor."""
        self.converts = True
        """Whether :func:`transform_batch` converts this field's batches."""
        self.learns_statistics = False
        for level, vocab in (vocabs or {}).items():
            self._vocabs[level] = [_checked_vocab(type(legacy).__name__, vocab)]
        self._fitted = True

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        raise NotImplementedError(self._no_processing())

    def process(self, value: Any) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError(self._no_processing())

    def collate(self, items: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
        raise NotImplementedError(
            f"{self._no_processing()} Build data loaders from the original "
            "dataset, not from a BridgeDataset."
        )

    def _no_processing(self) -> str:
        return (
            f"{type(self).__name__} does not process data: "
            f"{type(self.legacy).__name__} processes it, and transform_batch() "
            "converts its batches."
        )

    @abstractmethod
    def transform_batch(self, value: Any) -> tuple[torch.Tensor, ...]:
        """Converts one field's legacy batch into a :meth:`batch_schema` tuple."""

    def _passing_through(self) -> "UnifiedBridge":
        """A copy with the same metadata that passes batches through."""
        if not self.converts:
            return self
        bridge = copy.copy(self)
        bridge.converts = False
        return bridge

    def __repr__(self) -> str:
        suffix = "" if self.converts else ", converts=False"
        return f"{type(self).__name__}({self.legacy!r}{suffix})"


def _checked_vocab(name: str, vocab: Mapping[Any, int]) -> dict[Any, int]:
    """The vocabulary as a dict, checking the ids the unified format assumes."""
    vocab = dict(vocab)
    ids = sorted(vocab.values())
    if ids != list(range(len(ids))) or vocab.get("<pad>") != _PAD or vocab.get("<unk>") != _UNK:
        raise ValueError(
            f"{name}: the vocabulary must have <pad> at {_PAD}, <unk> at {_UNK} "
            "and contiguous ids to be bridged."
        )
    return vocab


# ----------------------------------------------------------------------
# Packing helpers
# ----------------------------------------------------------------------


def _offsets(counts: torch.Tensor) -> torch.Tensor:
    """CSR offsets ``[n + 1]`` from per-group record counts ``[n]``."""
    counts = counts.to(torch.long)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def _single(n: int, device: torch.device) -> torch.Tensor:
    """Offsets of ``n`` groups of one record each."""
    return torch.arange(n + 1, dtype=torch.long, device=device)


def _kept_length(empty: torch.Tensor) -> torch.Tensor:
    """Per row of ``empty [n, L]``, the length without its trailing True run."""
    if empty.shape[1] == 0:
        return torch.zeros(empty.shape[0], dtype=torch.long, device=empty.device)
    position = torch.arange(1, empty.shape[1] + 1, device=empty.device)
    return (position * ~empty).amax(dim=1)


def _prefix(length: torch.Tensor, width: int) -> torch.Tensor:
    """``[n, width]`` mask of the first ``length[i]`` positions of each row."""
    return torch.arange(width, device=length.device) < length.unsqueeze(1)


def _is_zero(x: torch.Tensor) -> torch.Tensor:
    return (x == 0) | torch.isnan(x)


def _values(x: torch.Tensor) -> torch.Tensor:
    """Float32 values with NaN (which the unified format rejects) set to 0."""
    return torch.nan_to_num(x.to(torch.float32), nan=0.0)


def _tokens(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The ids of ``ids [n, L]`` before each row's trailing ``<pad>``, and offsets."""
    length = _kept_length(ids == _PAD)
    return ids[_prefix(length, ids.shape[1])].to(torch.long), _offsets(length)


def _nested_tokens(ids: torch.Tensor) -> list[torch.Tensor]:
    """``[B, ..., C]`` ids -> tokens, then offsets from the bottom level up.

    Each axis but the last is a level of groups; a group whose ids are all
    ``<pad>`` is padding when it trails its parent, and an empty record when
    it does not.
    """
    groups = ids
    offsets: list[torch.Tensor] = []  # from the batch offsets down
    while groups.dim() > 2:
        # groups: [n, k, ..., C]; keep each row's children before its
        # trailing all-<pad> ones.
        length = _kept_length((groups == _PAD).flatten(2).all(dim=2))
        offsets.append(_offsets(length))
        groups = groups[_prefix(length, groups.shape[1])]
    tokens, bottom = _tokens(groups)
    offsets.append(bottom)
    return [tokens, *reversed(offsets)]


def _trailing_rows(
    rows: torch.Tensor, padding: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keeps each sample's rows of ``rows [B, L, ...]`` before its trailing padding.

    Returns the kept rows ``[N, ...]`` and the batch offsets ``[B + 1]``.
    """
    length = _kept_length(padding)
    return rows[_prefix(length, rows.shape[1])], _offsets(length)


# ----------------------------------------------------------------------
# Token processors
# ----------------------------------------------------------------------


@_bridges(SequenceProcessor)
class SequenceToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.SequenceProcessor` as one ``sequence`` level of tokens.

    The batch ``[B, L]`` keeps each sample's ids before its trailing
    ``<pad>`` (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.datasets import collate_fn_dict_with_padding
        >>> from pyhealth.processors import SequenceProcessor
        >>> legacy = SequenceProcessor()
        >>> legacy.fit([{"x": ["A", "B", "C"]}], "x")
        >>> bridge = SequenceToUnifiedBridge(legacy)
        >>> batch = collate_fn_dict_with_padding(
        ...     [{"x": legacy.process(["A", "B"])}, {"x": legacy.process(["C"])}]
        ... )
        >>> bridge.transform_batch(batch["x"])
        (tensor([2, 3, 4]), tensor([0, 2, 3]))
    """

    def __init__(self, legacy: SequenceProcessor) -> None:
        super().__init__(
            legacy, [{"ordering": "sequence", "tokens": 1}], {0: legacy.code_vocab}
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _tokens(value)


@_bridges(CehrProcessor)
class CehrToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.CehrProcessor` as one ``sequence`` level of tokens.

    CEHR tasks left-pad their concept lists with ``<pad>``: those leading
    ``<pad>`` ids stay ``<pad>`` records, in their positions, and only
    trailing ``<pad>`` ids are dropped (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.processors import CehrProcessor
        >>> legacy = CehrProcessor()
        >>> _ = legacy.fit([{"x": ["A"]}], "x")
        >>> bridge = CehrToUnifiedBridge(legacy)
        >>> bridge.transform_batch(legacy.process(["<pad>", "<cls>", "A"]).unsqueeze(0))
        (tensor([0, 2, 6]), tensor([0, 3]))
    """

    def __init__(self, legacy: CehrProcessor) -> None:
        vocab = legacy.vocab
        if vocab.vocab_size != len(vocab.token_to_id):
            raise ValueError(
                "CehrProcessor: the vocabulary has unused ids, so it cannot be bridged."
            )
        super().__init__(
            legacy, [{"ordering": "sequence", "tokens": 1}], {0: vocab.token_to_id}
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _tokens(value)


@_bridges(NestedSequenceProcessor)
class NestedSequenceToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.NestedSequenceProcessor` as a ``sequence`` of visits.

    Each visit is a ``set`` of tokens. In the batch ``[B, V, C]``, a visit of
    ``<pad>`` ids is padding when it trails the sample (including the one
    padded visit of an empty sample) and an empty visit otherwise, so a
    sample's real trailing empty visits are dropped (see
    :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.datasets import collate_fn_dict_with_padding
        >>> from pyhealth.processors import NestedSequenceProcessor
        >>> legacy = NestedSequenceProcessor()
        >>> legacy.fit([{"x": [["A", "B"], ["C"]]}], "x")
        >>> bridge = NestedSequenceToUnifiedBridge(legacy)
        >>> batch = collate_fn_dict_with_padding(
        ...     [{"x": legacy.process([["A", "B"], [], ["C"]])}, {"x": legacy.process([["C"]])}]
        ... )
        >>> bridge.transform_batch(batch["x"])
        (tensor([2, 3, 4, 4]), tensor([0, 2, 2, 3, 4]), tensor([0, 3, 4]))
    """

    def __init__(self, legacy: NestedSequenceProcessor) -> None:
        super().__init__(
            legacy,
            [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}],
            {0: legacy.code_vocab},
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(_nested_tokens(value))


@_bridges(DeepNestedSequenceProcessor)
class DeepNestedSequenceToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.DeepNestedSequenceProcessor` as groups of visits.

    Levels, bottom up: a ``set`` of tokens per visit, a ``sequence`` of
    visits per group and a ``sequence`` of groups. Trailing all-``<pad>``
    visits of a group and trailing all-``<pad>`` groups of a sample are
    padding; empty ones in between are kept (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.processors import DeepNestedSequenceProcessor
        >>> legacy = DeepNestedSequenceProcessor()
        >>> legacy.fit([{"x": [[["A"], ["B", "C"]], [["C"]]]}], "x")
        >>> bridge = DeepNestedSequenceToUnifiedBridge(legacy)
        >>> bridge.transform_batch(legacy.process([[["A"], ["B", "C"]], [["C"]]]).unsqueeze(0))
        (tensor([2, 3, 4, 4]), tensor([0, 1, 3, 4]), tensor([0, 2, 3]), tensor([0, 2]))
    """

    def __init__(self, legacy: DeepNestedSequenceProcessor) -> None:
        super().__init__(
            legacy,
            [
                {"ordering": "set", "tokens": 1},
                {"ordering": "sequence"},
                {"ordering": "sequence"},
            ],
            {0: legacy.code_vocab},
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(_nested_tokens(value))


def _hot_tokens(hot: torch.Tensor, shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The nonzero columns of ``hot [n, V]`` (plus ``shift``) and each row's offsets."""
    keep = hot != 0
    columns = torch.arange(hot.shape[1], device=hot.device).expand_as(hot)
    return (columns[keep] + shift).to(torch.long), _offsets(keep.sum(dim=1))


@_bridges(MultiHotProcessor)
class MultiHotToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.MultiHotProcessor` as a ``set`` of tokens.

    The multi-hot vocabulary has no ``<pad>`` or ``<unk>``, so its ids are
    shifted by 2 to make room for them. The batch ``[B, V]`` gives each
    sample its nonzero columns.

    Examples:
        >>> from pyhealth.processors import MultiHotProcessor
        >>> legacy = MultiHotProcessor()
        >>> legacy.fit([{"x": ["a", "b"]}, {"x": ["c"]}], "x")
        >>> bridge = MultiHotToUnifiedBridge(legacy)
        >>> bridge.vocab(0)
        {'<pad>': 0, '<unk>': 1, 'a': 2, 'b': 3, 'c': 4}
        >>> bridge.transform_batch(legacy.process(["a", "c"]).unsqueeze(0))
        (tensor([2, 4]), tensor([0, 2]))
    """

    def __init__(self, legacy: MultiHotProcessor) -> None:
        vocab: dict[Any, int] = {"<pad>": _PAD, "<unk>": _UNK}
        vocab.update({key: index + 2 for key, index in legacy.label_vocab.items()})
        super().__init__(legacy, [{"ordering": "set", "tokens": 1}], {0: vocab})

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _hot_tokens(value, shift=2)


@_bridges(NestedMultiHotProcessor)
class NestedMultiHotToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.NestedMultiHotProcessor` as a ``sequence`` of visits.

    Each visit is the ``set`` of its nonzero columns, whose index is the
    token id. An all-zero visit is padding when it trails the sample and an
    empty visit otherwise (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.processors import NestedMultiHotProcessor
        >>> legacy = NestedMultiHotProcessor()
        >>> legacy.fit([{"x": [["A", "B"], ["C"]]}], "x")
        >>> bridge = NestedMultiHotToUnifiedBridge(legacy)
        >>> bridge.transform_batch(legacy.process([["A", "B"], [], ["C"]]).unsqueeze(0))
        (tensor([2, 3, 4]), tensor([0, 2, 2, 3]), tensor([0, 3]))
    """

    def __init__(self, legacy: NestedMultiHotProcessor) -> None:
        super().__init__(
            legacy,
            [{"ordering": "set", "tokens": 1}, {"ordering": "sequence"}],
            {0: legacy.code_vocab},
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        visits, batch = _trailing_rows(value, (value == 0).all(dim=2))
        tokens, offsets = _hot_tokens(visits, shift=0)
        return tokens, offsets, batch


@_bridges(StageNetProcessor)
class StageNetToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.StageNetProcessor` as a ``series`` of codes or visits.

    Flat codes are one ``series`` level of tokens; nested codes are a
    ``series`` of visits, each a ``set`` of tokens. The batch is the legacy
    ``(time, ids)`` pair. Times are passed through unchanged, in whatever
    unit and origin the task used: some StageNet tasks store the time since
    the previous visit rather than a timeline. A batch without time (any
    sample had none) gets time 0 for every record. Trailing ``<pad>`` ids
    and trailing all-``<pad>`` visits are padding (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.processors import StageNetProcessor
        >>> legacy = StageNetProcessor()
        >>> legacy.fit([{"x": ([0.0, 2.0], ["A", "B"])}], "x")
        >>> bridge = StageNetToUnifiedBridge(legacy)
        >>> time, ids = legacy.process(([0.0, 2.0], ["A", "B"]))
        >>> bridge.transform_batch((time.unsqueeze(0), ids.unsqueeze(0)))
        (tensor([2, 3]), tensor([0., 2.]), tensor([0, 2]))
    """

    def __init__(self, legacy: StageNetProcessor) -> None:
        if legacy._is_nested is None:
            raise ValueError(
                "StageNetProcessor: fit() saw no codes, so whether they are "
                "nested is unknown and the processor cannot be bridged."
            )
        levels: list[LevelConfig] = (
            [{"ordering": "set", "tokens": 1}, {"ordering": "series"}]
            if legacy._is_nested
            else [{"ordering": "series", "tokens": 1}]
        )
        super().__init__(legacy, levels, {0: legacy.code_vocab})

    def transform_batch(
        self, value: tuple[torch.Tensor | None, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        time, ids = value
        if self.legacy._is_nested:
            tokens, offsets, batch = _nested_tokens(ids)
        else:
            tokens, batch = _tokens(ids)
        width = ids.shape[1]
        top = _prefix(batch.diff(), width)
        if time is None:
            record_time = torch.zeros(int(top.sum()), device=ids.device)
        else:
            record_time = _values(_fit_width(time, width))[top]
        if self.legacy._is_nested:
            return tokens, offsets, record_time, batch
        return tokens, record_time, batch


def _fit_width(x: torch.Tensor, width: int) -> torch.Tensor:
    """``x [B, L]`` cut or zero-padded to ``[B, width]``."""
    if x.shape[1] >= width:
        return x[:, :width]
    return torch.nn.functional.pad(x, (0, width - x.shape[1]))


# ----------------------------------------------------------------------
# Float processors
# ----------------------------------------------------------------------


def _check_saw_values(legacy: NestedFloatsProcessor | DeepNestedFloatsProcessor) -> None:
    """Rejects a processor whose ``fit()`` saw no values (or that is unfitted).

    Its inner width then defaults to 1 rather than coming from the data. A
    processor pickled before this was recorded (``_saw_values`` None) keeps
    its width.
    """
    if legacy._saw_values is False:
        raise ValueError(
            f"{type(legacy).__name__}: fit() saw no values, so the width of a "
            "visit is unknown and the processor cannot be bridged."
        )


@_bridges(NestedFloatsProcessor)
class NestedFloatsToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.NestedFloatsProcessor` as a ``sequence`` of value rows.

    Each visit is one record with the ``[C]`` values of the fitted inner
    width; a processor whose ``fit()`` saw no values is rejected. Each
    sample's trailing all-zero (or NaN) visits are padding, so a real
    trailing visit of zeros is dropped (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.processors import NestedFloatsProcessor
        >>> legacy = NestedFloatsProcessor()
        >>> legacy.fit([{"x": [[1.0, 2.0], [3.0]]}], "x")
        >>> bridge = NestedFloatsToUnifiedBridge(legacy)
        >>> bridge.value_dim()
        {0: (2,)}
        >>> bridge.transform_batch(legacy.process([[1.0, 2.0], [3.0]]).unsqueeze(0))
        (tensor([[1., 2.],
                [3., 2.]]), tensor([0, 2]))
    """

    def __init__(self, legacy: NestedFloatsProcessor) -> None:
        _check_saw_values(legacy)
        super().__init__(
            legacy, [{"ordering": "sequence", "values": [legacy._max_inner_len]}]
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rows, batch = _trailing_rows(value, _is_zero(value).all(dim=2))
        return _values(rows), batch


@_bridges(DeepNestedFloatsProcessor)
class DeepNestedFloatsToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.DeepNestedFloatsProcessor` as groups of value rows.

    Levels, bottom up: a ``sequence`` of visits per group, each one record
    with ``[C]`` values, and a ``sequence`` of groups. A processor whose
    ``fit()`` saw no values is rejected. Trailing all-zero (or NaN) visits of
    a group and trailing all-zero groups of a sample are padding (see
    :class:`UnifiedBridge`). With ``forward_fill``, the legacy processor
    pads a group's visits by repeating its last visit, which is not zero, so
    those padded visits are kept as records.

    Examples:
        >>> from pyhealth.processors import DeepNestedFloatsProcessor
        >>> legacy = DeepNestedFloatsProcessor(forward_fill=False)
        >>> legacy.fit([{"x": [[[1.0], [2.0]], [[3.0]]]}], "x")
        >>> bridge = DeepNestedFloatsToUnifiedBridge(legacy)
        >>> bridge.transform_batch(legacy.process([[[1.0], [2.0]], [[3.0]]]).unsqueeze(0))
        (tensor([[1.],
                [2.],
                [3.]]), tensor([0, 2, 3]), tensor([0, 2]))
    """

    def __init__(self, legacy: DeepNestedFloatsProcessor) -> None:
        _check_saw_values(legacy)
        super().__init__(
            legacy,
            [
                {"ordering": "sequence", "values": [legacy._max_inner_len]},
                {"ordering": "sequence"},
            ],
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        zero = _is_zero(value)  # [B, G, V, C]
        groups, batch = _trailing_rows(value, zero.flatten(2).all(dim=2))
        rows, offsets = _trailing_rows(groups, _is_zero(groups).all(dim=2))
        return _values(rows), offsets, batch


@_bridges(StageNetTensorProcessor)
class StageNetTensorToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.StageNetTensorProcessor` as a ``series`` of value rows.

    Each time step is one record with ``[F]`` values (``[1]`` for flat
    numerics); the batch is the legacy ``(time, values)`` pair. Times are
    passed through unchanged (see :class:`StageNetToUnifiedBridge`), or 0
    when the batch has none. A sample's trailing rows are padding when their
    values are all 0 and their time is 0 (or the batch has no time), so a
    real trailing row like that is dropped (see :class:`UnifiedBridge`).

    Examples:
        >>> from pyhealth.processors import StageNetTensorProcessor
        >>> legacy = StageNetTensorProcessor()
        >>> legacy.fit([{"x": ([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]])}], "x")
        >>> bridge = StageNetTensorToUnifiedBridge(legacy)
        >>> time, values = legacy.process(([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]]))
        >>> bridge.transform_batch((time.unsqueeze(0), values.unsqueeze(0)))
        (tensor([[1., 2.],
                [3., 4.]]), tensor([0., 1.]), tensor([0, 2]))
    """

    def __init__(self, legacy: StageNetTensorProcessor) -> None:
        if legacy._size is None:
            raise ValueError(
                "StageNetTensorProcessor: fit() saw no values, so the feature "
                "size is unknown and the processor cannot be bridged."
            )
        super().__init__(legacy, [{"ordering": "series", "values": [legacy._size]}])

    def transform_batch(
        self, value: tuple[torch.Tensor | None, torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        time, values = value
        if values.dim() == 2:  # flat numerics: one feature per step
            values = values.unsqueeze(-1)
        padding = _is_zero(values).all(dim=2)
        if time is not None:
            time = _fit_width(time, values.shape[1])
            padding &= time == 0
        rows, batch = _trailing_rows(values, padding)
        if time is None:
            record_time = torch.zeros(rows.shape[0], device=values.device)
        else:
            record_time = _values(time[_prefix(batch.diff(), values.shape[1])])
        return _values(rows), record_time, batch


@_bridges(TimeseriesProcessor)
class TimeseriesToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.TimeseriesProcessor` as a ``series`` of value rows.

    Each grid step is one record with ``[F]`` values and time ``step *
    sampling_rate`` in hours. With ``missing_indicators``, the appended
    indicators become the ``mask``. Each sample's trailing steps whose
    values (and indicators) are all 0 are padding, so real trailing steps
    like that are dropped (see :class:`UnifiedBridge`).

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.processors import TimeseriesProcessor
        >>> t0 = datetime(2020, 1, 1)
        >>> raw = ([t0, t0 + timedelta(hours=2)], [[1.0], [3.0]])
        >>> legacy = TimeseriesProcessor(sampling_rate=timedelta(hours=2))
        >>> legacy.fit([{"x": raw}], "x")
        >>> bridge = TimeseriesToUnifiedBridge(legacy)
        >>> bridge.transform_batch(legacy.process(raw).unsqueeze(0))
        (tensor([[1.],
                [3.]]), tensor([0., 2.]), tensor([0, 2]))
    """

    def __init__(self, legacy: TimeseriesProcessor) -> None:
        if legacy.n_features is None:
            raise ValueError(
                "TimeseriesProcessor: fit() saw no values, so the feature count "
                "is unknown and the processor cannot be bridged."
            )
        level: LevelConfig = {"ordering": "series", "values": [legacy.n_features]}
        if legacy.missing_indicators:
            level["impute"] = {"method": legacy.impute_strategy}
            level["missing_indicator"] = True
        super().__init__(legacy, [level])
        self._hours = legacy.sampling_rate.total_seconds() / 3600.0

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        rows, batch = _trailing_rows(value, _is_zero(value).all(dim=2))
        position = torch.arange(value.shape[1], device=value.device).expand(
            value.shape[:2]
        )
        time = (position[_prefix(batch.diff(), value.shape[1])] * self._hours).to(
            torch.float32
        )
        n = self.legacy.n_features
        if self.legacy.missing_indicators:
            return _values(rows[:, :n]), rows[:, n:] != 0, time, batch
        return _values(rows), time, batch


@_bridges(TemporalTimeseriesProcessor)
class TemporalTimeseriesToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.TemporalTimeseriesProcessor` as a ``series`` of value rows.

    The legacy collate keeps this processor's ``{"value", "time"}`` dicts as
    a list, so the records and times are exact, with no padding to guess.

    Examples:
        >>> from datetime import datetime, timedelta
        >>> from pyhealth.processors import TemporalTimeseriesProcessor
        >>> t0 = datetime(2020, 1, 1)
        >>> raw = ([t0, t0 + timedelta(hours=1)], [[0.0], [2.0]])
        >>> legacy = TemporalTimeseriesProcessor()
        >>> legacy.fit([{"x": raw}], "x")
        >>> bridge = TemporalTimeseriesToUnifiedBridge(legacy)
        >>> bridge.transform_batch([legacy.process(raw)])
        (tensor([[0.],
                [2.]]), tensor([0., 1.]), tensor([0, 2]))
    """

    def __init__(self, legacy: TemporalTimeseriesProcessor) -> None:
        if legacy.n_features is None:
            raise ValueError(
                "TemporalTimeseriesProcessor: fit() saw no values, so the feature "
                "count is unknown and the processor cannot be bridged."
            )
        super().__init__(
            legacy, [{"ordering": "series", "values": [legacy.n_features]}]
        )

    def transform_batch(self, value: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, ...]:
        return _exact_series(
            [item["value"] for item in value], [item["time"] for item in value]
        )


def _exact_series(
    values: list[torch.Tensor], times: list[torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenated records of a list of samples, with their times and offsets."""
    counts = torch.tensor([len(v) for v in values], dtype=torch.long)
    return (
        _values(torch.cat(values)),
        _values(torch.cat(times)),
        _offsets(counts).to(values[0].device),
    )


@_bridges(TimeImageProcessor)
class TimeImageToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.TimeImageProcessor` as a ``series`` of images.

    Each image is one record with ``[C, H, W]`` values, the height and width
    being spatial. The legacy collate keeps this processor's ``(images,
    time, "image")`` tuples as a list, so the records are exact. The
    processor must resize (``image_size``) and output tensors, and its
    channel count must be known from ``fit()``.

    Examples:
        >>> from pyhealth.processors import TimeImageProcessor
        >>> legacy = TimeImageProcessor(image_size=4)
        >>> legacy.n_channels = 1  # what fit() infers from the images
        >>> bridge = TimeImageToUnifiedBridge(legacy)
        >>> bridge.value_dim(), bridge.spatial()
        ({0: (1, 4, 4)}, {0: (False, True, True)})
    """

    def __init__(self, legacy: TimeImageProcessor) -> None:
        if not legacy.to_tensor or legacy.image_size is None:
            raise ValueError(
                "TimeImageProcessor: only a processor with to_tensor=True and an "
                "image_size has a fixed image shape to bridge."
            )
        if legacy.n_channels is None:
            raise ValueError(
                "TimeImageProcessor: the channel count is unknown before fit(), "
                "so the processor cannot be bridged."
            )
        size = legacy.image_size
        super().__init__(
            legacy,
            [
                {
                    "ordering": "series",
                    "values": [legacy.n_channels, size, size],
                    "spatial": [False, True, True],
                }
            ],
        )

    def transform_batch(
        self, value: list[tuple[torch.Tensor, torch.Tensor, str]]
    ) -> tuple[torch.Tensor, ...]:
        return _exact_series([item[0] for item in value], [item[1] for item in value])


# ----------------------------------------------------------------------
# Static processors: one record per sample
# ----------------------------------------------------------------------


def _static(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B, *S]`` -> B records and their batch offsets."""
    return _values(value), _single(value.shape[0], value.device)


def _check_shape(name: str, got: Sequence[int], fitted: Sequence[int]) -> None:
    if tuple(got) != tuple(fitted):
        raise ValueError(
            f"{name}: a batch has shape {tuple(got)} per record, but fit() saw "
            f"{tuple(fitted)}."
        )


@_bridges(TensorProcessor)
class TensorToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.TensorProcessor` as one record, or one per row.

    ``fit()`` records the shape of every sample's tensor. When its first
    axis has the same length in every sample, each sample is one record of
    a ``set`` level, holding the whole tensor (``[1]`` for a scalar). When
    the first axis varies in length, each row along it is one record,
    holding the rest of the tensor (``[1]`` for an element of a vector);
    the level is a ``sequence`` when ``spatial_dims`` marks the first axis
    spatial, and a ``set`` otherwise. Each sample's trailing all-zero (or
    NaN) rows are then padding (see :class:`UnifiedBridge`).

    With ``missing_indicators``, the appended indicators become the
    ``mask``. Integer tensors become float values, and NaN (left by a
    processor without ``impute_strategy``) becomes 0, as the unified format
    holds finite floats. A batch whose records do not have the fitted shape
    (say, a longer fixed axis at evaluation) raises ``ValueError``.

    Not bridged: tensors whose other axes (or number of axes) vary between
    samples, and variable-length vectors with ``missing_indicators``, whose
    indicators are appended along the same axis as the values, so the batch
    padding hides where they start. A processor fitted before the shape was
    recorded is bridged only for scalars and vectors, as one record.

    Examples:
        >>> from pyhealth.processors import TensorProcessor
        >>> legacy = TensorProcessor(spatial_dims=(True, False))
        >>> legacy.fit([{"x": [[1.0, 2.0]]}, {"x": [[3.0, 4.0], [5.0, 6.0]]}], "x")
        >>> bridge = TensorToUnifiedBridge(legacy)
        >>> bridge.ordering(), bridge.value_dim()
        ({0: 'sequence'}, {0: (2,)})
    """

    def __init__(self, legacy: TensorProcessor) -> None:
        if legacy._n_dim is None:
            raise ValueError(
                "TensorProcessor: fit() saw no values, so the shape is unknown "
                "and the processor cannot be bridged."
            )
        shape = legacy._shape
        if shape is None:
            if legacy._n_dim > 1:
                raise ValueError(
                    "TensorProcessor: the shape varies between samples beyond "
                    "the first axis, or fit() did not record it, so the "
                    "processor cannot be bridged."
                )
            shape = (legacy._feature_dim,) if legacy._n_dim == 1 else ()
        spatial = legacy._spatial_dims
        if spatial is not None and len(spatial) != len(shape):
            spatial = None
        self._rows = shape[:1] == (None,)
        if self._rows:
            if legacy.missing_indicators and len(shape) == 1:
                raise ValueError(
                    "TensorProcessor: a vector of varying length with "
                    "missing_indicators cannot be bridged, as the indicators "
                    "follow the values on the same axis."
                )
            record = [int(n) for n in shape[1:]] or [1]
            ordering = "sequence" if spatial and spatial[0] else "set"
            record_spatial = list(spatial[1:]) if spatial and len(shape) > 1 else None
        else:
            record = [int(n) for n in shape] or [1]
            ordering = "set"
            record_spatial = list(spatial) if spatial and shape else None
        level: LevelConfig = {"ordering": ordering, "values": record}
        if record_spatial is not None:
            level["spatial"] = record_spatial
        if legacy.missing_indicators:
            level["impute"] = {"method": "zero"}
            level["missing_indicator"] = True
        super().__init__(legacy, [level])
        self._record = tuple(record)

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        indicators = self.legacy.missing_indicators
        n = self._record[-1]
        fitted = (*self._record[:-1], 2 * n) if indicators else self._record
        if self._rows:
            if len(self._record) == 1 and value.dim() == 2:
                value = value.unsqueeze(-1)  # elements of vectors
            _check_shape("TensorProcessor", value.shape[2:], fitted)
            rows, batch = _trailing_rows(value, _is_zero(value).flatten(2).all(dim=2))
        else:
            if value.dim() == 1:  # a batch of scalars
                value = value.unsqueeze(1)
            _check_shape("TensorProcessor", value.shape[1:], fitted)
            rows, batch = value, _single(value.shape[0], value.device)
        if indicators:
            return _values(rows[..., :n]), rows[..., n:] != 0, batch
        return _values(rows), batch


@_bridges(ImageProcessor)
class ImageToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.ImageProcessor` as a ``set`` of one image.

    Each sample is one record with ``[C, H, W]`` values, the height and
    width being spatial. The shape is the widest that ``fit()`` saw: the
    most channels of ``mode`` (or of the images' own modes, without one),
    and ``image_size`` (or the largest height and width, without one). A
    batch narrower than that is zero-padded at the end of each axis, as the
    legacy collate pads a batch's channels; a wider one (an image unlike any
    seen in ``fit()``) raises ``ValueError``. The processor must output
    tensors. One fitted before the shape was recorded needs ``mode`` and
    ``image_size``.

    Examples:
        >>> import torch
        >>> from pyhealth.processors import ImageProcessor
        >>> legacy = ImageProcessor(image_size=4, mode="RGB")
        >>> legacy.fit([], "x")  # the shape is known from mode and image_size
        >>> bridge = ImageToUnifiedBridge(legacy)
        >>> bridge.value_dim()
        {0: (3, 4, 4)}
        >>> values, batch = bridge.transform_batch(torch.ones(2, 1, 4, 4))
        >>> values.shape, values[:, 1:].abs().sum().item(), batch
        (torch.Size([2, 3, 4, 4]), 0.0, tensor([0, 1, 2]))
    """

    def __init__(self, legacy: ImageProcessor) -> None:
        if not legacy.to_tensor:
            raise ValueError(
                "ImageProcessor: only a processor with to_tensor=True outputs "
                "tensors to bridge."
            )
        shape = legacy._shape
        if shape is None:
            if legacy.mode is None or legacy.image_size is None:
                raise ValueError(
                    "ImageProcessor: fit() has not recorded the image shape, and "
                    "without a mode and an image_size it is unknown, so the "
                    "processor cannot be bridged."
                )
            from PIL import Image

            size = legacy.image_size
            shape = (Image.getmodebands(legacy.mode), size, size)
        super().__init__(
            legacy,
            [
                {
                    "ordering": "set",
                    "values": list(shape),
                    "spatial": [False, True, True],
                }
            ],
        )
        self._shape = tuple(shape)

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        got = value.shape[1:]
        if any(n > fitted for n, fitted in zip(got, self._shape)):
            _check_shape("ImageProcessor", got, self._shape)
        (c, h, w), (fc, fh, fw) = got, self._shape
        value = torch.nn.functional.pad(value, (0, fw - w, 0, fh - h, 0, fc - c))
        return _static(value)


@_bridges(AudioProcessor)
class AudioToUnifiedBridge(UnifiedBridge):
    """:class:`~pyhealth.processors.AudioProcessor` as a ``set`` of one recording.

    Each sample is one record: the ``[1, samples]`` waveform, or the
    ``[1, n_mels, frames]`` mel spectrogram, with the time (and mel) axes
    spatial.

    Limitations:
        :class:`~pyhealth.processors.AudioProcessor` has no ``fit()``, and
        the bridge reads no recording: the shape comes from the
        configuration alone. It is only fixed for a processor with
        ``to_mono``, a ``sample_rate`` and a ``duration``, so others are
        rejected: their recordings may differ in channel count, sample rate
        or length. The mel frame count follows torchaudio's
        ``MelSpectrogram`` defaults (centered frames, hop ``n_fft // 2``).

    Examples:
        >>> from pyhealth.processors import AudioProcessor
        >>> legacy = AudioProcessor(sample_rate=100, duration=2.0)  # doctest: +SKIP
        >>> AudioToUnifiedBridge(legacy).value_dim()  # doctest: +SKIP
        {0: (1, 200)}
    """

    def __init__(self, legacy: AudioProcessor) -> None:
        if not legacy.to_mono or legacy.sample_rate is None or legacy.duration is None:
            raise ValueError(
                "AudioProcessor: only a processor with to_mono=True, a sample_rate "
                "and a duration has a fixed shape to bridge."
            )
        samples = int(legacy.duration * legacy.sample_rate)
        if legacy.n_mels is None:
            shape, spatial = [1, samples], [False, True]
        else:
            # torchaudio's MelSpectrogram: centered frames of hop n_fft // 2.
            hop = legacy.hop_length or legacy.n_fft // 2
            shape, spatial = [1, legacy.n_mels, 1 + samples // hop], [False, True, True]
        super().__init__(
            legacy, [{"ordering": "set", "values": shape, "spatial": spatial}]
        )

    def transform_batch(self, value: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return _static(value)


# ----------------------------------------------------------------------
# Public functions
# ----------------------------------------------------------------------


def _bridge_class(processor: FeatureProcessor) -> type[UnifiedBridge] | None:
    for cls in type(processor).__mro__:
        if cls in _BRIDGES:
            return _BRIDGES[cls]
    return None


def transform_processor(
    processors: Mapping[str, FeatureProcessor],
) -> dict[str, FeatureProcessor]:
    """Replaces every legacy processor with its :class:`UnifiedBridge`.

    A legacy processor is bridged by the bridge of its class, or of its
    nearest base class that has one; that bridge converts. A bridge, which
    only comes in when an outer model already converts, becomes a copy that
    passes batches through (:attr:`UnifiedBridge.converts` is False). Other
    processors are returned as they are.

    Args:
        processors: Fitted processors by field, such as a dataset's
            ``input_processors``.

    Returns:
        A new dict with the same fields.

    Raises:
        TypeError: If a legacy processor has no bridge (``GraphProcessor``
            and ``TupleTimeTextProcessor``).

    Examples:
        >>> from pyhealth.processors import RawProcessor, SequenceProcessor
        >>> codes = SequenceProcessor()
        >>> codes.fit([{"codes": ["A"]}], "codes")
        >>> bridged = transform_processor({"codes": codes, "note": RawProcessor()})
        >>> type(bridged["codes"]).__name__, type(bridged["note"]).__name__
        ('SequenceToUnifiedBridge', 'RawProcessor')
        >>> bridged["codes"].converts, transform_processor(bridged)["codes"].converts
        (True, False)
    """
    bridged: dict[str, FeatureProcessor] = {}
    for field, processor in processors.items():
        if isinstance(processor, UnifiedBridge):
            bridged[field] = processor._passing_through()
            continue
        if not isinstance(processor, LegacyFeatureProcessor):
            bridged[field] = processor
            continue
        bridge_cls = _bridge_class(processor)
        if bridge_cls is None:
            raise TypeError(
                f"'{field}': {type(processor).__name__} has no unified bridge yet."
            )
        bridged[field] = bridge_cls(processor)
    return bridged


def transform_batch(
    processors: Mapping[str, FeatureProcessor], batch: Mapping[str, Any]
) -> dict[str, Any]:
    """Converts the fields of a legacy batch whose bridge converts.

    A field whose processor is a converting :class:`UnifiedBridge` is
    converted; every other field (a passing-through bridge, a unified, label
    or other processor, or no processor) is returned as it is.

    Args:
        processors: The output of :func:`transform_processor`.
        batch: A batch of :func:`~pyhealth.datasets.get_dataloader`.

    Returns:
        A new dict: each bridged field is its :meth:`UnifiedBridge.transform_batch`
        tuple, and every other key is as it was.

    Examples:
        >>> from pyhealth.datasets import collate_fn_dict_with_padding
        >>> from pyhealth.processors import SequenceProcessor
        >>> codes = SequenceProcessor()
        >>> codes.fit([{"codes": ["A", "B"]}], "codes")
        >>> processors = transform_processor({"codes": codes})
        >>> batch = collate_fn_dict_with_padding(
        ...     [{"codes": codes.process(["A", "B"]), "id": "p1"}]
        ... )
        >>> transform_batch(processors, batch)
        {'codes': (tensor([2, 3]), tensor([0, 2])), 'id': ['p1']}
    """
    converting = {
        key
        for key, processor in processors.items()
        if isinstance(processor, UnifiedBridge) and processor.converts
    }
    return {
        key: processors[key].transform_batch(value) if key in converting else value
        for key, value in batch.items()
    }


class BridgeDataset:
    """A dataset whose legacy input processors are replaced by unified bridges.

    For model constructors only: a V2 model wraps the dataset it is given,
    so it reads each legacy field as the levels its bridge describes, and
    converts each batch with :func:`transform_batch` on its own
    ``input_processors``. The input processors are
    :func:`transform_processor` of the wrapped dataset's: a legacy processor
    becomes a converting bridge, and a bridge (when the wrapped dataset is
    itself a ``BridgeDataset``, for a model nested in another) becomes one
    that passes batches through, so only the outermost model converts.
    Everything else (``output_processors``, ``input_schema``,
    ``output_schema``, ``len()`` and any other attribute) is the wrapped
    dataset's, which is not changed. Data loaders are built from the
    original dataset.

    Args:
        dataset: The dataset to wrap.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> samples = [{"patient_id": f"p{i}", "codes": ["A"], "y": i} for i in range(2)]
        >>> dataset = create_sample_dataset(samples, {"codes": "sequence"}, {"y": "binary"})
        >>> bridged = BridgeDataset(dataset)
        >>> type(bridged.input_processors["codes"]).__name__, len(bridged)
        ('SequenceToUnifiedBridge', 2)
        >>> bridged.output_processors is dataset.output_processors
        True
        >>> BridgeDataset(bridged).input_processors["codes"].converts
        False
    """

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self.input_processors = transform_processor(dataset.input_processors)

    @property
    def dataset(self) -> Any:
        """The wrapped dataset."""
        return self._dataset

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on the wrapper itself.
        if name == "_dataset":
            raise AttributeError(name)
        return getattr(self._dataset, name)

    def __repr__(self) -> str:
        return f"BridgeDataset({self._dataset!r})"
