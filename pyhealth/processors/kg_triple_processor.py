import hashlib
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch

from . import register_processor
from .base_processor import FeatureProcessor


class _FittedDicts:
    """The dicts a ``KGTripleProcessor`` learns in ``fit``.

    ``BaseDataset.set_task`` puts ``vars()`` of pre-fitted processors into a
    JSON cache key. Tuple-keyed dicts cannot be written as JSON, and their
    full content would not belong in a key anyway, so they live here and the
    key gets ``str()`` of this object: a digest of the content, the same on
    every run and different whenever the fitted dicts differ.
    """

    def __init__(
        self,
        true_head: dict[tuple[int, int], list[int]],
        true_tail: dict[tuple[int, int], list[int]],
        count: dict[tuple[int, int], int],
    ):
        self.true_head = true_head
        self.true_tail = true_tail
        self.count = count
        content = repr(
            (sorted(true_head.items()), sorted(true_tail.items()), sorted(count.items()))
        )
        self.digest = hashlib.sha256(content.encode()).hexdigest()

    def __str__(self) -> str:
        return f"fitted-dicts-sha256:{self.digest}"


@register_processor("kg_triple")
class KGTripleProcessor(FeatureProcessor):
    """Encodes a knowledge-graph triple and keeps training-graph dicts.

    ``process`` turns a triple ``(head, relation, tail)`` of integer ids into
    a ``LongTensor`` of shape ``(3,)``. ``fit`` reads the triples it is given
    and fills three plain dicts, built as in the reference implementation of
    Sun et al. (2019, RotatE), whose training dataset computes them from the
    training triples only:

    - ``true_head[(relation, tail)]``: the heads seen with that pair;
    - ``true_tail[(head, relation)]``: the tails seen with that pair;
    - ``count``: word2vec-style frequencies of ``(head, relation)`` and
      ``(tail, -relation - 1)``, starting at ``count_start``, from which
      :meth:`subsampling_weight` derives the weight of a training triple.

    The models use ``true_head`` / ``true_tail`` to keep known positives out
    of the negatives they draw for training. Evaluation does not use them:
    filtered ranking uses the ground-truth sets of the whole graph, carried
    by the samples.

    These dicts must describe the training triples only, otherwise the
    negatives and weights of training depend on validation and test
    triples. ``set_task(task, split=PatientSplit(...))`` fits processors on
    the training part only; without ``split``, PyHealth warns, because
    ``learns_statistics`` is set.

    Entity and relation ids are global (fixed on the full graph), so
    ``num_entities`` and ``num_relations`` are given, not fitted: an entity
    absent from the training triples keeps its id. ``size()`` returns
    ``num_entities``.

    Args:
        num_entities: Number of entities; ids are ``0..num_entities - 1``.
        num_relations: Number of relations; ids are
            ``0..num_relations - 1``.
        count_start: Value of a frequency the first time a pair is seen.
            Default is 4, as in Sun et al. (2019).

    Raises:
        ValueError: If a count is not a positive integer, or if ``fit``
            meets a malformed triple or an id out of range.

    Examples:
        >>> processor = KGTripleProcessor(num_entities=3, num_relations=1)
        >>> processor.fit([{"triple": (0, 0, 1)}, {"triple": (0, 0, 2)}], "triple")
        >>> processor.process((0, 0, 1))
        tensor([0, 0, 1])
        >>> processor.true_tail[(0, 0)]
        [1, 2]
        >>> processor.true_head[(0, 2)]
        [0]
        >>> processor.subsampling_weight(torch.tensor([[0, 0, 1]]))
        tensor([0.3333])
    """

    learns_statistics = True

    def __init__(self, num_entities: int, num_relations: int, count_start: int = 4):
        for name, value in (
            ("num_entities", num_entities),
            ("num_relations", num_relations),
            ("count_start", count_start),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.count_start = count_start
        self._fitted = _FittedDicts({}, {}, {})

    @property
    def true_head(self) -> dict[tuple[int, int], list[int]]:
        """``(relation, tail)`` to the sorted heads seen with it in ``fit``."""
        return self._fitted.true_head

    @property
    def true_tail(self) -> dict[tuple[int, int], list[int]]:
        """``(head, relation)`` to the sorted tails seen with it in ``fit``."""
        return self._fitted.true_tail

    @property
    def count(self) -> dict[tuple[int, int], int]:
        """Frequencies of ``(head, relation)`` and ``(tail, -relation - 1)``."""
        return self._fitted.count

    def _check(self, value: Any) -> tuple[int, int, int]:
        ids = [int(x) for x in value]
        if len(ids) != 3:
            raise ValueError(f"A triple has 3 ids, got {tuple(value)}.")
        head, relation, tail = ids
        if not (0 <= head < self.num_entities and 0 <= tail < self.num_entities):
            raise ValueError(
                f"Triple {tuple(value)} has an entity id outside "
                f"[0, {self.num_entities})."
            )
        if not 0 <= relation < self.num_relations:
            raise ValueError(
                f"Triple {tuple(value)} has a relation id outside "
                f"[0, {self.num_relations})."
            )
        return head, relation, tail

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
        """Builds ``true_head``, ``true_tail`` and ``count`` from ``samples``.

        Refitting replaces the dicts; it does not add to them.

        Args:
            samples: Samples whose ``field`` holds a triple of ids. With
                ``set_task(..., split=...)`` these are the training samples.
            field: Name of the triple field.
        """
        true_head: dict[tuple[int, int], set[int]] = defaultdict(set)
        true_tail: dict[tuple[int, int], set[int]] = defaultdict(set)
        count: dict[tuple[int, int], int] = {}
        for sample in samples:
            head, relation, tail = self._check(sample[field])
            true_head[(relation, tail)].add(head)
            true_tail[(head, relation)].add(tail)
            # The inverse key (tail, -relation - 1) keeps head-side and
            # tail-side frequencies apart in one dict, as in the reference.
            for key in ((head, relation), (tail, -relation - 1)):
                count[key] = count[key] + 1 if key in count else self.count_start
        # Sorted lists make the fitted state independent of sample order.
        self._fitted = _FittedDicts(
            {k: sorted(v) for k, v in true_head.items()},
            {k: sorted(v) for k, v in true_tail.items()},
            count,
        )

    def process(self, value: Any) -> torch.Tensor:
        """Returns the triple as a ``LongTensor`` of shape ``(3,)``.

        Args:
            value: A triple ``(head, relation, tail)`` of integer ids.
        """
        triple = torch.as_tensor(value, dtype=torch.long)
        if triple.shape != (3,):
            raise ValueError(f"A triple has 3 ids, got shape {tuple(triple.shape)}.")
        return triple

    def subsampling_weight(self, triples: torch.Tensor) -> torch.Tensor:
        """Weights of training triples, ``1 / sqrt(count(h, r) + count(t, -r-1))``.

        Args:
            triples: ``LongTensor`` of shape ``(batch, 3)``.

        Returns:
            Float tensor of shape ``(batch,)``.

        Raises:
            KeyError: If ``(head, relation)`` or ``(tail, -relation - 1)`` was
                not seen in ``fit``. Only that is checked: a triple absent
                from ``fit`` whose two pairs were each seen in other triples
                gets a weight. Pass training triples only.
        """
        totals = []
        for head, relation, tail in triples.tolist():
            try:
                totals.append(
                    self.count[(head, relation)] + self.count[(tail, -relation - 1)]
                )
            except KeyError:
                raise KeyError(
                    f"Triple {(head, relation, tail)} has a pair not seen in "
                    "fit; subsampling weights exist for training triples only."
                ) from None
        return torch.sqrt(1.0 / torch.tensor(totals, dtype=torch.float32))

    def size(self) -> int:
        """Number of entities."""
        return self.num_entities

    def is_token(self) -> bool:
        """Entity and relation ids are discrete indices."""
        return True

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (1,)

    def spatial(self) -> tuple[bool, ...]:
        return (False,)

    def __repr__(self) -> str:
        return (
            f"KGTripleProcessor(num_entities={self.num_entities}, "
            f"num_relations={self.num_relations}, "
            f"fitted_pairs={len(self.true_tail)})"
        )
