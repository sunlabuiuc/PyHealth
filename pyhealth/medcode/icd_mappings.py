"""Mapping backend built on the ``icd-mappings`` package.

PyHealth hosts its own mapping tables on a resource server, and those remain
the preferred source: they are the ones the library has always shipped, and
they are what existing pipelines were validated against. This module supplies
the mappings PyHealth has no table for -- ICD-9 <-> ICD-10 translation and the
grouper vocabularies -- from the ``icd-mappings`` package, whose data is
bundled in its wheel and therefore needs no network access.

Author: John Wu (johnwu3)
"""

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

#: Vocabulary pairs PyHealth serves from its own resource server. Listed in
#: both orderings because ``CrossMap`` derives the reverse direction from the
#: same CSV via its reverse-filename fallback. These always win over the
#: ``icd-mappings`` backend under ``backend="auto"``.
PYHEALTH_NATIVE_PAIRS: frozenset = frozenset(
    {
        ("ICD9CM", "CCSCM"),
        ("CCSCM", "ICD9CM"),
        ("ICD10CM", "CCSCM"),
        ("CCSCM", "ICD10CM"),
        ("ICD9PROC", "CCSPROC"),
        ("CCSPROC", "ICD9PROC"),
        ("ICD10PROC", "CCSPROC"),
        ("CCSPROC", "ICD10PROC"),
        ("NDC", "ATC"),
        ("ATC", "NDC"),
        ("NDC", "RxNorm"),
        ("RxNorm", "NDC"),
    }
)

#: Vocabulary pairs served by ``icd-mappings``, mapped to the ``(source,
#: target)`` argument pair that package expects.
ICD_MAPPINGS_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("ICD9CM", "ICD10CM"): ("icd9", "icd10"),
    ("ICD10CM", "ICD9CM"): ("icd10", "icd9"),
    ("ICD10CM", "CCSR"): ("icd10", "ccsr"),
    ("ICD9CM", "CCI"): ("icd9", "cci"),
    ("ICD10CM", "CCIR"): ("icd10", "ccir"),
    ("ICD9CM", "ICD9CHAPTER"): ("icd9", "chapter"),
    ("ICD10CM", "ICD10CHAPTER"): ("icd10", "chapter"),
    ("ICD10CM", "ICD10BLOCK"): ("icd10", "block"),
    ("ICD9CM", "CCC"): ("icd9", "ccc_category"),
    ("ICD10CM", "CCC"): ("icd10", "ccc_category"),
    ("ICD9CM", "CCCSUB"): ("icd9", "ccc_subcategory"),
    ("ICD10CM", "CCCSUB"): ("icd10", "ccc_subcategory"),
}

#: Pairs ``icd-mappings`` can also serve, but which PyHealth serves from its
#: own table by default. Reachable only by asking for
#: ``backend="icdmappings"`` explicitly -- useful when an offline CCS mapping
#: is wanted. Never selected by ``backend="auto"``, so the default behavior of
#: long-standing pipelines cannot change.
ICD_MAPPINGS_OPTIONAL_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    ("ICD9CM", "CCSCM"): ("icd9", "ccs"),
}

#: Every pair the backend can serve, however it was selected.
ALL_ICD_MAPPINGS_PAIRS: dict[tuple[str, str], tuple[str, str]] = {
    **ICD_MAPPINGS_PAIRS,
    **ICD_MAPPINGS_OPTIONAL_PAIRS,
}

#: Pairs where the underlying relation is many-to-many but ``icd-mappings``
#: returns a single primary target. Mapping through these loses information
#: and does not round-trip.
LOSSY_PAIRS: frozenset = frozenset(
    {
        ("ICD9CM", "ICD10CM"),
        ("ICD10CM", "ICD9CM"),
    }
)


def available_icd_mapping_pairs() -> list[tuple[str, str]]:
    """Lists the vocabulary pairs served by the ``icd-mappings`` backend.

    Returns:
        Sorted ``(source_vocabulary, target_vocabulary)`` pairs.

    Examples:
        >>> from pyhealth.medcode.icd_mappings import (
        ...     available_icd_mapping_pairs,
        ... )
        >>> ("ICD9CM", "ICD10CM") in available_icd_mapping_pairs()
        True
    """
    return sorted(ICD_MAPPINGS_PAIRS)


@lru_cache(maxsize=1)
def _get_mapper():
    """Builds the upstream ``Mapper`` once per process (it loads ~30 MB)."""
    try:
        from icdmappings import Mapper
    except ImportError as exc:  # pragma: no cover - icd-mappings is required
        raise ImportError(
            "icd-mappings is required for ICD translation and grouper "
            "vocabularies. Install it with `pip install icd-mappings`."
        ) from exc
    return Mapper()


class _ICDMappingsBackend:
    """Read-only ``{source_code: [target_code]}`` view over ``icd-mappings``.

    Duck-types the ``dict`` that ``CrossMap`` builds from a CSV, so that
    ``CrossMap.map()`` needs no changes: it still just subscripts
    ``self.mapping``.
    """

    def __init__(
        self,
        source_vocabulary: str,
        target_vocabulary: str,
        standardize_target: Callable[[Any], str],
    ) -> None:
        self.s_vocab = source_vocabulary
        self.t_vocab = target_vocabulary
        self._src, self._tgt = ALL_ICD_MAPPINGS_PAIRS[
            (source_vocabulary, target_vocabulary)
        ]
        self._standardize_target = standardize_target
        #: Source codes seen by this backend that had no target. Inspect it
        #: after a pass over a dataset to quantify mapping loss.
        self.unmapped_codes: set[str] = set()

    def __getitem__(self, code: str) -> list[str]:
        # icd-mappings accepts dotted and undotted input, but PyHealth's
        # canonical form is dotted; strip so both spellings hit one cache key.
        raw = _get_mapper().map(
            str(code).replace(".", ""), source=self._src, target=self._tgt
        )
        if raw is None:
            self.unmapped_codes.add(code)
            return []
        # The target vocabulary owns its own normalization: re-dotting for ICD
        # targets, bool -> "1"/"0" for the indicators, label stripping for
        # chapters and blocks.
        return [self._standardize_target(raw)]

    def __contains__(self, code: str) -> bool:
        return bool(self[code])

    def __repr__(self) -> str:
        return (
            f"_ICDMappingsBackend(source_vocabulary={self.s_vocab}, "
            f"target_vocabulary={self.t_vocab})"
        )
