import logging
import os
from collections import defaultdict
from typing import List, Optional, Dict
from urllib.error import HTTPError

import pyhealth.medcode as medcode
from pyhealth.medcode.icd_mappings import (
    ALL_ICD_MAPPINGS_PAIRS,
    ICD_MAPPINGS_PAIRS,
    LOSSY_PAIRS,
    PYHEALTH_NATIVE_PAIRS,
    _ICDMappingsBackend,
)
from pyhealth.medcode.utils import MODULE_CACHE_PATH, download_and_read_csv
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)

#: Mapping tables PyHealth hosts itself.
BACKEND_PYHEALTH = "pyhealth"
#: Mapping tables supplied by the ``icd-mappings`` package.
BACKEND_ICDMAPPINGS = "icdmappings"
#: Prefer PyHealth's own tables, fall back to ``icd-mappings``.
BACKEND_AUTO = "auto"


class CrossMap:
    """Contains mapping between two medical code systems.

    `CrossMap` is a base class for all possible mappings. It will be
    initialized with two specific medical code systems with
    `CrossMap.load(source_vocabulary, target_vocabulary)`.

    Mappings PyHealth hosts itself are always preferred. Pairs PyHealth has no
    table for -- ICD-9 <-> ICD-10 and the grouper vocabularies -- are served by
    the `icd-mappings` package instead. `CrossMap.backend` records which source
    was used.

    Examples:
        >>> from pyhealth.medcode import CrossMap
        >>> # PyHealth's own table, selected automatically
        >>> mapping = CrossMap.load("ICD9CM", "CCSCM")
        >>> mapping.backend
        'pyhealth'
        >>> mapping.map("428.0")
        ['108']
        >>> # translating between ICD versions
        >>> CrossMap.load("ICD9CM", "ICD10CM").map("250.00")
        ['E11.9']
        >>> CrossMap.load("ICD10CM", "ICD9CM").map("N17.9")
        ['584.9']
        >>> # grouping into a coarser vocabulary
        >>> CrossMap.load("ICD10CM", "CCSR").map("J18.9")
        ['RSP002']
        >>> CrossMap.load("ICD10CM", "ICD10CHAPTER").map("E11.9")
        ['E00-E89']
        >>> # translation is not symmetric: mapping back lands elsewhere
        >>> CrossMap.load("ICD9CM", "ICD10CM").map("038.9")
        ['A41.9']
        >>> CrossMap.load("ICD10CM", "ICD9CM").map("A41.9")
        ['995.91']
    """

    def __init__(
        self,
        source_vocabulary: str,
        target_vocabulary: str,
        refresh_cache: bool = False,
        backend: str = BACKEND_AUTO,
    ):
        self.s_vocab = source_vocabulary
        self.t_vocab = target_vocabulary
        self.backend = self._resolve_backend(
            source_vocabulary, target_vocabulary, backend
        )

        if self.backend == BACKEND_ICDMAPPINGS:
            if (self.s_vocab, self.t_vocab) in LOSSY_PAIRS:
                logger.warning(
                    "%s->%s collapses a many-to-many relation to a single "
                    "primary target, so some codes have no mapping and the "
                    "reverse mapping is not its inverse. Inspect "
                    "CrossMap.unmapped_codes after mapping a dataset.",
                    self.s_vocab,
                    self.t_vocab,
                )
            self.mapping = _ICDMappingsBackend(
                self.s_vocab,
                self.t_vocab,
                standardize_target=getattr(medcode, self.t_vocab).standardize,
            )
            # Bind the vocabulary *classes*, not instances. map() only calls
            # their standardize()/convert() staticmethods, and instantiating
            # an InnerMap would download an ontology file -- which would
            # defeat the point of a backend whose data ships offline.
            self._s_class = getattr(medcode, self.s_vocab)
            self._t_class = getattr(medcode, self.t_vocab)
            return

        # load mapping
        pickle_filename = f"{self.s_vocab}_to_{self.t_vocab}.pkl"
        pickle_filepath = os.path.join(MODULE_CACHE_PATH, pickle_filename)
        if os.path.exists(pickle_filepath) and (not refresh_cache):
            logger.debug(
                f"Loaded {self.s_vocab}->{self.t_vocab} mapping "
                f"from {pickle_filepath}"
            )
            self.mapping = load_pickle(pickle_filepath)
        else:
            logger.debug(f"Processing {self.s_vocab}->{self.t_vocab} mapping...")
            try:
                local_filename = f"{self.s_vocab}_to_{self.t_vocab}.csv"
                df = download_and_read_csv(local_filename, refresh_cache)
            except HTTPError:
                local_filename = f"{self.t_vocab}_to_{self.s_vocab}.csv"
                df = download_and_read_csv(local_filename, refresh_cache)
            self.mapping = defaultdict(list)
            for _, row in df.iterrows():
                self.mapping[row[self.s_vocab]].append(row[self.t_vocab])
            logger.debug(
                f"Saved {self.s_vocab}->{self.t_vocab} mapping " f"to {pickle_filepath}"
            )
            save_pickle(self.mapping, pickle_filepath)

        # Vocabulary classes are resolved lazily: map() needs only their
        # standardize()/convert() staticmethods, while instantiating one
        # downloads an ontology file we would never read.
        self._s_class = None
        self._t_class = None
        return

    @staticmethod
    def _resolve_backend(
        source_vocabulary: str, target_vocabulary: str, backend: str
    ) -> str:
        """Chooses the data source for a vocabulary pair."""
        pair = (source_vocabulary, target_vocabulary)
        if backend == BACKEND_AUTO:
            if pair in PYHEALTH_NATIVE_PAIRS:
                return BACKEND_PYHEALTH
            if pair in ICD_MAPPINGS_PAIRS:
                return BACKEND_ICDMAPPINGS
            raise ValueError(
                f"No mapping available for {source_vocabulary}->"
                f"{target_vocabulary}. PyHealth serves "
                f"{sorted(PYHEALTH_NATIVE_PAIRS)} and icd-mappings serves "
                f"{sorted(ICD_MAPPINGS_PAIRS)}."
            )
        if backend == BACKEND_ICDMAPPINGS:
            if pair not in ALL_ICD_MAPPINGS_PAIRS:
                raise ValueError(
                    f"icd-mappings cannot serve {source_vocabulary}->"
                    f"{target_vocabulary}. It serves "
                    f"{sorted(ALL_ICD_MAPPINGS_PAIRS)}."
                )
            return BACKEND_ICDMAPPINGS
        if backend == BACKEND_PYHEALTH:
            return BACKEND_PYHEALTH
        raise ValueError(
            f"Unknown backend {backend!r}. Expected one of "
            f"{[BACKEND_AUTO, BACKEND_PYHEALTH, BACKEND_ICDMAPPINGS]}."
        )

    @property
    def s_class(self):
        """The source vocabulary instance, resolved on first use."""
        if self._s_class is None:
            self._s_class = getattr(medcode, self.s_vocab)()
        return self._s_class

    @property
    def t_class(self):
        """The target vocabulary instance, resolved on first use."""
        if self._t_class is None:
            self._t_class = getattr(medcode, self.t_vocab)()
        return self._t_class

    @property
    def unmapped_codes(self):
        """Source codes seen so far that had no target in this mapping."""
        return getattr(self.mapping, "unmapped_codes", frozenset())

    def __repr__(self):
        return f"CrossMap(source_vocabulary={self.s_vocab}, source_class={self.s_class} target_vocabulary={self.t_vocab}, target_class={self.t_class})"

    @classmethod
    def load(
        cls,
        source_vocabulary: str,
        target_vocabulary: str,
        refresh_cache: bool = False,
        backend: str = BACKEND_AUTO,
    ):
        """Initializes the mapping between two medical code systems.

        Args:
            source_vocabulary: source medical code system.
            target_vocabulary: target medical code system.
            refresh_cache: whether to refresh the cache. Default is False.
            backend: which data source to use. "auto" (the default) prefers
                PyHealth's own mapping tables and falls back to the
                ``icd-mappings`` package for pairs PyHealth has no table for.
                "pyhealth" and "icdmappings" force one or the other.

        Examples:
            >>> from pyhealth.medcode import CrossMap
            >>> mapping = CrossMap("ICD9CM", "CCSCM")
            >>> mapping.map("428.0")
            ['108']

            >>> mapping = CrossMap.load("NDC", "ATC")
            >>> mapping.map("00527051210", target_kwargs={"level": 3})
            ['A11C']

            >>> mapping = CrossMap.load("ICD10CM", "CCSR")
            >>> mapping.map("I50.9")
            ['CIR019']

            Forcing a backend. ICD9CM->CCSCM is servable by both sources; it
            stays on PyHealth's table by default, and the same mapping can be
            had offline on request:

            >>> CrossMap.load("ICD9CM", "CCSCM", backend="icdmappings").backend
            'icdmappings'
        """
        return cls(
            source_vocabulary, target_vocabulary, refresh_cache, backend
        )

    def map(
        self,
        source_code: str,
        source_kwargs: Optional[Dict] = None,
        target_kwargs: Optional[Dict] = None,
    ) -> List[str]:
        """Maps a source code to a list of target codes.

        Args:
            source_code: source code.
            **source_kwargs: additional arguments for the source code. Will be
                passed to `self.s_class.convert()`. Default is empty dict.
            **target_kwargs: additional arguments for the target code. Will be
                passed to `self.t_class.convert()`. Default is empty dict.

        Returns:
            A list of target codes.
        """
        if source_kwargs is None:
            source_kwargs = {}
        if target_kwargs is None:
            target_kwargs = {}
        source_code = self.s_class.standardize(source_code)
        source_code = self.s_class.convert(source_code, **source_kwargs)
        target_codes = self.mapping[source_code]
        target_codes = [self.t_class.convert(c, **target_kwargs) for c in target_codes]
        return target_codes