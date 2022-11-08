import logging
import os
from collections import defaultdict
from urllib.error import HTTPError

import pyhealth.medcode as medcode
from pyhealth.medcode.utils import MODULE_CACHE_PATH, download_and_read_csv
from pyhealth.utils import load_pickle, save_pickle


class CrossMap:
    """Contains mapping between two medical code systems.

    `CrossMap` is a base class for all possible mappings. It will be
    initialized with two specific medical code systems with
    `CrossMap.load(source_vocabulary, target_vocabulary)`.
    """

    def __init__(
            self,
            source_vocabulary: str,
            target_vocabulary: str,
            refresh_cache: bool = False,
            **kwargs,
    ):
        self.s_vocab = source_vocabulary
        self.t_vocab = target_vocabulary
        self.kwargs = kwargs

        # load mapping
        pickle_filename = f"{self.s_vocab}_to_{self.t_vocab}.pkl"
        pickle_filepath = os.path.join(MODULE_CACHE_PATH, pickle_filename)
        if os.path.exists(pickle_filepath) and (not refresh_cache):
            logging.info(f"Loaded {self.s_vocab}->{self.t_vocab} mapping "
                         f"from {pickle_filepath}")
            self.mapping = load_pickle(pickle_filepath)
        else:
            logging.info(f"Processing {self.s_vocab}->{self.t_vocab} mapping...")
            try:
                local_filename = f"{self.s_vocab}_to_{self.t_vocab}.csv"
                df = download_and_read_csv(local_filename, refresh_cache)
            except HTTPError:
                local_filename = f"{self.t_vocab}_to_{self.s_vocab}.csv"
                df = download_and_read_csv(local_filename, refresh_cache)
            self.mapping = defaultdict(list)
            for _, row in df.iterrows():
                self.mapping[row[self.s_vocab]].append(row[self.t_vocab])
            logging.info(f"Saved {self.s_vocab}->{self.t_vocab} mapping "
                         f"to {pickle_filepath}")
            save_pickle(self.mapping, pickle_filepath)

        # load source and target vocabulary classes
        self.s_class = getattr(medcode, source_vocabulary)()
        self.t_class = getattr(medcode, target_vocabulary)()
        return

    @classmethod
    def load(
            cls,
            source_vocabulary: str,
            target_vocabulary: str,
            refresh_cache: bool = False,
            **kwargs
    ):
        """Initializes the mapping between two medical code systems.

        Args:
            source_vocabulary: source medical code system.
            target_vocabulary: target medical code system.
            refresh_cache: whether to refresh the cache. Default is False.
            **kwargs: additional arguments for the target medical code system.
                Will be passed to `self.tgt_class.convert()`.

        Examples:
            >>> from pyhealth.medcode import CrossMap
            >>> mapping = CrossMap("ICD9CM", "CCSCM")
            >>> mapping.map("428.0")
            ['108']

            >>> mapping = CrossMap.load("NDC", "ATC", level=3)
            >>> mapping.map("00527051210")
            ['A11']
        """
        return cls(source_vocabulary, target_vocabulary, refresh_cache, **kwargs)

    def map(self, source_code):
        source_code = self.s_class.standardize(source_code)
        tgt_codes = self.mapping[source_code]
        tgt_codes = [self.t_class.convert(c, **self.kwargs) for c in tgt_codes]
        return tgt_codes
