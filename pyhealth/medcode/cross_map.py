import os
from collections import defaultdict
from urllib.error import HTTPError

import pyhealth.medcode as medcode
from pyhealth import BASE_CACHE_PATH
from pyhealth.medcode.utils import download_and_read_csv
from pyhealth.utils import create_directory, load_pickle, save_pickle

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")
create_directory(MODULE_CACHE_PATH)


# TODO: add comments

class CrossMap:
    def __init__(
            self,
            src_vocab,
            tgt_vocab,
            refresh_cache: bool = False,
            **kwargs,
    ):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.kwargs = kwargs

        self.mapping = self.load_mapping(refresh_cache)

        self.src_class = getattr(medcode, src_vocab)()
        self.tgt_class = getattr(medcode, tgt_vocab)()

        return

    def load_mapping(self, refresh_cache: bool = False):
        pickle_filename = f"{self.src_vocab}_to_{self.tgt_vocab}.pkl"
        pickle_filepath = os.path.join(MODULE_CACHE_PATH, pickle_filename)

        if os.path.exists(pickle_filepath) and (not refresh_cache):
            print(f"Loaded {self.src_vocab}->{self.tgt_vocab} mapping "
                  f"from {pickle_filepath}")
            mapping = load_pickle(pickle_filepath)
        else:
            print(f"Processing {self.src_vocab}->{self.tgt_vocab} mapping...")
            try:
                local_filename = f"{self.src_vocab}_to_{self.tgt_vocab}.csv"
                df = download_and_read_csv(local_filename, refresh_cache)
            except HTTPError:
                local_filename = f"{self.tgt_vocab}_to_{self.src_vocab}.csv"
                df = download_and_read_csv(local_filename, refresh_cache)
            mapping = defaultdict(list)
            for _, row in df.iterrows():
                mapping[row[self.src_vocab]].append(row[self.tgt_vocab])
            print(f"Saved {self.src_vocab}->{self.tgt_vocab} mapping "
                  f"to {pickle_filepath}")
            save_pickle(mapping, pickle_filepath)

        return mapping

    def map(self, src_code):
        src_code = self.src_class.standardize(src_code)
        tgt_codes = self.mapping[src_code]
        tgt_codes = [self.tgt_class.postprocess(c, **self.kwargs) for c in tgt_codes]
        return tgt_codes


if __name__ == "__main__":
    codemap = CrossMap("ICD9CM", "CCSCM")
    print(codemap.map("82101"))

    codemap = CrossMap("NDC", "ATC", level=3)
    print(codemap.map("00527051210"))
