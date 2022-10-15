import os
from abc import ABC
from collections import defaultdict
from typing import Optional, List
from urllib.error import HTTPError
from urllib.parse import urljoin

import networkx as nx
import pandas as pd

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory, download, load_pickle, save_pickle

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")
create_directory(MODULE_CACHE_PATH)


class BaseCode(ABC):
    """Abstract base coding system."""

    def __init__(
        self,
        vocabulary: str,
        valid_mappings: Optional[List[str]] = None,
        refresh_cache: bool = False,
    ):
        if valid_mappings is None:
            valid_mappings = []
        self.vocabulary = vocabulary
        self.valid_mappings = valid_mappings
        self.refresh_cache = refresh_cache
        self.cached_mappings = {}

        pickle_filepath = os.path.join(MODULE_CACHE_PATH, self.vocabulary + ".pkl")
        csv_filename = self.vocabulary + ".csv"
        if os.path.exists(pickle_filepath) and not refresh_cache:
            print(f"Loaded {vocabulary} code from {pickle_filepath}")
            self.graph = load_pickle(pickle_filepath)
        else:
            print(f"Processing {vocabulary} code...")
            df = self.download_and_read_csv(csv_filename, refresh_cache)
            self.graph = self.build_graph(df)
            print(f"Saved {vocabulary} code to {pickle_filepath}")
            save_pickle(self.graph, pickle_filepath)

    @staticmethod
    def download_and_read_csv(filename: str, refresh_cache: bool = False):
        if (
            not os.path.exists(os.path.join(MODULE_CACHE_PATH, filename))
        ) or refresh_cache:
            download(
                urljoin(BASE_URL, filename), os.path.join(MODULE_CACHE_PATH, filename)
            )
        return pd.read_csv(os.path.join(MODULE_CACHE_PATH, filename), dtype=str)

    @staticmethod
    def build_graph(df):
        df = df.set_index("code")
        graph = nx.DiGraph()
        for code, row in df.iterrows():
            graph.add_node(code, **row)
        for code, row in df.iterrows():
            if "parent_code" in row:
                if not pd.isna(row["parent_code"]):
                    graph.add_edge(row["parent_code"], code)
        return graph

    def lookup(self, code):
        return self.graph.nodes[code]

    def contains(self, code):
        return code in self.graph.nodes

    def get_ancestors(self, code):
        # ordered ancestors
        ancestors = nx.ancestors(self.graph, code)
        ancestors = list(ancestors)
        ancestors = sorted(
            ancestors, key=lambda x: nx.shortest_path_length(self.graph, x, code)
        )
        return ancestors

    def load_mapping(self, target_vocabulary):
        if target_vocabulary not in self.valid_mappings:
            raise ValueError(
                f"Cannot map from {self.vocabulary} to {target_vocabulary}"
            )

        pickle_filepath = os.path.join(
            MODULE_CACHE_PATH, self.vocabulary + "_to_" + target_vocabulary + ".pkl"
        )
        if os.path.exists(pickle_filepath) and not self.refresh_cache:
            print(
                f"Loaded {self.vocabulary}->{target_vocabulary} mapping from {pickle_filepath}"
            )
            mapping = load_pickle(pickle_filepath)
        else:
            print(f"Processing {self.vocabulary}->{target_vocabulary} mapping...")
            try:
                csv_filename = self.vocabulary + "_to_" + target_vocabulary + ".csv"
                df = self.download_and_read_csv(
                    csv_filename, refresh_cache=self.refresh_cache
                )
            except HTTPError:
                csv_filename = target_vocabulary + "_to_" + self.vocabulary + ".csv"
                df = self.download_and_read_csv(
                    csv_filename, refresh_cache=self.refresh_cache
                )
            mapping = defaultdict(list)
            for _, row in df.iterrows():
                mapping[row[self.vocabulary]].append(row[target_vocabulary])
            print(
                f"Saved {self.vocabulary}->{target_vocabulary} mapping to {pickle_filepath}"
            )
            save_pickle(mapping, pickle_filepath)

        return mapping

    def map_to(self, code, target_vocabulary):
        if target_vocabulary not in self.valid_mappings:
            raise ValueError(
                f"Cannot map from {self.vocabulary} to {target_vocabulary}"
            )
        if target_vocabulary not in self.cached_mappings:
            self.cached_mappings[target_vocabulary] = self.load_mapping(
                target_vocabulary
            )
        return self.cached_mappings[target_vocabulary][code]
