import os
from abc import ABC
from collections import defaultdict
from urllib.parse import urljoin

import networkx as nx
import pandas as pd

from pyhealth import CACHE_PATH
from pyhealth.utils import create_directory, download

from urllib.error import HTTPError


BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"


class BaseCode(ABC):
    """ Abstract base coding system """""

    def __init__(self, valid_mappings: list = None):
        if valid_mappings is None:
            valid_mappings = []
        self.vocabulary = self.__class__.__name__
        self.cache_path = os.path.join(CACHE_PATH, "medcode")
        create_directory(self.cache_path)
        df = self.read_or_download_csv(self.vocabulary + '.csv')
        self.graph = self.build_graph(df)
        self.valid_mappings = valid_mappings
        self.cached_mappings = {}

    def read_or_download_csv(self, filename: str):
        if not os.path.exists(os.path.join(self.cache_path, filename)):
            download(urljoin(BASE_URL, filename), os.path.join(self.cache_path, filename))
        return pd.read_csv(os.path.join(self.cache_path, filename), dtype=str)

    @staticmethod
    def build_graph(df):
        df = df.set_index("code")
        graph = nx.DiGraph()
        for code, row in df.iterrows():
            graph.add_node(code, **row)
        for code, row in df.iterrows():
            if 'parent_code' in row:
                graph.add_edge(row['parent_code'], code)
        return graph

    def load_mapping(self, target_vocabulary):
        if target_vocabulary not in self.valid_mappings:
            raise ValueError(f"Cannot map from {self.vocabulary} to {target_vocabulary}")
        try:
            df = self.read_or_download_csv(self.vocabulary + '_to_' + target_vocabulary + '.csv')
        except HTTPError:
            df = self.read_or_download_csv(target_vocabulary + '_to_' + self.vocabulary + '.csv')
        mapping = defaultdict(list)
        for _, row in df.iterrows():
            mapping[row[self.vocabulary]].append(row[target_vocabulary])
        return mapping

    def map_to(self, code, target_vocabulary):
        if target_vocabulary not in self.valid_mappings:
            raise ValueError(f"Cannot map from {self.vocabulary} to {target_vocabulary}")
        if target_vocabulary not in self.cached_mappings:
            self.cached_mappings[target_vocabulary] = self.load_mapping(target_vocabulary)
        return self.cached_mappings[target_vocabulary][code]
