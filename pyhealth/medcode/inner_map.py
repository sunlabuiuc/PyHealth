import os
from abc import ABC

import networkx as nx
import pandas as pd

from pyhealth import BASE_CACHE_PATH
from pyhealth.medcode.utils import download_and_read_csv
from pyhealth.utils import create_directory, load_pickle, save_pickle

BASE_URL = "https://storage.googleapis.com/pyhealth/resource/"
MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "medcode")
create_directory(MODULE_CACHE_PATH)


# TODO: add comments

class InnerMap(ABC):
    """Abstract class for inner mapping within a coding system."""

    def __init__(
            self,
            vocabulary: str,
            refresh_cache: bool = False,
    ):
        self.vocabulary = vocabulary
        self.refresh_cache = refresh_cache

        pickle_filepath = os.path.join(MODULE_CACHE_PATH, self.vocabulary + ".pkl")
        csv_filename = self.vocabulary + ".csv"
        if os.path.exists(pickle_filepath) and (not refresh_cache):
            print(f"Loaded {vocabulary} code from {pickle_filepath}")
            self.graph = load_pickle(pickle_filepath)
        else:
            print(f"Processing {vocabulary} code...")
            df = download_and_read_csv(csv_filename, refresh_cache)
            self.graph = self.build_graph(df)
            print(f"Saved {vocabulary} code to {pickle_filepath}")
            save_pickle(self.graph, pickle_filepath)

    @staticmethod
    def build_graph(df):
        df = df.set_index("code")
        graph = nx.DiGraph()
        # add nodes
        for code, row in df.iterrows():
            row_dict = row.to_dict()
            row_dict.pop("parent_code", None)
            graph.add_node(code, **row_dict)
        # add edges
        for code, row in df.iterrows():
            if "parent_code" in row:
                if not pd.isna(row["parent_code"]):
                    graph.add_edge(row["parent_code"], code)
        return graph

    @property
    def available_attributes(self):
        return list(list(self.graph.nodes.values())[0].keys())

    def stat(self):
        print()
        print(f"Statistics for {self.vocabulary}:")
        print(f"\t- Number of nodes: {len(self.graph.nodes)}")
        print(f"\t- Number of edges: {len(self.graph.edges)}")
        print(f"\t- Available attributes: {self.available_attributes}")
        print()

    def standardize(self, code: str):
        return code

    def postprocess(self, code: str, **kwargs):
        return code

    def lookup(self, code, attribute: str = "name"):
        code = self.standardize(code)
        return self.graph.nodes[code][attribute]

    def __contains__(self, code):
        code = self.standardize(code)
        return code in self.graph.nodes

    def get_ancestors(self, code):
        code = self.standardize(code)
        # ordered ancestors
        ancestors = nx.ancestors(self.graph, code)
        ancestors = list(ancestors)
        ancestors = sorted(
            ancestors,
            key=lambda x: (nx.shortest_path_length(self.graph, x, code), x)
        )
        return ancestors

    def get_descendants(self, code):
        code = self.standardize(code)
        # ordered descendants
        descendants = nx.descendants(self.graph, code)
        descendants = list(descendants)
        descendants = sorted(
            descendants,
            key=lambda x: (nx.shortest_path_length(self.graph, code, x), x)
        )
        return descendants
