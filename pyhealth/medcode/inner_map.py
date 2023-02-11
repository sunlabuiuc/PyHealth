import logging
import os
from abc import ABC, abstractmethod
from typing import List

import networkx as nx
import pandas as pd

import pyhealth.medcode as medcode
from pyhealth.medcode.utils import MODULE_CACHE_PATH, download_and_read_csv
from pyhealth.utils import load_pickle, save_pickle

logger = logging.getLogger(__name__)


# TODO: add this callable method: InnerMap(vocab)
class InnerMap(ABC):
    """Contains information for a specific medical code system.

    `InnerMap` is a base abstract class for all medical code systems.
    It will be instantiated as a specific medical code system with
    `InnerMap.load(vocabulary).`

    Note:
        This class cannot be instantiated using `__init__()` (throws an error).
    """

    @abstractmethod
    def __init__(
        self,
        vocabulary: str,
        refresh_cache: bool = False,
    ):
        # abstractmethod prevents initialization of this class
        self.vocabulary = vocabulary

        pickle_filepath = os.path.join(MODULE_CACHE_PATH, self.vocabulary + ".pkl")
        csv_filename = self.vocabulary + ".csv"
        if os.path.exists(pickle_filepath) and (not refresh_cache):
            logger.debug(f"Loaded {vocabulary} code from {pickle_filepath}")
            self.graph = load_pickle(pickle_filepath)
        else:
            logger.debug(f"Processing {vocabulary} code...")
            df = download_and_read_csv(csv_filename, refresh_cache)
            # create graph
            df = df.set_index("code")
            self.graph = nx.DiGraph()
            # add nodes
            for code, row in df.iterrows():
                row_dict = row.to_dict()
                row_dict.pop("parent_code", None)
                self.graph.add_node(code, **row_dict)
            # add edges
            for code, row in df.iterrows():
                if "parent_code" in row:
                    if not pd.isna(row["parent_code"]):
                        self.graph.add_edge(row["parent_code"], code)
            logger.debug(f"Saved {vocabulary} code to {pickle_filepath}")
            save_pickle(self.graph, pickle_filepath)
        return

    def __repr__(self):
        return f"InnerMap(vocabulary={self.vocabulary}, graph={self.graph})"

    @classmethod
    def load(_, vocabulary: str, refresh_cache: bool = False):
        """Initializes a specific medical code system inheriting from `InnerMap`.

        Args:
            vocabulary: vocabulary name. E.g., "ICD9CM", "ICD9PROC".
            refresh_cache: whether to refresh the cache. Default is False.

        Examples:
            >>> from pyhealth.medcode import InnerMap
            >>> icd9cm = InnerMap.load("ICD9CM")
            >>> icd9cm.lookup("428.0")
            'Congestive heart failure, unspecified'
            >>> icd9cm.get_ancestors("428.0")
            ['428', '420-429.99', '390-459.99', '001-999.99']
        """
        cls = getattr(medcode, vocabulary)
        return cls(refresh_cache=refresh_cache)

    @property
    def available_attributes(self) -> List[str]:
        """Returns a list of available attributes.

        Returns:
            List of available attributes.
        """
        return list(list(self.graph.nodes.values())[0].keys())

    def stat(self):
        """Prints statistics of the code system."""
        print()
        print(f"Statistics for {self.vocabulary}:")
        print(f"\t- Number of nodes: {len(self.graph.nodes)}")
        print(f"\t- Number of edges: {len(self.graph.edges)}")
        print(f"\t- Available attributes: {self.available_attributes}")
        print()

    @staticmethod
    def standardize(code: str) -> str:
        """Standardizes a given code.

        Subclass will override this method based on different
        medical code systems.
        """
        return code

    @staticmethod
    def convert(code: str, **kwargs) -> str:
        """Converts a given code.

        Subclass will override this method based on different
        medical code systems.
        """
        return code

    def lookup(self, code: str, attribute: str = "name"):
        """Looks up the code.

        Args:
            code: code to look up.
            attribute: attribute to look up. One of `self.available_attributes`.
                Default is "name".

        Returns:
            The attribute value of the code.
        """
        code = self.standardize(code)
        return self.graph.nodes[code][attribute]

    def __contains__(self, code: str) -> bool:
        """Checks if the code is in the code system."""
        code = self.standardize(code)
        return code in self.graph.nodes

    def get_ancestors(self, code: str) -> List[str]:
        """Gets the ancestors of the code.

        Args:
            code: code to look up.

        Returns:
            List of ancestors ordered from the closest to the farthest.
        """
        code = self.standardize(code)
        # ordered ancestors
        ancestors = nx.ancestors(self.graph, code)
        ancestors = list(ancestors)
        ancestors = sorted(
            ancestors, key=lambda x: (nx.shortest_path_length(self.graph, x, code), x)
        )
        return ancestors

    def get_descendants(self, code: str) -> List[str]:
        """Gets the descendants of the code.

        Args:
            code: code to look up.

        Returns:
            List of ancestors ordered from the closest to the farthest.
        """
        code = self.standardize(code)
        # ordered descendants
        descendants = nx.descendants(self.graph, code)
        descendants = list(descendants)
        descendants = sorted(
            descendants, key=lambda x: (nx.shortest_path_length(self.graph, code, x), x)
        )
        return descendants


if __name__ == "__main__":
    icd9cm = InnerMap.load("ICD9CM")
    print(icd9cm.stat())
    print("428.0" in icd9cm)
    print(icd9cm.lookup("4280"))
    print(icd9cm.get_ancestors("428.0"))
    print(icd9cm.get_descendants("428.0"))