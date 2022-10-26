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

class InnerMap:
    """lookup function within one coding system

    Parameters:
        vocabulary: str, vocabulary name. For example, "ICD9CM", "ICD10CM", "ICD9PROC", "ICD10PROC", "CCSCM", "CCSPROC", "ATC", "NDC", "RxNorm"
        regresh_cache: bool, whether to refresh the cache
            
    **Examples:**
        >>> from pyhealth.medcode import InnerMap
        >>> rxnorm = InnerMap(vocabulary="RxNorm")
        Loaded RxNorm code from /root/.cache/pyhealth/medcode/RxNorm.pkl
        >>> rxnorm.lookup("209387")
        acetaminophen 325 MG Oral Tablet [Tylenol]
        
        >>> ccscm = InnerMap("CCSCM")
        Loaded CCSCM code from /root/.cache/pyhealth/medcode/CCSCM.pkl
        >>> ccscm.lookup("108")
        'chf;nonhp: Congestive heart failure, nonhypertensive'
        
        >>> icd9cm = InnerMap("ICD9CM")
        Processing ICD9CM code...
        Saved ICD9CM code to /root/.cache/pyhealth/medcode/ICD9CM.pkl
        >>> "428.0" in icd9cm # let's first check if the code is in ICD9CM
        True
        >>> icd9cm.get_ancestors("428.0")
        ['428', '420-429.99', '390-459.99', '001-999.99']
        >>> icd9cm.lookup("4280") # non-standard format
        'Congestive heart failure, unspecified'
        
        >>> atc = InnerMap("ATC")
        Processing ATC code...
        Saved ATC code to /root/.cache/pyhealth/medcode/ATC.pkl
        >>> atc.lookup("M01AE51")
        'ibuprofen, combinations'
        >>> atc.lookup("M01AE51", "drugbank_id")
        DB01050
        >>> atc.lookup("M01AE51", "description")
        Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) derived from propionic acid and it is considered the first ...
        On the available products, ibuprofen is administered as a racemic mixture. Once administered, the R-enantiomer ...
        >>> atc.lookup("M01AE51", "indication")
        Ibuprofen is the most commonly used and prescribed NSAID. It is very common ...[A39097]
        Due to its activity against prostaglandin and thromboxane synthesis, ibuprofen ...:
        * Patent Ductus Arteriosus - it is a neonatal condition wherein the ductus arteriosus ...
        * Rheumatoid- and osteo-arthritis - ibuprofen is very commonly used in the symptomatic ...
        * Investigational uses - efforts have been put into developing ibuprofen for the prophy ...
        >>> atc.lookup("M01AE51", "smiles")
        CC(C)CC1=CC=C(C=C1)C(C)C(O)=O
        
    """

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
