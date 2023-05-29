import logging
import os
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from pyhealth.medcode.kg_emb.datasets import BaseKGDataset

logger = logging.getLogger(__name__)

class UMLSDataset(BaseKGDataset):
    """Base UMLS knowleddge graph dataset

    Dataset is available at https://www.nlm.nih.gov/research/umls/index.html

    Args:
        dataset_name: name of the dataset.
        root: root directory of the raw data (should contain many csv files).
        dev: whether to enable dev mode (only use a small subset of the data).
            Default is False.
        refresh_cache: whether to refresh the cache; if true, the dataset will
            be processed from scratch and the cache will be updated. Default is False.
    
    """

    def raw_graph_process(self):
        pandarallel.initialize(progress_bar=False)
        if self.dev == False:
            self.graph_path = os.path.join(self.root, "graph.txt")
        else:
            self.graph_path = os.path.join(self.root, "graph_filtered.txt")

        if os.path.exists(self.graph_path):
            logger.debug("umls knowledge graph exists and load umls")
        else:
            logger.debug("umls does not exist")

        graph_df = pd.read_csv(
            self.graph_path, 
            sep='\t',
            names=['e1', 'r', 'e2']
        )

        entity_list = pd.unique(graph_df[['e1', 'e2']].values.ravel('K'))
        relation_list = pd.unique(graph_df['r'].values)
        self.entity2id = {val: i for i, val in enumerate(entity_list)}
        self.relation2id = {val: i for i, val in enumerate(relation_list)}
        self.entity_num = len(self.entity2id)
        self.relation_num = len(self.relation2id)

        for index, row in graph_df.iterrows():
            self.triples.append(
                (self.entity2id[row['e1']], self.relation2id[row['r']], self.entity2id[row['e2']]))

        return


if __name__ == "__main__":
    dataset = UMLSDataset(
        root="https://storage.googleapis.com/pyhealth/umls/",
        dev=True,
        refresh_cache=True,
    )
    dataset.stat()
    dataset.info()