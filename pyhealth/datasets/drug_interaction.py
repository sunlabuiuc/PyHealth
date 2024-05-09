"""
File: drug_interaction.ipynb

PyHealth Drug Interaction Dataset
"""

import sqlite3
from enum import Enum
from tempfile import NamedTemporaryFile
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from torch.utils.data import Dataset, random_split


class DrugBankFeatureType(Enum):
    """Enum of different extractable DrugBank features.
    
    These features will compose the similarity matrix of drugs.
    
    Enzyme: The enzyme responsible for processing a drug within the body.
    Pathway: The pathway through which a drug is absorbed, distributed, and metabolized in the body.
    Smile: The ASCII-encoded molecular structure of a drug.
    Target: Molecule within the body the drug intends to impact.
    """

    enzyme = 'enzyme'
    pathway = 'pathway'
    smile = 'smile'
    target = 'target'


class DrugInteractionDataset(Dataset):
    """This class is a datset for maintaining drug similarity and
    drug interaction data. It loads drug and feature lists from the DrugBank DB provided by
    DDIMDL (Deng et Al): https://github.com/YifanDengWHU/DDIMDL. 

    This class is a Pytorch dataset for positive/negative examples of various drug interactions
    types provided as defined by Deng et Al. Alongisde these examples, the dataset
    includes a drug similarity matrix which can be used for generating a GNN of drug relations.
    Together, the matrix and interaction data can be used to create and train a GNN 
    for generating drug interaction embeddings.
    """

    _feature_type: DrugBankFeatureType

    drug_list: pd.DataFrame
    drug_features: pd.DataFrame
    _interactions: pd.DataFrame
    _non_interactions: pd.DataFrame
    full_interactions: pd.DataFrame
    
    drug_similarity_matrix: pd.DataFrame


    def __init__(self, drug_bank_feature: DrugBankFeatureType, include_non_interactions: bool = True, do_pca: bool = True) -> None:
        """Initialize a Dataset of Drug Interactions.
        
        Args:
            drug_bank_feature: Feature to extract from Drugbank DB.
            include_non_interactions: Should non-interactions be included in Dataset.
            do_pca: Should PCA be applied to the similarity matrix.
        """
        self._feature_type = drug_bank_feature
        self.drug_list, self.drug_features, self._interactions, self._non_interactions = self._load_drug_features(drug_bank_feature)

        if include_non_interactions:
            self.full_interactions = pd.concat([self._interactions, self._non_interactions], sort=True)
        else:
            self.full_interactions = self._interactions

        self.dataset_iter = self.full_interactions.iterrows()
        self.drug_similarity_matrix = self._generate_jaccard_similarity(self.drug_features, do_pca)
    
    def _load_drug_features(self, feature: DrugBankFeatureType) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads drug features from EventDB.
        
        Args:
            feature: Drug Bank feature to extract from EventDB.
        Returns:
            Tuple of dataframes:
                * Drug names
                * One hot encoding of drug features
                * Drug interactions
                * Non-drug interactions (event_type = -1)
        """
        with NamedTemporaryFile() as tmp:
            # Download data from event DB.
            url = 'https://raw.githubusercontent.com/YifanDengWHU/DDIMDL/master/event.db'
            response = requests.get(url)
            tmp.write(response.content)
            conn = sqlite3.connect(tmp.name)

            # Generate one-hot encoding of drug features.
            raw_drug = pd.read_sql('select * from drug;', conn)
            raw_drug_name = raw_drug.set_index('name')
            one_hot_encoded = raw_drug[feature.value].str.get_dummies('|')

            # Generate indexing of raw events.
            extraction = pd.read_sql('select * from extraction;', conn)
            extraction['event'] = extraction['mechanism'] + ' ' + extraction['action']
            raw_events = extraction['event'].value_counts().to_frame()
            raw_events['event_index'] = np.arange(raw_events.shape[0])

            # Generate drug interactions with event type.
            df_raw_full_pos = extraction.join(raw_events, on='event', lsuffix='_eventL')[['drugA', 'drugB', 'event_index']]
            df_raw_full_pos = df_raw_full_pos.join(raw_drug_name['index'], on='drugA')
            df_raw_full_pos = df_raw_full_pos.join(raw_drug_name['index'], on='drugB', rsuffix='_drugB')
            df_raw_full_pos = df_raw_full_pos[['event_index', 'index', 'index_drugB']].rename({"index": "drugA", "index_drugB": "drugB"}, axis=1)

            # Generate non drug interactions, with event type of -1.
            all_interactions = {(row["drugA"], row["drugB"]) for _, row in df_raw_full_pos.iterrows()}.union({(row["drugB"], row["drugA"]) for _, row in df_raw_full_pos.iterrows()})
            all_drug_combos = {(idx1, idx2) for idx1 in range(len(raw_drug_name)) for idx2 in range(len(raw_drug_name))}
            unique_non_interactions = {tuple(sorted(drug_combo)) for drug_combo in (all_drug_combos - all_interactions)}
            non_interactions = pd.DataFrame.from_records(data=list(unique_non_interactions), columns=["drugA", "drugB"])
            non_interactions["event_index"] = -1

            return raw_drug_name, one_hot_encoded, df_raw_full_pos, non_interactions

    def _generate_jaccard_similarity(self, drug_features: pd.DataFrame, do_pca: bool = True) -> pd.DataFrame:
        """Generates a Jaccard similarity matrix from a matrix of drug features.
        
        Args:
            drug_features: One hot encoded dataframe of drug features.
            do_pca: Should PCA be performed on the similarity matrix.
        Returns:
            Dataframe of drug similarity matrix.
        """
        similarity = 1 - pairwise_distances(drug_features.to_numpy().astype(bool), metric='jaccard')

        if not do_pca:
            return similarity

        sim_pca = PCA(n_components=len(similarity))
        sim_pca.fit(similarity)
        return pd.DataFrame(sim_pca.transform(similarity))

    def __len__(self) -> int:
        """Number of samples in the Dataset."""
        return len(self.full_interactions)

    def __getitem__(self, index):
        """Generates one sample of data"""

        try:
            _, ex_row = next(self.dataset_iter)
        except StopIteration:
            self.it = self.full_interactions.iterrows ()
            _, ex_row = next(self.dataset_iter)

        x = {
            "drugA": ex_row["drugA"],
            "drugB": ex_row["drugB"]
        }

        return x, ex_row["event_index"]

    def stat(self) -> str:
        """Returns statistics of the dataset."""
        lines = list()
        lines.append("")
        lines.append(f"Statistics of Dataset:")
        lines.append(f"\t- Dataset Name: {self._feature_type.value}")
        lines.append(f"\t- Number of samples: {len(self)}")
        lines.append(f"\t- Number of drugs: {len(self.drug_list)}")
        lines.append(f"\t- Number of drug interactions: {len(self._interactions)}")
        lines.append(f"\t- Number of drug non-interactions: {len(self._non_interactions)}")
        lines.append(f"\t- Number of distinct interactions: {self._interactions['event_index'].nunique()}")
        lines.append(f"\t- Number of drug features: {len(self.drug_features.columns) - 1}")
        lines.append("")
        print("\n".join(lines))
        return "\n".join(lines)

if __name__ == "__main__":
    from math import floor

    from torch.utils.data import DataLoader

    for feature_type in DrugBankFeatureType.__members__.values():
        print(f"Generating dataset for feature type: {feature_type.value}")
        dataset = DrugInteractionDataset(drug_bank_feature=feature_type)

        dataset.stat()

        dataset_splits = [floor(len(dataset) * 0.4), floor(len(dataset) * 0.3)]
        dataset_splits.append(len(dataset) - sum(dataset_splits))

        train, validation, test = random_split(dataset, dataset_splits)

        # General data loader params
        params = {'batch_size': 16,
                'shuffle': True,
                'num_workers': 1}

        # Example creating train/validation/test data loaders
        training_generator = DataLoader(train, **params)
        validation_generator = DataLoader(validation, **params)
        test_generator = DataLoader(test, **params)

### Output:
# Generating dataset for feature type: enzyme
# 
# Statistics of Dataset:
#         - Dataset Name: enzyme
#         - Number of samples: 163878
#         - Number of drugs: 572
#         - Number of drug interactions: 37264
#         - Number of drug non-interactions: 126614
#         - Number of distinct interactions: 65
#         - Number of drug features: 201
# 
# Generating dataset for feature type: pathway
# 
# Statistics of Dataset:
#         - Dataset Name: pathway
#         - Number of samples: 163878
#         - Number of drugs: 572
#         - Number of drug interactions: 37264
#         - Number of drug non-interactions: 126614
#         - Number of distinct interactions: 65
#         - Number of drug features: 956
# 
# Generating dataset for feature type: smile
# 
# Statistics of Dataset:
#         - Dataset Name: smile
#         - Number of samples: 163878
#         - Number of drugs: 572
#         - Number of drug interactions: 37264
#         - Number of drug non-interactions: 126614
#         - Number of distinct interactions: 65
#         - Number of drug features: 582
# 
# Generating dataset for feature type: target
# 
# Statistics of Dataset:
#         - Dataset Name: target
#         - Number of samples: 163878
#         - Number of drugs: 572
#         - Number of drug interactions: 37264
#         - Number of drug non-interactions: 126614
#         - Number of distinct interactions: 65
#         - Number of drug features: 1161