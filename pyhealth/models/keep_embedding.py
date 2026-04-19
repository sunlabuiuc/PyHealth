import os
import sys
import logging
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter

import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd
# from scipy import sparse

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel

logger = logging.getLogger(__name__)

class N2V():
    """
    """
    def __init__(
        self, 
        path:str, 
        domain_type:str, 
        # depth_limit:int, 
        embedding_dim:int,
        walk_length:int,
        num_walks:int
    ):
        self.path = path
        self.domain_type = domain_type
        # self.depth_limit = depth_limit
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        # Map domain_type to OMOP domain names
        self.domain_map = {
            "condition": ["Condition"],
            "drug": ["Drug"],
            "procedure": ["Procedure"],
            "all": ["Condition", "Drug", "Procedure"],
        }
        # If all then no filtering needed 
    
        # Create graph from concept and their relationships data
    
    def create_graph(self) -> nx.DiGraph:
        """
        Create a directed graph from OMOP concept relationships.
        
        Loads concepts and their relationships from CSV files, filters by domain_type,
        and builds a NetworkX DiGraph where nodes are concept IDs and edges are
        concept relationships (maps_to).
        
        Returns:
            nx.DiGraph: Directed graph with concept_id as nodes and relationships as edges.
        
        Raises:
            FileNotFoundError: If CSV files are not found.
            ValueError: If no concepts found for specified domains.
        """
        # Load concept table
        concept_path = os.path.join(self.path, "2b_concept.csv")
        
        # Load concept relationships table
        concept_relationship_path = os.path.join(self.path, "2b_concept_relationship.csv")

        # if not os.path.exists(concept_path):
        #     raise FileNotFoundError(f"2b_concept.csv not found at {concept_path}")
        # if not os.path.exists(concept_relationship_path):
        #     raise FileNotFoundError(f"2b_concept_relationship.csv not found at {concept_relationship_path}")
        
        # Read CSV files
        print(f"Loading concepts from {concept_path}")
        concept_df = pd.read_csv(concept_path, dtype=str)
        
        print(f"Loading concept relationships from {concept_relationship_path}")
        concept_rel_df = pd.read_csv(concept_relationship_path, dtype=str)
        
        target_domains = self.domain_map[self.domain_type]

        # Filter concepts by target domains
        # concept_df = concept_df[concept_df["domain_id"].isin(target_domains)].copy()
        
        # if len(concept_df) == 0:
        #     raise ValueError(f"No concepts found for domains: {target_domains}")
        
        # print(f"Filtered to {len(concept_df)} concepts in domains: {target_domains}")
        
        # # Create set of filtered concept IDs for quick lookup
        filtered_concept_ids = set(concept_df["concept_id"].values)
        
        # # Filter to relationships where both concepts are in our domain set
        concept_rel_df = concept_rel_df[
            (concept_rel_df["concept_id_1"].isin(filtered_concept_ids)) &
            (concept_rel_df["concept_id_2"].isin(filtered_concept_ids))
        ].copy()
        
        print(f"Found {len(concept_rel_df)} relationships between filtered concepts")
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Add all filtered concepts as nodes
        for concept_id, row in concept_df.iterrows():
            graph.add_node(
                row["concept_id"],
                name=row["concept_name"],
                domain=row["domain_id"]
            )
        
        print(f"Added {len(graph.nodes())} nodes to graph")
        
        # Add edges from concept relationships
        # Typically "maps_to" relationship indicates concept_id_1 maps to concept_id_2
        for _, row in concept_rel_df.iterrows():
            concept_1 = row["concept_id_1"]
            concept_2 = row["concept_id_2"]
            rel_type = row.get("relationship_id", "maps_to")
            
            # Add directed edge from concept_1 to concept_2
            graph.add_edge(concept_1, concept_2, relationship=rel_type)
        
        print(f"Added {len(graph.edges())} edges to graph")
        
        return graph

    def generate_embeddings(self):
        """
        Generate node embeddings using Node2Vec algorithm.
        
        Creates a graph from OMOP concepts and applies Node2Vec to generate
        embeddings for each concept based on its network structure.
        
        Returns:
            gensim.models.Word2Vec: Trained Node2Vec model for concept embeddings.
        """
        # Create graph from concepts and relationships
        logger.info("Creating concept graph")
        graph = self.create_graph()
        
        logger.info(f"Graph created with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        
        if len(graph.nodes()) == 0:
            raise ValueError("Graph is empty, cannot generate embeddings")
        
        # Initialize and fit Node2Vec
        logger.info(
            f"Initializing Node2Vec with embedding_dim={self.embedding_dim}, "
            f"walk_length={self.walk_length}, num_walks={self.num_walks}"
        )
        node2vec = Node2Vec(
            graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            workers=4
        )
        
        # Train the model
        logger.info("Training Node2Vec model")
        self.model = node2vec.fit(window=10, min_count=1, epochs=1)
        
        logger.info("Node2Vec training completed")
        logger.info(f"Model vocabulary size: {len(self.model.wv)}")
        
        return self.model



class KeepEmbedding(BaseModel):
    """Knowledge-Enhanced Patient Embedding model using OMOP data and node2vec."""
    
    def __init__(self, dataset: SampleDataset):
        """
        Initialize KeepEmbedding model.
        
        Args:
            dataset: An OMOPDataset instance containing patient clinical data.
        """
        super().__init__(dataset=dataset)
    
    # def build_cooccurrence_matrix(
    #     self,
    #     graph: nx.DiGraph,
    #     domain_type: str = "condition",
    #     min_occurrences: int = 2,
    # ) -> Tuple[sparse.csr_matrix, List[str]]:
    #     """
    #     Build co-occurrence matrix from patient histories using dense roll-up.
        
    #     Iterates through all patients in the dataset, collects concept codes from
    #     their complete medical history, applies dense roll-up to ancestor concepts,
    #     and builds a sparse co-occurrence matrix.
        
    #     Args:
    #         graph (nx.DiGraph): NetworkX graph from Node2Vec.create_graph()
    #         domain_type (str): Concept domain to include:
    #             - "condition": condition_occurrence events
    #             - "drug": drug_exposure events
    #             - "procedure": procedure_occurrence events
    #             - "all": All three event types. Default is "condition".
    #         min_occurrences (int): Minimum number of times a concept must appear
    #             in a patient's history to be retained (per paper requirement).
    #             Default is 2.
        
    #     Returns:
    #         Tuple[sparse.csr_matrix, List[str]]:
    #             - X: Sparse CSR matrix where X[i,j] = co-occurrence frequency
    #                  between concept_i and concept_j across all patients
    #             - concept_ids: List of concept IDs corresponding to matrix rows/columns
        
    #     Raises:
    #         ValueError: If domain_type is invalid or dataset is empty.
    #     """
    #     logger.info(f"Building co-occurrence matrix for domain_type={domain_type}")
        
    #     # Map domain_type to event types and fields
    #     domain_map = {
    #         "condition": [("condition_occurrence", "condition_concept_id")],
    #         "drug": [("drug_exposure", "drug_concept_id")],
    #         "procedure": [("procedure_occurrence", "procedure_concept_id")],
    #         "all": [
    #             ("condition_occurrence", "condition_concept_id"),
    #             ("drug_exposure", "drug_concept_id"),
    #             ("procedure_occurrence", "procedure_concept_id"),
    #         ],
    #     }
        
    #     if domain_type not in domain_map:
    #         raise ValueError(
    #             f"domain_type must be one of {list(domain_map.keys())}, got {domain_type}"
    #         )
        
    #     event_types = domain_map[domain_type]
        
    #     # Extract concept IDs from graph nodes
    #     concept_ids = sorted(list(graph.nodes()))
    #     concept_id_to_idx = {cid: idx for idx, cid in enumerate(concept_ids)}
        
    #     logger.info(f"Graph has {len(concept_ids)} concepts")
        
    #     if len(concept_ids) == 0:
    #         raise ValueError("Graph is empty, cannot build co-occurrence matrix")
        
    #     if self.dataset is None or len(self.dataset.unique_patient_ids) == 0:
    #         raise ValueError("Dataset is empty, cannot build co-occurrence matrix")
        
    #     # Initialize co-occurrence counter
    #     cooc_counts = defaultdict(int)
        
    #     # Iterate through all patients
    #     patient_ids = self.dataset.unique_patient_ids
    #     logger.info(f"Processing {len(patient_ids)} patients")
        
    #     for patient_id in patient_ids:
    #         try:
    #             patient = self.dataset.get_patient(patient_id)
    #         except Exception as e:
    #             logger.warning(f"Failed to load patient {patient_id}: {e}")
    #             continue
            
    #         # Collect all codes from patient's complete history
    #         all_codes = []
    #         for event_type, field in event_types:
    #             try:
    #                 events = patient.get_events(event_type=event_type)
    #                 codes = []
    #                 for event in events:
    #                     code = str(getattr(event, field, ""))
    #                     if code and code != "nan":
    #                         codes.append(code)
    #                 all_codes.extend(codes)
    #             except Exception as e:
    #                 logger.debug(f"Could not get {event_type} for patient {patient_id}: {e}")
    #                 continue
            
    #         if len(all_codes) == 0:
    #             continue
            
    #         # Count occurrences of each code
    #         code_counts = Counter(all_codes)
            
    #         # Filter codes with min_occurrences
    #         retained_codes = [
    #             code for code, count in code_counts.items()
    #             if count >= min_occurrences
    #         ]
            
    #         if len(retained_codes) == 0:
    #             continue
            
    #         # Apply dense roll-up: map each code to ALL ancestors in graph
    #         rolled_codes = set()
    #         for code in retained_codes:
    #             rolled_codes.add(code)  # Include self
    #             if code in graph.nodes():
    #                 # Find all ancestors
    #                 try:
    #                     ancestors = nx.ancestors(graph, code)
    #                     rolled_codes.update(ancestors)
    #                 except Exception as e:
    #                     logger.debug(f"Could not find ancestors for {code}: {e}")
            
    #         # Build co-occurrence pairs (only for codes in graph)
    #         rolled_codes_in_graph = [c for c in rolled_codes if c in graph.nodes()]
            
    #         if len(rolled_codes_in_graph) > 1:
    #             # Create all pairs
    #             for i, code_i in enumerate(rolled_codes_in_graph):
    #                 for code_j in rolled_codes_in_graph[i + 1 :]:
    #                     idx_i = concept_id_to_idx[code_i]
    #                     idx_j = concept_id_to_idx[code_j]
                        
    #                     # Store symmetric pairs
    #                     if idx_i <= idx_j:
    #                         cooc_counts[(idx_i, idx_j)] += 1
    #                     else:
    #                         cooc_counts[(idx_j, idx_i)] += 1
        
    #     logger.info(f"Generated {len(cooc_counts)} unique co-occurrence pairs")
        
    #     # Build sparse matrix
    #     if len(cooc_counts) == 0:
    #         logger.warning("No co-occurrences found, returning empty sparse matrix")
    #         X = sparse.csr_matrix((len(concept_ids), len(concept_ids)), dtype=np.float32)
    #         return X, concept_ids
        
    #     # Extract rows, columns, and data
    #     rows, cols, data = [], [], []
    #     for (i, j), count in cooc_counts.items():
    #         rows.append(i)
    #         cols.append(j)
    #         data.append(count)
    #         # Add symmetric entry
    #         rows.append(j)
    #         cols.append(i)
    #         data.append(count)
        
    #     # Create COO matrix and convert to CSR
    #     X = sparse.coo_matrix(
    #         (data, (rows, cols)),
    #         shape=(len(concept_ids), len(concept_ids)),
    #         dtype=np.float32,
    #     )
    #     X = X.tocsr()
        
    #     logger.info(
    #         f"Built sparse co-occurrence matrix: shape={X.shape}, nnz={X.nnz}, "
    #         f"sparsity={1 - X.nnz / (X.shape[0] * X.shape[1]):.4f}"
    #     )
        
    #     return X, concept_ids