import os
from collections import defaultdict, Counter

import networkx as nx
from node2vec import Node2Vec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from pyhealth.datasets import SampleDataset
from .base_model import BaseModel

class N2V:
    """Generate Node2Vec embeddings for OMOP concepts.

    Builds a directed knowledge graph from OMOP concept relationship tables
    and trains Node2Vec to create graph embeddings.

    Attributes:
        embedding_dim: Dimension of learned embeddings.
        walk_length: Length of each random walk.
        num_walks: Number of walks per node.
    """
    def __init__(
        self,
        embedding_dim: int | None = None,
        walk_length: int | None = None,
        num_walks: int | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
    
    def create_graph(
        self, path: str, domain_type: list[str]
    ) -> nx.DiGraph:
        """Create a Network directed graph from OMOP concept relationships.

        Args:
            path: Path to OMOP concept CSV files.
            domain_type: List of domain IDs to include.

        Returns:
            Directed graph with concept IDs as nodes and relationships
                as edges.
        """
        # Load concept table
        concept_path = os.path.join(path, "2b_concept.csv")  
        print(f"Loading concepts from {concept_path}")
        concept_df = pd.read_csv(concept_path, dtype=str)
 
        # Load concept relationships table
        concept_relationship_path = os.path.join(path, "2b_concept_relationship.csv")
        print(f"Loading concept relationships from {concept_relationship_path}")
        concept_rel_df = pd.read_csv(concept_relationship_path, dtype=str)
        
        print(f"Loaded {len(concept_df)} concepts and {len(concept_rel_df)} relationships")

        if domain_type != ["all"]:
            # Filter concepts by target domain
            concept_df = concept_df[concept_df["domain_id"].isin(domain_type)].copy()
        
            print(f"Filtered to {len(concept_df)} concepts in domains: {domain_type}")
        
        # Create set of filtered concept IDs for quick lookup
        filtered_concept_ids = set(concept_df["concept_id"].values)
        print(f"Created set of {len(filtered_concept_ids)} concept IDs")
        
        # Filter to relationships where both concepts are in our domain set
        concept_rel_df = concept_rel_df[
            (concept_rel_df["concept_id_1"].isin(filtered_concept_ids)) &
            (concept_rel_df["concept_id_2"].isin(filtered_concept_ids))
        ].copy()
        
        print(f"Found {len(concept_rel_df)} relationships between concepts")
        
        # Create directed graph
        graph = nx.DiGraph()
        
        # Add all filtered concepts as nodes
        for _, row in concept_df.iterrows():
            graph.add_node(
                row["concept_id"],
                name=row["concept_name"],
                domain=row["domain_id"]
            )
        
        # Add edges from concept relationships
        for _, row in concept_rel_df.iterrows():
            concept_1 = row["concept_id_1"]
            concept_2 = row["concept_id_2"]
            rel_type = row.get("relationship_id")
            
            # Add directed edge from concept_1 to concept_2
            if graph.has_edge(concept_1, concept_2):
                # Append to existing relationships list
                graph[concept_1][concept_2]["relationships"].append(rel_type)
            else:
                # Create new edge with relationships list
                graph.add_edge(concept_1, concept_2, relationships=[rel_type])
        
        return graph

    def _build_index_mapping(
        self, node_embeddings: object
    ) -> dict[int, int]:
        """Map concept codes to embedding indices.

        Args:
            node_embeddings: Word2Vec model word vectors.

        Returns:
            Mapping from concept_id to index in embeddings.
        """
        return {int(key): i for i, key in enumerate(node_embeddings.index_to_key)}

    def _get_vector_iso(
        self,
        code: int,
        node_embeddings: object,
        index_mapping: dict,
        mean_vector: np.ndarray,
    ) -> np.ndarray:
        """Get embedding vector for code or mean vector if not found.

        Args:
            code: Concept ID.
            node_embeddings: Word2Vec model word vectors.
            index_mapping: Concept ID to embedding index mapping.
            mean_vector: Fallback vector to use if code not found.

        Returns:
            Embedding vector for the concept.
        """
        index = index_mapping.get(int(code))
        if index is not None:
            return node_embeddings.get_vector(index)
        else:
            print(f"Code {code} not found, returning mean vector.")
            return mean_vector

    def generate_embeddings(
        self, graph: nx.DiGraph
    ) -> tuple[np.ndarray, list]:
        """Generate node embeddings using Node2Vec.

        Args:
            graph: Directed graph of OMOP concepts and relationships.

        Returns:
            Tuple of (embedding_matrix, node_ids) where embedding_matrix
                is numpy array of embeddings and node_ids is the list of
                node IDs in order.
        """
        print(f"Graph created with {len(graph.nodes())} nodes and {len(graph.edges())} edges")
        
        if len(graph.nodes()) == 0:
            raise ValueError("Graph is empty, cannot generate embeddings")
        
        # Initialize and fit Node2Vec
        print(f"Initializing Node2Vec with embedding_dim={self.embedding_dim} walk_length={self.walk_length}, num_walks={self.num_walks}")

        node2vec = Node2Vec(
            graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=1, q=1, workers=4
        )
        
        # Train the model
        self.model = node2vec.fit(window=10, min_count=1, epochs=1)
        
        # Extract embeddings from trained model
        keys = list(graph.nodes())
        node_embeddings = self.model.wv
        
        # Build index mapping for efficient lookup
        index_mapping = self._build_index_mapping(node_embeddings)
        mean_vector = np.mean(node_embeddings.vectors, axis=0)
        
        # Create embedding vectors for all concepts
        print(f"Creating embedding vectors for {len(keys)} concepts...")
        vectors = [self._get_vector_iso(key, node_embeddings, index_mapping, mean_vector) for key in keys]
        
        # Stack into matrix
        embedding_matrix = np.vstack(vectors)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        
        return embedding_matrix, keys

class KeepEmbedding(BaseModel):
    """KEEP Embedding Framework


    Fine-tune Node2Vec embeddings using GloVe with graph regularization.

    Balances co-occurrence structure (GloVe) with graph (Node2Vec) via regularization
    to generate medical concept embeddings.

    Args:
        dataset: Dataset to train the model.
        graph: Directed graph of concepts and relationships.
        embedding_dim: Dimension of embeddings.
        walk_length: Length of random walks for Node2Vec.
        num_walks: Number of random walks per node.
        num_words: Size of vocabulary.
        lambda_reg: Regularization strength (default: 1.0).
        reg_norm: Norm type ('cosine' or numeric p-norm, default: None).
        log_scale: Apply log scaling to regularization distance
            (default: False).
        code_to_index: Optional mapping from concept codes to indices.
        device: Device to use ('cuda' or 'cpu', default: 'cpu').

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> from pyhealth.models import KeepEmbedding
        >>> dataset = SampleDataset(num_patients=100, num_visits=10, num_codes=50)
        >>> graph = n2v.create_graph()  # Build knowledge graph from concept and relationship tables
        >>> dataset = OMOPDataset(...)
        >>> # Build co-occurrence matrix from dataset
        >>> # Load co-occurrence matrix as GloveDatset Dataloader
        >>> model = KeepEmbedding(
        ...     dataset=None,
        ...     graph=graph,
        ...     embedding_dim=128,
        ...     walk_length=10,
        ...     num_walks=5,
        ...     num_words=50,
        ...     lambda_reg=0.5,
        ...     reg_norm='cosine',
        ...     log_scale=True,
        ...     device='cuda'
        ... )
        >>> # Use embeddings for with downstream PyHealth models
    """
    
    def __init__(
        self,
        dataset: SampleDataset,
        graph: nx.Graph,
        embedding_dim: int,
        walk_length: int,
        num_walks: int,
        num_words: int,
        lambda_reg: float = 1.0,
        reg_norm: str | float | None = None,
        log_scale: bool = False,
        code_to_index: dict | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize KEEP Embedding model."""
        super().__init__(dataset=dataset)
        
        self.embedding_dim = embedding_dim
        self.lambda_reg = lambda_reg
        self.reg_norm = reg_norm
        self.log_scale = log_scale
        self._device = device
        self.mode = "regression"  # Set mode for compatibility with BaseModel
        
        # Generate Node2Vec embeddings
        print(f"Initializing Node2Vec with embedding_dim={embedding_dim}...")
        self.n2v = N2V(
            embedding_dim=embedding_dim,
            walk_length=walk_length,
            num_walks=num_walks
        )
        
        embedding_matrix, node_ids = self.n2v.generate_embeddings(graph)
        print(f"Created embedding matrix with shape: {embedding_matrix.shape}")
        
        # Filter embeddings if code_to_index mapping is provided
        if code_to_index is not None:
            print(f"Filtering embeddings from {len(node_ids)} concepts to {num_words} vocabulary items...")
            # Create a mapping from node IDs to embeddings
            node_to_embedding = {node_id: embedding_matrix[i] for i, node_id in enumerate(node_ids)}
            
            # Build filtered embedding matrix for only codes in vocabulary
            filtered_vectors = []
            missing_count = 0
            for idx in range(num_words):
                # Find the concept code for this index
                code = next((code for code, code_idx in code_to_index.items() if code_idx == idx), None)
                if code is not None and code in node_to_embedding:
                    filtered_vectors.append(node_to_embedding[code])
                else:
                    missing_count += 1
                    # Use mean of all embeddings as fallback
                    filtered_vectors.append(np.mean(embedding_matrix, axis=0))
            
            if missing_count > 0:
                print(f"  {missing_count}/{num_words} codes not found in graph, using mean embedding as fallback")
            
            embedding_matrix = np.vstack(filtered_vectors)
            print(f"Filtered embedding matrix shape: {embedding_matrix.shape}")
        
        # Create learnable embedding and bias parameters
        self.embeddings_v = nn.Embedding(num_words, embedding_dim)
        self.embeddings_u = nn.Embedding(num_words, embedding_dim)
        self.biases_v = nn.Embedding(num_words, 1)
        self.biases_u = nn.Embedding(num_words, 1)
        
        # Initialize with Node2Vec embeddings
        embedding_tensor = torch.from_numpy(embedding_matrix).float()
        self.embeddings_v.weight.data.copy_(embedding_tensor)
        self.embeddings_u.weight.data.copy_(embedding_tensor)
        
        # Initialize biases to zero
        self.biases_v.weight.data.fill_(0)
        self.biases_u.weight.data.fill_(0)

        # Store initial embeddings for regularization
        self.register_buffer(
            "initial_embeddings",
            embedding_tensor.clone().to(device)
        )
        
        print(f"Initialized KEEP Embedding with {num_words} tokens")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"Regularization lambda: {lambda_reg}")
        print(f"Regularization norm: {reg_norm}")
        print(f"Log scaling: {log_scale}")
    
    def forward(
        self,
        i_indices: torch.Tensor | None = None,
        j_indices: torch.Tensor | None = None,
        counts: torch.Tensor | None = None,
        weights: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute GloVe loss with optional Node2Vec regularization.

        Args:
            i_indices: Token indices (batch_size,).
            j_indices: Context token indices (batch_size,).
            counts: Co-occurrence counts (batch_size,).
            weights: Weights for loss terms (batch_size,).
            **kwargs: Additional arguments for compatibility.

        Returns:
            Dictionary with keys: loss, logit, y_prob, y_true,
                reg_loss.
        """
        
        # If no GloVe inputs provided, return dummy output
        if i_indices is None or j_indices is None:
            dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return {
                "loss": dummy_loss,
                "logit": dummy_loss,
                "y_prob": dummy_loss,
                "y_true": dummy_loss,
            }
        
        # Move inputs to correct device
        i_indices = i_indices.to(self._device)
        j_indices = j_indices.to(self._device)
        counts = counts.to(self._device)
        weights = weights.to(self._device)
        
        # Get embeddings and biases
        embedding_i = self.embeddings_v(i_indices)  # (batch_size, embedding_dim)
        embedding_j = self.embeddings_u(j_indices)  # (batch_size, embedding_dim)
        bias_i = self.biases_v(i_indices).squeeze(-1)  # (batch_size,)
        bias_j = self.biases_u(j_indices).squeeze(-1)  # (batch_size,)
        
        # Compute GloVe loss: weighted squared difference
        # GloVe objective: w(i,j) * (u_i · v_j + b_i + b_j - log(X_ij))^2
        dot_product = torch.sum(embedding_i * embedding_j, dim=1)  # (batch_size,)
        glove_target = torch.log(counts + 1e-8)  # Avoid log(0)
        
        squared_diff = (dot_product + bias_i + bias_j - glove_target) ** 2
        glove_loss = torch.sum(weights * squared_diff)
        
        total_loss = glove_loss
        reg_loss = torch.tensor(0.0, device=self._device)
        
        # Add Node2Vec regularization if lambda > 0
        if self.lambda_reg > 0:
            # Average embeddings: (u_i + v_i) / 2 and (u_j + v_j) / 2
            u_plus_v_i = (embedding_i + self.embeddings_u(i_indices)) / 2
            u_plus_v_j = (embedding_j + self.embeddings_v(j_indices)) / 2
            
            # Get initial embeddings
            initial_i = self.initial_embeddings[i_indices]
            initial_j = self.initial_embeddings[j_indices]
            
            # Compute regularization distance based on norm type
            if self.reg_norm is None or self.reg_norm == "cosine":
                # Cosine distance: 1 - cosine_similarity
                reg_dist_i = 1 - F.cosine_similarity(u_plus_v_i, initial_i, dim=1)
                reg_dist_j = 1 - F.cosine_similarity(u_plus_v_j, initial_j, dim=1)
            else:
                # Lp norm distance
                p_norm = float(self.reg_norm)
                reg_dist_i = torch.norm(u_plus_v_i - initial_i, p=p_norm, dim=1)
                reg_dist_j = torch.norm(u_plus_v_j - initial_j, p=p_norm, dim=1)
            
            # Apply log scaling if enabled
            if self.log_scale:
                reg_dist_i = torch.log(reg_dist_i + 1e-8)
                reg_dist_j = torch.log(reg_dist_j + 1e-8)
            
            # Compute regularization loss
            reg_loss = self.lambda_reg * (torch.sum(reg_dist_i) + torch.sum(reg_dist_j))
            total_loss = glove_loss + reg_loss
        
        return {
            "loss": total_loss,
            "logit": glove_loss.detach(),  # Return GloVe component as logit for reference
            "y_prob": torch.zeros(i_indices.shape[0], device=self._device),  # Placeholder
            "y_true": torch.zeros(i_indices.shape[0], device=self._device),  # Placeholder
            "reg_loss": reg_loss.detach(),  # Return regularization loss for monitoring
        }
