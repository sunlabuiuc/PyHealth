# Authors: Yujia Li (yujia9@illinois.edu)
# Paper: Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction
# Paper link: https://arxiv.org/abs/2502.10689
# Description: Self-explaining hypergraph neural network using UniGIN message
# passing and Gumbel-Softmax phenotype extraction
# for personalized diagnosis prediction

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


def compute_indexed_average(
        source: torch.Tensor, index: torch.Tensor, total_count: int
) -> torch.Tensor:
    """Computes the mean of vectors grouped by index.

    This utility function is used during hypergraph message passing to aggregate
    node (diagnosis) embeddings into hyperedge (visit) embeddings, or vice versa,
    by calculating the average embedding for each group.

    Args:
        source (torch.Tensor): The source tensor containing embeddings to be averaged.
        index (torch.Tensor): A 1D tensor of indices indicating the group for each
            vector in the source.
        total_count (int): The total number of unique groups (e.g., number of visits
            or number of unique diagnosis codes).

    Returns:
        torch.Tensor: A tensor of shape (total_count, source_dimension) containing
            the averaged embeddings for each index.
    """
    # Initialize output and count tensors
    output_dim = source.shape[3]
    summed_val = torch.zeros(total_count, output_dim, device=source.device)
    element_counts = torch.zeros(total_count, 1, device=source.device)

    # Expand indices to match source shape
    expanded_idx = index.unsqueeze(1).expand_as(source)

    # Perform scatter addition
    summed_val.scatter_add_(0, expanded_idx, source)
    element_counts.scatter_add_(
        0, index.unsqueeze(1), torch.ones(index.shape, 1, device=source.device)
    )

    # Average the results while avoiding division by zero
    return summed_val / element_counts.clamp(min=1)


class HypergraphMessagePassing(nn.Module):
    """Implements a single layer of UniGIN hypergraph convolution.

    This class performs personalized disease representation learning by capturing
    higher-order disease interactions within and across visits using the
    UniGIN mechanism.
    """

    def __init__(self, input_dim: int, output_dim: int):
        """Initializes the UniGIN hypergraph convolution layer.

        Args:
            input_dim (int): Dimensionality of the input embeddings.
            output_dim (int): Dimensionality of the output embeddings.
        """
        super(HypergraphMessagePassing, self).__init__()
        self.linear_transform = nn.Linear(input_dim, output_dim)
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.activation = nn.LeakyReLU(0.2)

    def forward(
            self, node_features: torch.Tensor, incidence_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Performs a two-stage message passing on the patient hypergraph.

        The process first aggregates diagnosis embeddings into visit embeddings
        and then updates diagnosis embeddings using those visit representations.

        Args:
            node_features (torch.Tensor): Personalized diagnosis embeddings M_i.
            incidence_matrix (torch.Tensor): The incidence matrix P representing
                the patient hypergraph.

        Returns:
            torch.Tensor: Updated personalized diagnosis embeddings M_i^(z+1).
        """
        # incidence_matrix shape: [num_nodes, num_visits]
        node_indices, visit_indices = torch.where(incidence_matrix > 0)

        # Phase 1: Node to Hyperedge (Visit) aggregation
        visit_features = compute_indexed_average(
            node_features[node_indices], visit_indices, incidence_matrix.size(1)
        )

        # Phase 2: Hyperedge to Node aggregation
        # Aggregate visit features back to nodes that participate in them
        summed_visit_feats = torch.zeros_like(node_features)
        summed_visit_feats.index_add_(0, node_indices, visit_features[visit_indices])

        # UniGIN update rule: (1 + epsilon) * features + sum(neighbor_features)
        combined = (1 + self.epsilon) * node_features + summed_visit_feats
        return self.activation(self.linear_transform(combined))


class PhenotypeMiner(nn.Module):
    """
    Extracts a temporal phenotype using the Gumbel-Softmax trick.

    It generates a binary mask to identify a specific comorbidity pattern.
    """

    def __init__(self, feature_dim: int):
        """Initializes the phenotype extractor module.

        Args:
            feature_dim (int): Dimension of the diagnosis code embeddings.
        """
        super(PhenotypeMiner, self).__init__()
        self.mask_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid()
        )

    def forward(
            self, node_feats: torch.Tensor, visit_feats: torch.Tensor,
            tau: float = 1.0, hard_sampling: bool = True
    ) -> torch.Tensor:
        """Extracts a binary masking matrix to identify a temporal phenotype.

        Uses the Gumbel-Softmax trick to produce a discrete binary mask while
        allowing for backpropagation during training.

        Args:
            node_feats (torch.Tensor): Personalized diagnosis embeddings (M_i).
            visit_feats (torch.Tensor): Aggregated visit embeddings (V_j).
            tau (float): Temperature parameter for the Gumbel-Softmax distribution.
                Lower values lead to more discrete-like samples.
            hard_sampling (bool): If True, the output will be a discrete (one-hot)
                binary mask in the forward pass, while using the continuous
                approximation for the backward pass.

        Returns:
            torch.Tensor: The incidence matrix (Psi_k) of the extracted phenotype,
                formed by the element-wise product of the mask and the augmented
                incidence matrix.
        """
        num_nodes, num_visits = node_feats.size(0), visit_feats.size(0)

        # Concatenate node and visit features for all possible pairs
        # Resulting shape: [num_nodes, num_visits, feature_dim * 2]
        expanded_nodes = node_feats.unsqueeze(1).expand(-1, num_visits, -1)
        expanded_visits = visit_feats.unsqueeze(0).expand(num_nodes, -1, -1)
        pair_features = torch.cat([expanded_nodes, expanded_visits], dim=-1)

        # Generate probabilities for the Bernoulli distribution
        probs = self.mask_generator(pair_features).squeeze(-1)

        # Apply Gumbel-Softmax for differentiable discrete sampling
        logits = torch.stack(
            [torch.log(probs + 1e-9), torch.log(1 - probs + 1e-9)], dim=-1
        )
        mask = F.gumbel_softmax(logits, tau=tau, hard=hard_sampling)[:, :, 0]

        return mask


class SHy(BaseModel):
    """
    SHy: Self-Explaining Hypergraph Neural Network for Diagnosis Prediction.

    This model represents patients as hypergraphs and extracts temporal
    phenotypes as personalized explanations.
    """

    def __init__(
            self,
            dataset: SampleDataset,
            embedding_dim: int = 16,
            hgnn_dim: int = 16,
            hgnn_layers: int = 1,
            num_tp: int = 2,
            hidden_dim: int = 16,
            num_heads: int = 2,
            dropout: float = 0.0,
            **kwargs
    ):
        """Initializes the SHy model architecture.

        Args:
            dataset (SampleDataset): The PyHealth dataset containing vocabularies.
            embedding_dim (int): Dimension of the base hierarchical embeddings.
            hgnn_dim (int): Hidden dimension for the hypergraph message passing layers.
            hgnn_layers (int): Number of UniGIN layers (Z) to stack.
            num_tp (int): Number of temporal phenotypes (K) to extract.
            hidden_dim (int): Hidden dimension for prediction and phenotype extraction
                MLPs.
            num_heads (int): Number of attention heads for the self-attention
                mechanism.
            dropout (float): Dropout rate applied to neural network layers.
            **kwargs: Additional keyword arguments.
        """
        super(SHy, self).__init__(
            dataset=dataset,
            **kwargs
        )

        # Basic model parameters
        self.label_key = self.label_keys[0]
        self.feature_key = self.feature_keys[0]
        self.num_phenotypes = num_tp
        self.embed_size = embedding_dim

        processor = dataset.input_processors[self.feature_key]
        self.vocab_size = processor.vocab_size()
        self.output_size = self.get_output_size()
        # Core components
        self.code_embeddings = nn.Embedding(self.vocab_size, embedding_dim)

        # GNN layers for personalized embedding learning
        self.gnn_stack = nn.ModuleList([
            HypergraphMessagePassing(
                embedding_dim if i == 0 else hgnn_dim, hgnn_dim
            ) for i in range(hgnn_layers)
        ])

        # Phenotype extraction module
        self.extractors = nn.ModuleList([
            PhenotypeMiner(hgnn_dim) for _ in range(num_tp)
        ])

        # Temporal modeling per phenotype
        self.phenotype_gru = nn.GRU(
            hgnn_dim, hidden_dim, batch_first=True, bidirectional=False
        )

        # Self-attention across phenotypes
        self.cross_phenotype_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        # Final diagnosis prediction head
        self.output_layer = nn.Linear(
            hidden_dim, self.output_size
        )

    def forward(
            self,
            diagnoses_hist: List[List[List[str]]],
            diagnoses: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass implementing the SHy architecture.

        Steps:
        1. Build incidence matrix P for each patient.
        2. Personalized embedding learning via UniGIN message passing.
        3. Structure augmentation (False Negative handling).
        4. Differentiable phenotype extraction using Gumbel-Softmax.
        5. Temporal modeling and final weighted diagnosis prediction.

        Args:
            diagnoses_hist (List[List[List[str]]]): A nested list of ICD-9 diagnosis
                codes representing the longitudinal record for each patient.
            diagnoses (Optional[torch.Tensor]): Ground truth multi-hot labels for
                the next visit. Defaults to None.
            **kwargs: Additional context such as timestamps or masking indices.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing prediction results,
                extracted phenotypes, and calculated loss components
        """

        # Get patient features and device info
        batch_size = len(diagnoses_hist)
        device = self.code_embeddings.weight.device

        # 1. INITIALIZE HIERARCHICAL EMBEDDINGS
        # Maps ICD-9 codes to their initial personalized vector space
        # (Simplified: in practice, this involves concatenating ancestor embeddings)

        # Prepare to store batch results
        batch_logits = []

        # Process each patient individually (or via advanced batching if optimized)
        for i in range(batch_size):
            patient_visits = diagnoses_hist[i] # data format: [[d1, d2], [d1, d4]]
            num_visits = len(patient_visits)

            # Map codes to their indices
            visit_indices = [
                self.feature_tokenizers["diagnoses_hist"].convert_tokens_to_ids(v)
                for v in patient_visits
            ]

            # 2. CONSTRUCT HYPERGRAPH INCIDENCE MATRIX (P)
            # Diseases are nodes, hospital visits are hyperedges
            unique_codes = sorted(list(set([c for v in visit_indices for c in v])))
            code_to_idx = {code: idx for idx, code in enumerate(unique_codes)}
            num_nodes = len(unique_codes)

            # P matrix shape: [num_nodes, num_visits]
            incidence_p = torch.zeros((num_nodes, num_visits), device=device)
            for v_idx, visit in enumerate(visit_indices):
                for code in visit:
                    incidence_p[code_to_idx[code], v_idx] = 1.0

            # Initialize node features for this specific patient
            node_features = self.code_embeddings(
                torch.tensor(unique_codes, device=device)
            )

            # 3. PERSONALIZED MESSAGE PASSING (UniGIN)
            # Captures higher-order disease interactions within and across visits
            for layer in self.gnn_stack:
                node_features = layer(node_features, incidence_p)

            # 4. STRUCTURE AUGMENTATION (Similarity-based FN Handling)
            # Add potential diagnosis-visit pairs based on embedding similarity
            # Compute visit features first (Equation 2 in paper)
            node_indices, visit_indices_p = torch.where(incidence_p > 0)
            visit_features = compute_indexed_average(
                node_features[node_indices], visit_indices_p, num_visits
            )

            # Augment P with extra connections (delta P) based on top-k similarity
            # (Logic: p ratio defines how many false negatives to recover)
            p_tilde = incidence_p  # Simplified: p_tilde = P + Delta_P

            # 5. TEMPORAL PHENOTYPE EXTRACTION
            # Extract K distinct subgraphs using Gumbel-Softmax masks
            phenotype_embeddings = []
            for extractor in self.extractors:
                # Mask matrix Gamma_k: [num_nodes, num_visits]
                mask = extractor(node_features, visit_features)
                sub_hypergraph = p_tilde * mask

                # Project phenotype into a vector space (Equation 9)
                # Weighted sum of node features per visit in this phenotype
                # [num_visits, dim]
                visit_reps = torch.matmul(sub_hypergraph.t(), node_features)

                # Temporal modeling per phenotype (GRU + Attention)
                # unsqueeze(0) for batch processing in GRU
                gru_out, _ = self.phenotype_gru(visit_reps.unsqueeze(0))
                # Final attention pooling to get single phenotype embedding U_k
                # (Simplified: using last hidden state or pooling)
                u_k = torch.mean(gru_out, dim=1)  # [1, hidden_dim]
                phenotype_embeddings.append(u_k)

            # Combine all K phenotype embeddings [1, num_tp, hidden_dim]
            all_u = torch.cat(phenotype_embeddings, dim=0).unsqueeze(0)

            # 6. CROSS-PHENOTYPE ATTENTION
            # Determine which phenotypes are more important for the prediction
            attn_output, _ = self.cross_phenotype_attn(all_u, all_u, all_u)
            final_patient_rep = torch.mean(attn_output, dim=1)  # [1, hidden_dim]

            # Final output layer
            patient_logits = self.output_layer(final_patient_rep)
            batch_logits.append(patient_logits)

        # Concatenate results into a batch tensor
        logits = torch.cat(batch_logits, dim=0)
        y_prob = torch.sigmoid(logits)

        # Prepare output dictionary
        output = {"logits": logits, "y_prob": y_prob}

        # If labels are provided, calculate multi-label loss (Binary Cross Entropy)
        if diagnoses is not None:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, diagnoses)
            # Note: In a full implementation, you would also add L_fidelity,
            # L_distinct, and L_alpha to the total loss.

        return output
