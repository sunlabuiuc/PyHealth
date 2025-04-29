"""
Jack Fan (jackfan2)
ConceptExplainerModel: Concept-based explanations for EHR (Mincu et al. 2020)
Paper: https://arxiv.org/pdf/2012.02308
Description:
This model implements a concept-based explainable prediction system for clinical data, based on Mincu et al.'s approach. It processes patient features through a neural network that identifies and weights relevant clinical concepts (e.g., diagnoses, medications) using multi-head attention. The architecture generates both predictions and explanations by: (1) projecting raw inputs to a concept space, (2) modeling concept uncertainty through probabilistic embeddings, (3) analyzing concept relationships via attention mechanisms, and (4) producing interpretable importance scores for each clinical concept. Designed for EHR data, the model outputs predictions with confidence intervals while highlighting the most influential medical concepts and their interactions, supporting transparent decision-making in healthcare applications.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from torch.distributions import Normal

class ConceptExplainerModel(nn.Module):
    def __init__(
        self,
        num_concepts: int = 100,
        input_dim: int = 64,
        output_dim: int = 1,
        concept_dim: int = 64,
        num_concepts_per_sample: int = 20,
        num_attention_heads: int = 4,
        concept_dropout: float = 0.1,
    ):
        super().__init__()
        
        self.concept_dim = concept_dim
        self.num_heads = num_attention_heads
        self.head_dim = concept_dim // num_attention_heads
        
        # 1. Feature projection
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, concept_dim),
            nn.LayerNorm(concept_dim),
            nn.GELU()
        )
        
        # 2. Concept embedding with uncertainty
        self.concept_embeddings = nn.Embedding(num_concepts, concept_dim)
        self.concept_logvars = nn.Embedding(num_concepts, concept_dim)
        
        # 3. Attention components
        self.query = nn.Linear(concept_dim, concept_dim)
        self.key = nn.Linear(concept_dim, concept_dim)
        self.value = nn.Linear(concept_dim, concept_dim)
        self.attn_dropout = nn.Dropout(concept_dropout)
        
        # 4. Concept importance
        self.importance_scorer = nn.Sequential(
            nn.Linear(concept_dim, concept_dim//2),
            nn.ReLU(),
            nn.Linear(concept_dim//2, 1)
        )
        
        # 5. Prediction
        self.label_predictor = nn.Sequential(
            nn.Linear(concept_dim, concept_dim),
            nn.LayerNorm(concept_dim),
            nn.GELU(),
            nn.Linear(concept_dim, output_dim)
        )
        
        # 6. Auxiliary
        self.concept_predictor = nn.Linear(concept_dim, num_concepts)

    def forward(self, x: torch.Tensor, concept_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # 1. Project features
        projected = self.feature_projection(x)  # [B, D]
        
        # 2. Get concept samples
        concept_embs = self.concept_embeddings(concept_ids)  # [B, K, D]
        concept_vars = torch.exp(self.concept_logvars(concept_ids))
        concept_samples = concept_embs + torch.randn_like(concept_vars) * concept_vars.sqrt()
        
        # 3. Manual multi-head attention
        q = self.query(projected).view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D/H]
        k = self.key(concept_samples).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, D/H]
        v = self.value(concept_samples).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, K, D/H]
        
        attn_weights = F.softmax((q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_output = (attn_weights @ v).transpose(1, 2).reshape(batch_size, 1, -1)  # [B, 1, D]
        
        # 4. Importance scores
        importance = F.softmax(self.importance_scorer(concept_samples), dim=1)
        
        # 5. Prediction
        logits = self.label_predictor(attn_output.squeeze(1))
        
        return {
            "y_prob": torch.sigmoid(logits),
            "importance_scores": importance,
            "concept_samples": concept_samples,
            "attention_weights": attn_weights.mean(dim=1)
        }

# Test Case
if __name__ == "__main__":
    # Config
    BATCH_SIZE = 32
    INPUT_DIM = 64
    NUM_CONCEPTS = 100
    CONCEPTS_PER_SAMPLE = 20
    
    model = ConceptExplainerModel(
        num_concepts=NUM_CONCEPTS,
        input_dim=INPUT_DIM,
        concept_dim=64,
        num_attention_heads=4,
        num_concepts_per_sample=CONCEPTS_PER_SAMPLE
    )
    
    # Generate properly sized inputs
    x = torch.randn(BATCH_SIZE, INPUT_DIM)
    concept_ids = torch.randint(0, NUM_CONCEPTS, (BATCH_SIZE, CONCEPTS_PER_SAMPLE))
    
    # Forward pass
    outputs = model(x, concept_ids)
    
    print("\nSuccessful Execution:")
    print(f"Input: {x.shape}")
    print(f"Output shapes:")
    print(f"• Predictions: {outputs['y_prob'].shape}")
    print(f"• Importance: {outputs['importance_scores'].shape}")
    print(f"• Attention: {outputs['attention_weights'].shape}")
    
    # Show attention
    print("\nSample attention (first 5 concepts):")
    print(outputs["attention_weights"][0, :5].tolist())