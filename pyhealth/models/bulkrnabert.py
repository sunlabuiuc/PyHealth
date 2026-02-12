"""BulkRNABert Model for Cancer Prognosis

This module provides a PyHealth-compatible implementation of BulkRNABert for
cancer prognosis tasks including:
1. Cancer type classification (33 TCGA cancer types)
2. Survival prediction using Cox proportional hazards

Based on: Gélard et al. (2025) "BulkRNABert: Cancer prognosis from bulk RNA-seq
based language models"

Paper: https://www.biorxiv.org/content/10.1101/2024.06.13.598798
Model: https://huggingface.co/InstaDeepAI/BulkRNABert

Author: Luis E. Fernandez de la Vara
Course: CS 598 DLH - Deep Learning for Healthcare (Fall 2025)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


# =============================================================================
# Configuration
# =============================================================================

BULKRNABERT_CONFIG = {
    "model_name": "InstaDeepAI/BulkRNABert",
    "hidden_dim": 768,
    "num_layers": 12,
    "num_attention_heads": 12,
    "max_genes": 19264,
}

TCGA_CANCER_TYPES = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]


# =============================================================================
# Model Layers
# =============================================================================

class BulkRNABertLayer(nn.Module):
    """BulkRNABert encoder layer for gene expression data.

    This layer transforms gene expression vectors into fixed-size embeddings
    using a transformer-based architecture similar to the original BulkRNABert
    model.

    Args:
        input_dim: Number of input genes (default: 19264 for TCGA)
        hidden_dim: Hidden dimension size (default: 768)
        num_layers: Number of transformer layers (default: 6)
        num_heads: Number of attention heads (default: 12)
        dropout: Dropout probability (default: 0.1)

    Example:
        >>> layer = BulkRNABertLayer(input_dim=19264, hidden_dim=768)
        >>> x = torch.randn(32, 19264)  # batch of gene expression
        >>> output = layer(x)
        >>> print(output.shape)  # (32, 768)
    """

    def __init__(
        self,
        input_dim: int = BULKRNABERT_CONFIG["max_genes"],
        hidden_dim: int = BULKRNABERT_CONFIG["hidden_dim"],
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Gene embedding projection
        self.gene_projection = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Pooler for [CLS]-like output
        self.pooler = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Gene expression tensor of shape (batch_size, num_genes)
            mask: Optional attention mask

        Returns:
            Pooled output tensor of shape (batch_size, hidden_dim)
        """
        # Project to hidden dim
        x = self.gene_projection(x)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Add sequence dimension for transformer (batch, seq_len=1, hidden)
        x = x.unsqueeze(1)

        # Apply transformer
        x = self.transformer(x, mask=mask)

        # Pool and return
        pooled = self.pooler(x.squeeze(1))

        return pooled


# =============================================================================
# Main Model
# =============================================================================

class BulkRNABert(BaseModel):
    """BulkRNABert model for cancer prognosis from bulk RNA-seq data.

    BulkRNABert is a BERT-style transformer pre-trained on bulk RNA-seq data
    using masked expression modeling. This implementation provides fine-tuning
    capabilities for cancer type classification and survival prediction.

    Paper: Gélard et al. (2025) "BulkRNABert: Cancer prognosis from bulk RNA-seq
    based language models"

    Note:
        This is a simplified implementation that mirrors the BulkRNABert
        architecture. For production use with pre-trained weights, use the
        HuggingFace implementation from InstaDeepAI/BulkRNABert.

    Args:
        dataset: A PyHealth SampleDataset object.
        feature_keys: List of feature keys to use from the dataset.
        label_key: The key for the label in the dataset.
        mode: Task mode - "binary", "multiclass", "multilabel", or "regression".
        input_dim: Number of input genes (default: 19264).
        hidden_dim: Hidden dimension size (default: 768).
        num_layers: Number of transformer layers (default: 6).
        num_heads: Number of attention heads (default: 12).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> from pyhealth.datasets import SampleDataset
        >>> dataset = SampleDataset(...)  # Your RNA-seq dataset
        >>> model = BulkRNABert(
        ...     dataset=dataset,
        ...     feature_keys=["gene_expression"],
        ...     label_key="cancer_type",
        ...     mode="multiclass",
        ... )
        >>> # Training
        >>> outputs = model(**batch)
        >>> loss = outputs["loss"]
    """

    def __init__(
        self,
        dataset: SampleDataset = None,
        feature_keys: List[str] = None,
        label_key: str = None,
        mode: str = "multiclass",
        input_dim: int = BULKRNABERT_CONFIG["max_genes"],
        hidden_dim: int = BULKRNABERT_CONFIG["hidden_dim"],
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(dataset)

        self.feature_keys = feature_keys or []
        self.label_key = label_key
        self.mode = mode
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = BulkRNABertLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Output layer - determine size based on mode and dataset
        if dataset is not None and label_key is not None:
            output_size = self.get_output_size()
        else:
            # Default to 33 TCGA cancer types for classification
            output_size = len(TCGA_CANCER_TYPES) if mode == "multiclass" else 1

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size),
        )

    def forward(
        self,
        gene_expression: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            gene_expression: Gene expression tensor (batch, num_genes).
                Can also be passed via feature_keys from dataset batch.
            labels: Optional labels for computing loss.
            **kwargs: Additional batch inputs from dataset.

        Returns:
            Dictionary containing:
                - logits: Output logits
                - y_prob: Predicted probabilities
                - loss: Loss value (if labels provided)
                - embeddings: Hidden representations
        """
        # Get gene expression from kwargs if not directly provided
        if gene_expression is None:
            if self.feature_keys:
                gene_expression = kwargs.get(self.feature_keys[0])
            else:
                raise ValueError("gene_expression must be provided")

        # Get labels from kwargs if not directly provided
        if labels is None and self.label_key:
            labels = kwargs.get(self.label_key)

        # Encode
        embeddings = self.encoder(gene_expression)

        # Get logits
        logits = self.output_layer(embeddings)

        # Prepare output
        outputs = {
            "logits": logits,
            "embeddings": embeddings,
        }

        # Compute probabilities
        outputs["y_prob"] = self.prepare_y_prob(logits)

        # Compute loss if labels provided
        if labels is not None:
            loss_fn = self.get_loss_function()
            if self.mode == "multiclass":
                loss = loss_fn(logits, labels.long())
            else:
                loss = loss_fn(logits, labels.float())
            outputs["loss"] = loss

        return outputs

    def prepare_y_prob(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to probabilities based on task mode."""
        if self.mode == "binary":
            return torch.sigmoid(logits)
        elif self.mode == "multiclass":
            return F.softmax(logits, dim=-1)
        elif self.mode == "multilabel":
            return torch.sigmoid(logits)
        else:  # regression
            return logits

    def get_loss_function(self):
        """Get the appropriate loss function for the task mode."""
        if self.mode == "binary":
            return F.binary_cross_entropy_with_logits
        elif self.mode == "multiclass":
            return F.cross_entropy
        elif self.mode == "multilabel":
            return F.binary_cross_entropy_with_logits
        else:  # regression
            return F.mse_loss


class BulkRNABertForSurvival(BaseModel):
    """BulkRNABert model for survival prediction using Cox proportional hazards.

    This model predicts patient risk scores from bulk RNA-seq data, which can be
    used for survival analysis with the concordance index (C-index) metric.

    Args:
        dataset: A PyHealth SampleDataset object.
        feature_keys: List of feature keys to use from the dataset.
        time_key: Key for survival time in the dataset.
        event_key: Key for event indicator (1=event, 0=censored).
        input_dim: Number of input genes (default: 19264).
        hidden_dim: Hidden dimension size (default: 768).
        num_layers: Number of transformer layers (default: 6).
        num_heads: Number of attention heads (default: 12).
        dropout: Dropout probability (default: 0.1).

    Example:
        >>> model = BulkRNABertForSurvival(
        ...     dataset=dataset,
        ...     feature_keys=["gene_expression"],
        ...     time_key="survival_time",
        ...     event_key="event",
        ... )
        >>> outputs = model(**batch)
        >>> risk_scores = outputs["risk_scores"]
    """

    def __init__(
        self,
        dataset: SampleDataset = None,
        feature_keys: List[str] = None,
        time_key: str = "survival_time",
        event_key: str = "event",
        input_dim: int = BULKRNABERT_CONFIG["max_genes"],
        hidden_dim: int = BULKRNABERT_CONFIG["hidden_dim"],
        num_layers: int = 6,
        num_heads: int = 12,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(dataset)

        self.feature_keys = feature_keys or []
        self.time_key = time_key
        self.event_key = event_key
        self.mode = "regression"  # Cox outputs are continuous risk scores

        # Encoder
        self.encoder = BulkRNABertLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        gene_expression: Optional[torch.Tensor] = None,
        survival_time: Optional[torch.Tensor] = None,
        event: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for survival prediction.

        Args:
            gene_expression: Gene expression tensor (batch, num_genes).
            survival_time: Observed survival times.
            event: Event indicators (1=event, 0=censored).
            **kwargs: Additional batch inputs.

        Returns:
            Dictionary with risk_scores, embeddings, and optional loss.
        """
        # Get inputs from kwargs if not directly provided
        if gene_expression is None:
            if self.feature_keys:
                gene_expression = kwargs.get(self.feature_keys[0])
            else:
                raise ValueError("gene_expression must be provided")

        if survival_time is None:
            survival_time = kwargs.get(self.time_key)
        if event is None:
            event = kwargs.get(self.event_key)

        # Encode
        embeddings = self.encoder(gene_expression)

        # Get risk scores
        risk_scores = self.risk_head(embeddings)

        outputs = {
            "risk_scores": risk_scores,
            "y_prob": risk_scores,  # For compatibility
            "embeddings": embeddings,
        }

        # Compute Cox loss if survival data provided
        if survival_time is not None and event is not None:
            loss = self._cox_loss(risk_scores, survival_time, event)
            outputs["loss"] = loss

        return outputs

    def _cox_loss(
        self,
        risk_scores: torch.Tensor,
        survival_times: torch.Tensor,
        events: torch.Tensor,
        eps: float = 1e-7,
    ) -> torch.Tensor:
        """Compute Cox proportional hazards partial likelihood loss.

        Args:
            risk_scores: Predicted risk scores (batch, 1)
            survival_times: Observed survival times (batch,)
            events: Event indicators (batch,)
            eps: Small constant for numerical stability

        Returns:
            Negative partial log-likelihood loss
        """
        risk_scores = risk_scores.squeeze()

        # Sort by survival time (descending)
        sorted_indices = torch.argsort(survival_times, descending=True)
        sorted_risk = risk_scores[sorted_indices]
        sorted_events = events[sorted_indices]

        # Log cumulative sum of exp(risk) for risk set
        log_cumsum_risk = torch.logcumsumexp(sorted_risk, dim=0)

        # Cox partial likelihood for uncensored observations
        uncensored_likelihood = sorted_risk - log_cumsum_risk

        # Only count uncensored (event=1) observations
        censored_mask = sorted_events.bool()
        num_events = censored_mask.sum()

        if num_events == 0:
            return torch.tensor(0.0, device=risk_scores.device)

        loss = -uncensored_likelihood[censored_mask].sum() / (num_events + eps)

        return loss


# =============================================================================
# Utility Functions
# =============================================================================

def compute_c_index(
    risk_scores: torch.Tensor,
    survival_times: torch.Tensor,
    events: torch.Tensor,
) -> float:
    """Compute concordance index for survival predictions.

    The C-index measures the model's ability to correctly rank patients
    by their predicted risk of experiencing the event.

    Args:
        risk_scores: Predicted risk scores
        survival_times: Observed survival times
        events: Event indicators (1=event, 0=censored)

    Returns:
        C-index value (0.5=random, 1.0=perfect ranking)

    Example:
        >>> c_index = compute_c_index(outputs["risk_scores"], times, events)
        >>> print(f"C-index: {c_index:.4f}")
    """
    risk_scores = risk_scores.detach().cpu().numpy().flatten()
    survival_times = survival_times.detach().cpu().numpy().flatten()
    events = events.detach().cpu().numpy().flatten()

    concordant = 0
    discordant = 0
    tied = 0
    comparable = 0

    n = len(survival_times)
    for i in range(n):
        for j in range(i + 1, n):
            if survival_times[i] == survival_times[j]:
                continue
            if survival_times[i] < survival_times[j]:
                if events[i] == 0:
                    continue
                shorter, longer = i, j
            else:
                if events[j] == 0:
                    continue
                shorter, longer = j, i

            comparable += 1
            if risk_scores[shorter] > risk_scores[longer]:
                concordant += 1
            elif risk_scores[shorter] < risk_scores[longer]:
                discordant += 1
            else:
                tied += 1

    if comparable == 0:
        return 0.5

    return (concordant + 0.5 * tied) / comparable
