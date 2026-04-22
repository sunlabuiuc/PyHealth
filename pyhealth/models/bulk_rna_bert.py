"""BulkRNABert model for PyHealth.

Paper: Gélard et al., "BulkRNABert: Cancer prognosis from bulk RNA-seq
based language models", bioRxiv 2024.

The model pre-trains a BERT-style encoder on bulk RNA-seq data via
Masked Language Modeling (MLM), then fine-tunes lightweight MLP heads
for cancer type classification or survival prediction.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import SampleDataset
from .base_model import BaseModel


class BulkRNABert(BaseModel):
    """BulkRNABert: Transformer encoder for bulk RNA-seq cancer prognosis.

    This model encodes a tokenized bulk RNA-seq sample (integer bin IDs per gene)
    into a fixed-size embedding via a BERT-style transformer encoder with gene
    embeddings.

    Gene expression is permutation-invariant, so standard positional
    encodings are replaced by learned gene embeddings (one per gene position
    in the fixed gene panel).

    Args:
        dataset: PyHealth SampleDataset with ``token_ids`` input and
            either ``cancer_type`` or ``survival_time`` / ``event`` outputs.
        n_genes: Number of genes in the input panel. Defaults to 19042.
        n_bins: Number of expression bins (vocabulary size). Defaults to 64.
        embedding_dim: Transformer embedding dimension. Defaults to 256.
        n_layers: Number of transformer encoder layers. Defaults to 4.
        n_heads: Number of attention heads. Defaults to 8.
        ffn_dim: Feed-forward network hidden dimension. Defaults to 512.
        dropout: Dropout rate. Defaults to 0.1.
        mlp_hidden: Hidden layer sizes for the task MLP head.
            Defaults to (256, 128) for classification.
        mode: Task mode. One of ``"classification"`` or ``"survival"``.
            Defaults to ``"classification"``.
        n_classes: Number of output classes for classification. Required
            when ``mode="classification"``.
        use_ia3: Whether to apply IA3 parameter-efficient fine-tuning.
            Adds learned rescaling vectors to attention keys, values, and
            feed-forward activations. Defaults to False.

    Attributes:
        gene_embedding: Learned embedding matrix of shape
            ``(n_genes, embedding_dim)``.
        expr_embedding: Embedding for discretized expression bins of shape
            ``(n_bins, embedding_dim)``.
        encoder: Transformer encoder stack.
        task_head: MLP head for the downstream task.

    Examples:
        >>> import torch
        >>> from pyhealth.models import BulkRNABert
        >>> model = BulkRNABert(
        ...     dataset=None,
        ...     n_genes=100,
        ...     n_bins=64,
        ...     embedding_dim=64,
        ...     n_layers=2,
        ...     n_heads=4,
        ...     mode="classification",
        ...     n_classes=5,
        ... )
        >>> token_ids = torch.randint(0, 64, (2, 100))
        >>> out = model(token_ids=token_ids)
        >>> print(out["logit"].shape)
        torch.Size([2, 5])
    """

    def __init__(
        self,
        dataset: Optional[SampleDataset],
        n_genes: int = 19042,
        n_bins: int = 64,
        embedding_dim: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        mlp_hidden: Tuple[int, ...] = (256, 128),
        mode: str = "classification",
        n_classes: int = 33,
        use_ia3: bool = False,
    ) -> None:
        super().__init__(dataset)

        self.n_genes = n_genes
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.task_mode = mode
        self.n_classes = n_classes
        self.use_ia3 = use_ia3

        self.expr_embedding = nn.Embedding(n_bins, embedding_dim)

        self.gene_embedding = nn.Embedding(n_genes, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if use_ia3:
            head_dim = embedding_dim // n_heads
            self.ia3_lk = nn.ParameterList([
                nn.Parameter(torch.ones(head_dim * n_heads))
                for _ in range(n_layers)
            ])
            self.ia3_lv = nn.ParameterList([
                nn.Parameter(torch.ones(head_dim * n_heads))
                for _ in range(n_layers)
            ])
            self.ia3_lff = nn.ParameterList([
                nn.Parameter(torch.ones(ffn_dim))
                for _ in range(n_layers)
            ])

        self.mlm_head = nn.Linear(embedding_dim, n_bins)

        # Task-specific MLP head
        if mode == "classification":
            self.task_head = _build_mlp(
                embedding_dim, list(mlp_hidden), n_classes, dropout
            )
        elif mode == "survival":
            survival_hidden = (512, 256)
            self.task_head = _build_mlp(
                embedding_dim, list(survival_hidden), 1, dropout
            )
        else:
            raise ValueError(
                f"mode must be 'classification' or 'survival', got {mode}"
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenized RNA-seq into mean-pooled embeddings.
        Args:
            token_ids: Tokenized RNA-seq tensor of shape (batch_size, sequence_length).
        Returns:
            Mean-pooled embeddings of shape (batch_size, embedding_dim).
        """
        bsz, seq_len = token_ids.shape
        device = token_ids.device

        gene_idx = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )

        x = self.expr_embedding(token_ids) + self.gene_embedding(gene_idx)
        x = self.encoder(x) 
        return x.mean(dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        cancer_type: Optional[torch.Tensor] = None,
        survival_time: Optional[torch.Tensor] = None,
        event: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for downstream task prediction.
        Args:
            token_ids: Tokenized RNA-seq tensor of shape (batch_size, sequence_length).
            cancer_type: Optional class labels of shape (batch,) for classification loss computation.
            survival_time: Optional survival times of shape (batch,).
            event: Optional event indicators of shape (batch,) where 1 = death, 0 = censored.
        Returns:
            A dictionary with the following keys:
                - logit: Raw output logits.
                - y_prob: Probabilities (classification) or risk scores (survival).
                - loss: Scalar loss tensor (if labels provided).
                - y_true: True labels (if labels provided).
        """

        if isinstance(token_ids, tuple):
            token_ids = token_ids[0]

        token_ids = token_ids.long()
        z = self.encode(token_ids)
        logit = self.task_head(z)

        results: Dict[str, torch.Tensor] = {}

        if self.task_mode == "classification":
            y_prob = torch.softmax(logit, dim=-1)
            results["logit"] = logit
            results["y_prob"] = y_prob

            if cancer_type is not None:
                if isinstance(cancer_type, tuple):
                    cancer_type = cancer_type[0]
                results["loss"] = F.cross_entropy(logit, cancer_type.long())
                results["y_true"] = cancer_type

        elif self.task_mode == "survival":
            log_risk = logit.squeeze(-1)
            results["logit"] = log_risk
            results["y_prob"] = log_risk

            if event is not None and survival_time is not None:
                if isinstance(event, tuple):
                    event = event[0]
                if isinstance(survival_time, tuple):
                    survival_time = survival_time[0]
                results["loss"] = cox_partial_likelihood_loss(
                    log_risk, survival_time.float(), event.float()
                )
                results["y_true"] = event

        return results

    def forward_mlm(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for MLM pre-training.
        Args:
            token_ids: Tokenized RNA-seq tensor of shape (batch_size, sequence_length).
            mask: Boolean mask of shape (batch_size, sequence_length) where True indicates a masked position.
            targets: Original token IDs of shape (batch_size, sequence_length).
        Returns:
            Scalar MLM cross-entropy loss over masked positions.
        """

        bsz, seq_len = token_ids.shape
        device = token_ids.device

        gene_idx = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )
        x = self.expr_embedding(token_ids) + self.gene_embedding(gene_idx)
        x = self.encoder(x)

        logits = self.mlm_head(x) 
        loss = F.cross_entropy(
            logits[mask],
            targets[mask].long(),
        )
        return loss


def _build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    dropout: float,
) -> nn.Sequential:
    """Build a MLP with SELU activations, dropout, and layer norm.
    Args:
        input_dim: Input dimension.
        hidden_dims: List of hidden dimensions.
        output_dim: Output dimension.
        dropout: Dropout rate.
    Returns:
        A sequential MLP module.
    """

    layers: List[nn.Module] = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [
            nn.Linear(in_dim, h),
            nn.LayerNorm(h),
            nn.SELU(),
            nn.Dropout(dropout),
        ]
        in_dim = h
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


def cox_partial_likelihood_loss(
    log_risk: torch.Tensor,
    survival_time: torch.Tensor,
    event: torch.Tensor,
) -> torch.Tensor:
    """Negative Cox partial log-likelihood loss. Implements the Breslow 
        approximation for ties.
    Note:
        Returns zero loss if no events are observed in the batch.
    Args:
        log_risk: Predicted log-risk scores of shape (batch,).
        survival_time: Observed survival times of shape (batch,).
        event: Event indicators of shape (batch,) where 1 = death, 0 = censored.
    Returns:
        Scalar negative partial log-likelihood loss.
    """

    order = torch.argsort(survival_time, descending=True)
    log_risk = log_risk[order]
    event = event[order]

    log_cumsum_risk = torch.logcumsumexp(log_risk, dim=0)

    observed = event.bool()
    if observed.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=log_risk.device)

    loss = -(log_risk[observed] - log_cumsum_risk[observed]).mean()
    return loss