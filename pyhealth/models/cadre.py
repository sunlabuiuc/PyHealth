from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class CADRE(BaseModel):
    """CADRE model for multilabel drug-response prediction.

    This is a clean PyHealth-style implementation inspired by the original
    CADRE architecture. It preserves the core ideas:

    1. Embed input gene indices.
    2. Optionally compute drug-contextual attention over genes.
    3. Build a drug-specific hidden representation for each sample.
    4. Decode to per-drug logits using learned drug embeddings.

    Expected input
    --------------
    This model expects one feature key containing integer gene indices,
    padded with 0 if needed. The tensor shape should be:

        gene_idx: [batch_size, num_selected_genes]

    It expects one multilabel target key containing per-drug binary labels:

        label: [batch_size, num_drugs]

    Optionally, a mask tensor can be provided to ignore missing labels:

        label_mask: [batch_size, num_drugs]

    Notes
    -----
    - This implementation is intentionally cleaner than the original training
      code and is designed to fit PyHealth's BaseModel API.
    - The first version focuses on expression/indexed gene input.
    - If contextual attention is enabled, the model uses drug IDs as context
      tokens, analogous to the original CADRE pathway/drug-context mechanism.

    Args:
        dataset: PyHealth sample dataset.
        feature_key: Name of the feature containing integer gene indices.
        label_key: Name of the multilabel drug-response target.
        mask_key: Optional mask key for missing labels.
        num_genes: Number of gene IDs excluding padding. Padding index is 0.
        num_drugs: Number of drug outputs.
        embedding_dim: Dimension of gene embeddings.
        hidden_dim: Hidden dimension used for drug decoding.
        attention_size: Intermediate size for attention scoring.
        attention_head: Number of attention heads.
        dropout: Dropout probability.
        use_attention: Whether to use attention over gene embeddings.
        use_cntx_attn: Whether to use drug-contextual attention.
        init_gene_emb: Optional pretrained gene embedding tensor of shape
            [num_genes + 1, embedding_dim].
        use_relu: Whether to apply ReLU before dropout on the encoded
            drug-specific representation.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str,
        label_key: str,
        num_genes: int,
        num_drugs: int,
        mask_key: Optional[str] = None,
        embedding_dim: int = 200,
        hidden_dim: int = 200,
        attention_size: int = 128,
        attention_head: int = 8,
        dropout: float = 0.6,
        use_attention: bool = True,
        use_cntx_attn: bool = True,
        init_gene_emb: Optional[torch.Tensor] = None,
        use_relu: bool = False,
    ):
        super().__init__(dataset)

        if label_key not in self.label_keys:
            raise ValueError(
                f"label_key='{label_key}' not found in dataset output schema. "
                f"Available label keys: {self.label_keys}"
            )
        if feature_key not in self.feature_keys:
            raise ValueError(
                f"feature_key='{feature_key}' not found in dataset input schema. "
                f"Available feature keys: {self.feature_keys}"
            )

        self.feature_key = feature_key
        self.label_key = label_key
        self.mask_key = mask_key

        self.num_genes = num_genes
        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.attention_size = attention_size
        self.attention_head = attention_head
        self.dropout_rate = dropout
        self.use_attention = use_attention
        self.use_cntx_attn = use_cntx_attn
        self.use_relu = use_relu

        # Force multilabel behavior for per-drug binary outputs.
        self.mode = "multilabel"

        if init_gene_emb is not None:
            expected_shape = (num_genes + 1, embedding_dim)
            if tuple(init_gene_emb.shape) != expected_shape:
                raise ValueError(
                    f"init_gene_emb must have shape {expected_shape}, "
                    f"got {tuple(init_gene_emb.shape)}"
                )
            self.gene_embedding = nn.Embedding.from_pretrained(
                init_gene_emb.float(),
                freeze=True,
                padding_idx=0,
            )
        else:
            self.gene_embedding = nn.Embedding(
                num_embeddings=num_genes + 1,
                embedding_dim=embedding_dim,
                padding_idx=0,
            )

        self.dropout = nn.Dropout(p=dropout)

        if use_attention:
            self.attn_proj = nn.Linear(embedding_dim, attention_size, bias=True)
            self.attn_beta = nn.Linear(attention_size, attention_head, bias=True)

            if use_cntx_attn:
                self.drug_context_embedding = nn.Embedding(
                    num_embeddings=num_drugs,
                    embedding_dim=attention_size,
                )

        # Optional projection to hidden_dim if embedding_dim != hidden_dim.
        if embedding_dim != hidden_dim:
            self.hidden_proj = nn.Linear(embedding_dim, hidden_dim)
        else:
            self.hidden_proj = nn.Identity()

        # Drug decoder, analogous to original DrugDecoder.
        self.drug_embedding = nn.Embedding(
            num_embeddings=num_drugs,
            embedding_dim=hidden_dim,
        )
        self.drug_bias = nn.Parameter(torch.zeros(num_drugs))

        # Loss: masked multilabel BCE with logits.
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.relu = nn.ReLU()

    def _encode(
        self,
        gene_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Encodes gene indices into drug-specific hidden representations.

        Args:
            gene_idx: Tensor of shape [batch_size, num_selected_genes].

        Returns:
            Tensor of shape [batch_size, num_drugs, hidden_dim].
        """
        if gene_idx.dtype != torch.long:
            gene_idx = gene_idx.long()

        # E_t: [B, G, E]
        gene_emb = self.gene_embedding(gene_idx)

        if self.use_attention:
            # Expand genes over drugs:
            # [B, 1, G, E] -> [B, D, G, E]
            gene_emb_exp = gene_emb.unsqueeze(1).repeat(1, self.num_drugs, 1, 1)

            # Base attention projection:
            # [B, D, G, A]
            attn_input = self.attn_proj(gene_emb_exp)

            if self.use_cntx_attn:
                # Drug-context embeddings: [D, A] -> [1, D, 1, A]
                drug_ids = torch.arange(
                    self.num_drugs, device=gene_idx.device, dtype=torch.long
                )
                drug_ctx = self.drug_context_embedding(drug_ids)
                drug_ctx = drug_ctx.unsqueeze(0).unsqueeze(2)
                attn_input = attn_input + drug_ctx

            attn_hidden = torch.tanh(attn_input)

            # [B, D, G, H]
            attn_scores = self.attn_beta(attn_hidden)

            # Softmax across genes, then collapse heads by summation.
            # [B, D, G, H] -> [B, D, G, 1]
            attn_weights = F.softmax(attn_scores, dim=2).sum(dim=3, keepdim=True)

            # Weighted sum across genes:
            # [B, D, 1, G] x [B, D, G, E] -> [B, D, 1, E] -> [B, D, E]
            drug_specific = torch.matmul(
                attn_weights.permute(0, 1, 3, 2), gene_emb_exp
            ).squeeze(2)
        else:
            # Mean pooling over genes, then repeat across drugs.
            # [B, G, E] -> [B, E] -> [B, D, E]
            pooled = gene_emb.mean(dim=1)
            drug_specific = pooled.unsqueeze(1).repeat(1, self.num_drugs, 1)

        drug_specific = self.hidden_proj(drug_specific)

        if self.use_relu:
            drug_specific = self.relu(drug_specific)

        drug_specific = self.dropout(drug_specific)
        return drug_specific

    def _decode(
        self,
        drug_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Decodes drug-specific hidden states into per-drug logits.

        Args:
            drug_hidden: Tensor of shape [batch_size, num_drugs, hidden_dim].

        Returns:
            Logits tensor of shape [batch_size, num_drugs].
        """
        batch_size = drug_hidden.shape[0]

        # [D, H] -> [B, D, H]
        drug_ids = torch.arange(
            self.num_drugs, device=drug_hidden.device, dtype=torch.long
        )
        drug_emb = self.drug_embedding(drug_ids).unsqueeze(0).repeat(batch_size, 1, 1)

        # Dot product across hidden dimension.
        logits = (drug_hidden * drug_emb).sum(dim=-1) + self.drug_bias
        return logits

    def _compute_loss(
        self,
        logits: torch.Tensor,
        y_true: torch.Tensor,
        y_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes masked multilabel BCE loss."""
        y_true = y_true.float()
        loss = self.loss_fn(logits, y_true)

        if y_mask is not None:
            y_mask = y_mask.float()
            denom = y_mask.sum().clamp_min(1.0)
            return (loss * y_mask).sum() / denom

        return loss.mean()

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of CADRE.

        Required kwargs:
            feature_key: gene index tensor [B, G]

        Optional kwargs:
            label_key: multilabel target tensor [B, D]
            mask_key: mask tensor [B, D]

        Returns:
            Dictionary containing:
                - logit: [B, D]
                - y_prob: [B, D]
                - loss: scalar tensor, if labels are provided
                - y_true: [B, D], if labels are provided
        """
        if self.feature_key not in kwargs:
            raise KeyError(
                f"Missing required feature key '{self.feature_key}' in forward input."
            )

        gene_idx = kwargs[self.feature_key]
        if isinstance(gene_idx, tuple):
            gene_idx = gene_idx[0]
        assert isinstance(gene_idx, torch.Tensor)

        drug_hidden = self._encode(gene_idx)
        logits = self._decode(drug_hidden)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key]
            if isinstance(y_true, tuple):
                y_true = y_true[0]
            assert isinstance(y_true, torch.Tensor)

            y_mask: Optional[torch.Tensor] = None
            if self.mask_key is not None and self.mask_key in kwargs:
                y_mask = kwargs[self.mask_key]
                if isinstance(y_mask, tuple):
                    y_mask = y_mask[0]
                assert isinstance(y_mask, torch.Tensor)

            loss = self._compute_loss(logits, y_true, y_mask)
            results["loss"] = loss
            results["y_true"] = y_true.float()

        return results
