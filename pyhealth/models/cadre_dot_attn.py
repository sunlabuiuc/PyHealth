"""CADREDotAttn: CADRE with scaled dot-product attention (Extension).

Replaces CADRE's additive contextual attention::

    score = W_beta · tanh(W_alpha · e_gene + e_pathway)   [CADRE]

with transformer-style scaled dot-product attention::

    score = (W_Q · e_drug) · (W_K · e_gene)^T / sqrt(d_k)  [this module]

Drug embeddings from :class:`~pyhealth.models.DrugDecoder` serve as
attention queries, so gradients flow through both the prediction path
(decoder dot-product) and the attention path (encoder alignment), jointly
shaping drug representations to predict sensitivity *and* attend to
relevant genes.

Reference extension of:
    Tao, Y. et al. (2020). Predicting Drug Sensitivity of Cancer Cell Lines
    via Collaborative Filtering with Contextual Attention. MLHC 2020.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.cadre import DrugDecoder


class DotProductExpEncoder(nn.Module):
    """Gene expression encoder using scaled dot-product (transformer) attention.

    Tensor shapes through the forward pass::

        gene_indices : (B, G)               B = batch, G = max genes
        G_emb        : (B, G, emb)          gene embeddings
        K            : (B, H, G, d_k)       gene key vectors
        Q            : (B, H, D, d_k)       drug query vectors
        scores       : (B, H, D, G)         scaled dot-product scores
        attn         : (B, H, D, G)         softmax (padding masked to -inf)
        context      : (B, H, D, d_k)
        output       : (B, D, emb)

    Args:
        gene_embeddings (np.ndarray): Pre-trained Gene2Vec matrix ``(3001, 200)``.
        num_drugs (int): Number of drugs.
        embedding_dim (int): Gene/drug embedding dimension. Default: ``200``.
        num_heads (int): Number of attention heads. Default: ``8``.
        d_k (int): Key/query dimension per head. Default: ``64``.
        dropout_rate (float): Dropout probability. Default: ``0.6``.
    """

    def __init__(
        self,
        gene_embeddings: np.ndarray,
        num_drugs: int,
        embedding_dim: int = 200,
        num_heads: int = 8,
        d_k: int = 64,
        dropout_rate: float = 0.6,
    ) -> None:
        super().__init__()

        self.num_drugs = num_drugs
        self.num_heads = num_heads
        self.d_k = d_k
        self.embedding_dim = embedding_dim

        self.layer_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(gene_embeddings), freeze=True, padding_idx=0
        )
        self.key_proj = nn.Linear(embedding_dim, num_heads * d_k, bias=False)
        self.query_proj = nn.Linear(embedding_dim, num_heads * d_k, bias=False)
        self.W_O = nn.Linear(num_heads * d_k, embedding_dim, bias=False)
        self.layer_dropout = nn.Dropout(p=dropout_rate)

        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self, gene_indices: torch.Tensor, drug_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute drug-specific cell representations via dot-product attention.

        Args:
            gene_indices (torch.Tensor): Shape ``(B, G)``; 0 = padding.
            drug_embeddings (torch.Tensor): Shape ``(D, emb)`` from
                :class:`~pyhealth.models.DrugDecoder`.

        Returns:
            torch.Tensor: Shape ``(B, D, embedding_dim)``.
        """
        B, G = gene_indices.shape
        D = drug_embeddings.shape[0]
        H, dk = self.num_heads, self.d_k

        G_emb = self.layer_emb(gene_indices)  # (B, G, emb)

        # Padding mask: positions where gene_index == 0
        pad_mask = (gene_indices == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, G)

        K = self.key_proj(G_emb).view(B, G, H, dk).transpose(1, 2)   # (B, H, G, dk)
        Q = (
            self.query_proj(drug_embeddings)
            .view(D, H, dk)
            .permute(1, 0, 2)
            .unsqueeze(0)
            .expand(B, -1, -1, -1)
        )  # (B, H, D, dk)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (dk ** 0.5)  # (B, H, D, G)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1).nan_to_num(0.0)

        self.attention_weights = attn.mean(dim=1).detach()  # (B, D, G)

        context = torch.matmul(attn, K)                           # (B, H, D, dk)
        context = context.permute(0, 2, 1, 3).contiguous()       # (B, D, H, dk)
        context = context.view(B, D, H * dk)
        out = self.W_O(context)                                    # (B, D, emb)

        return self.layer_dropout(out)


class CADREDotAttn(nn.Module):
    """CADRE variant using scaled dot-product attention (transformer-style).

    Replaces CADRE's additive contextual attention with multi-head
    scaled dot-product attention.  Drug embeddings act as queries so they
    receive gradients from both the prediction dot-product (decoder) and
    the attention alignment (encoder), shaping them jointly.

    See :class:`~pyhealth.models.CADRE` for the baseline model.

    Args:
        gene_embeddings (np.ndarray): Pre-trained Gene2Vec matrix ``(3001, 200)``.
        num_drugs (int): Number of drugs. ``260`` for GDSC.
        embedding_dim (int): Gene/drug embedding dimension. Default: ``200``.
        num_heads (int): Number of attention heads. Default: ``8``.
        d_k (int): Key/query dimension per head. Default: ``64``.
        dropout_rate (float): Dropout probability. Default: ``0.6``.

    Examples:
        >>> import numpy as np, torch
        >>> from pyhealth.models import CADREDotAttn
        >>> gene_emb = np.zeros((3001, 200))
        >>> model = CADREDotAttn(gene_embeddings=gene_emb, num_drugs=260)
        >>> gene_indices = torch.randint(1, 3001, (4, 1500))
        >>> out = model(gene_indices)
        >>> out["probs"].shape
        torch.Size([4, 260])
    """

    def __init__(
        self,
        gene_embeddings: np.ndarray,
        num_drugs: int,
        embedding_dim: int = 200,
        num_heads: int = 8,
        d_k: int = 64,
        dropout_rate: float = 0.6,
    ) -> None:
        super().__init__()

        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim

        self.register_buffer("drg_ids", torch.arange(num_drugs).unsqueeze(0))

        self.encoder = DotProductExpEncoder(
            gene_embeddings=gene_embeddings,
            num_drugs=num_drugs,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            d_k=d_k,
            dropout_rate=dropout_rate,
        )
        self.decoder = DrugDecoder(num_drugs=num_drugs, embedding_dim=embedding_dim)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
        self,
        gene_indices: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            gene_indices (torch.Tensor): Shape ``(B, G)``.
            labels (torch.Tensor, optional): Shape ``(B, D)`` binary labels.
            mask (torch.Tensor, optional): Shape ``(B, D)`` tested mask.

        Returns:
            dict: ``"logits"``, ``"probs"``, optionally ``"loss"``,
            ``"y_true"``, and ``"attention"``.
        """
        drug_emb = self.decoder.layer_emb_drg(self.drg_ids).squeeze(0)  # (D, emb)
        cell_repr = self.encoder(gene_indices, drug_emb)                  # (B, D, emb)
        logits = self.decoder(cell_repr, self.drg_ids)                    # (B, D)
        probs = torch.sigmoid(logits)

        result: Dict[str, torch.Tensor] = {"logits": logits, "probs": probs}

        if labels is not None and mask is not None:
            per_element = self.loss_fn(logits, labels.float())
            result["loss"] = (per_element * mask).sum() / (mask.sum() + 1e-5)
            result["y_true"] = labels

        if self.encoder.attention_weights is not None:
            result["attention"] = self.encoder.attention_weights

        return result

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return last batch attention weights for interpretability.

        Returns:
            torch.Tensor or None: Shape ``(B, D, G)``.
        """
        return self.encoder.attention_weights
