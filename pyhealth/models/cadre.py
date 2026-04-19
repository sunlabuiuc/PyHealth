"""CADRE: Contextual Attention-based Drug REsponse prediction.

Re-implementation of:
    Tao, Y. et al. (2020). Predicting Drug Sensitivity of Cancer Cell Lines
    via Collaborative Filtering with Contextual Attention.
    Proceedings of Machine Learning Research, 126, 456-477. PMLR (MLHC 2020).

Original code: https://github.com/yifengtao/CADRE

Architecture:
    1. Gene Embedding Layer    pretrained Gene2Vec (3001 x 200), frozen
    2. Contextual Attention    drug pathway conditions gene importance
    3. Collaborative Filtering learned drug embeddings + dot-product decoder
    4. Prediction Head         logit per (cell line, drug) pair
    5. Masked BCE Loss         only scored on tested (cell line, drug) pairs
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpEncoder(nn.Module):
    """Gene expression encoder with contextual attention.

    Produces drug-specific cell-line representations by weighting gene
    embeddings according to the drug's target pathway context.

    Tensor shapes through the forward pass::

        gene_indices : (B, G)           B = batch, G = max active genes
        E            : (B, G, emb)      gene embeddings
        E_exp        : (B, D, G, emb)   expanded across D drugs
        Ep           : (1, D, 1, q)     pathway embeddings, broadcast
        H            : (B, D, G, q)     tanh(W·e_gene + e_pathway)
        A            : (B, D, G, h)     multi-head scores, softmax over G
        A_sum        : (B, D, G, 1)     summed across heads
        out          : (B, D, emb)      weighted gene representation per drug

    Args:
        gene_embeddings (np.ndarray): Pre-trained Gene2Vec matrix ``(3001, 200)``.
        num_pathways (int): Number of unique drug target pathways.
        embedding_dim (int): Gene embedding dimension. Default: ``200``.
        attention_size (int): Attention hidden dimension (q). Default: ``128``.
        attention_head (int): Number of attention heads (h). Default: ``8``.
        dropout_rate (float): Dropout probability. Default: ``0.6``.
        use_attention (bool): Enable attention mechanism. Default: ``True``.
        use_cntx_attn (bool): Enable contextual (pathway) conditioning.
            Default: ``True``.  If ``False``, runs self-attention only
            (reproduces the SADRE ablation from the paper).
    """

    def __init__(
        self,
        gene_embeddings: np.ndarray,
        num_pathways: int,
        embedding_dim: int = 200,
        attention_size: int = 128,
        attention_head: int = 8,
        dropout_rate: float = 0.6,
        use_attention: bool = True,
        use_cntx_attn: bool = True,
        freeze_gene_emb: bool = True,
    ) -> None:
        super().__init__()

        self.use_attention = use_attention
        self.use_cntx_attn = use_cntx_attn

        # Pretrained gene embeddings; frozen by default.
        # Set freeze_gene_emb=False to fine-tune them (CADRE∆pretrain variant).
        self.layer_emb = nn.Embedding.from_pretrained(
            torch.FloatTensor(gene_embeddings), freeze=freeze_gene_emb, padding_idx=0
        )
        self.layer_dropout = nn.Dropout(p=dropout_rate)

        if self.use_attention:
            self.layer_w_0 = nn.Linear(embedding_dim, attention_size, bias=True)
            self.layer_beta = nn.Linear(attention_size, attention_head, bias=True)
            if self.use_cntx_attn:
                self.layer_emb_ptw = nn.Embedding(
                    num_embeddings=num_pathways,
                    embedding_dim=attention_size,
                )

        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self, gene_indices: torch.Tensor, ptw_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute drug-specific cell representations.

        Args:
            gene_indices (torch.Tensor): Shape ``(B, G)`` — active gene
                indices (1-indexed; 0 = padding).
            ptw_ids (torch.Tensor): Shape ``(1, D)`` — pathway ID per drug.

        Returns:
            torch.Tensor: Shape ``(B, D, embedding_dim)``.
        """
        num_drugs = ptw_ids.shape[1]
        E = self.layer_emb(gene_indices)  # (B, G, emb)

        if self.use_attention:
            E_exp = E.unsqueeze(1).expand(-1, num_drugs, -1, -1)  # (B, D, G, emb)

            if self.use_cntx_attn:
                Ep = self.layer_emb_ptw(ptw_ids).unsqueeze(2)  # (1, D, 1, q)
                H = torch.tanh(self.layer_w_0(E_exp) + Ep)
            else:
                H = torch.tanh(self.layer_w_0(E_exp))

            A = F.softmax(self.layer_beta(H), dim=2)       # (B, D, G, h)
            A = A.sum(dim=3, keepdim=True)                  # (B, D, G, 1)
            self.attention_weights = A.squeeze(3)           # (B, D, G) for interpretability

            out = torch.matmul(A.permute(0, 1, 3, 2), E_exp).squeeze(2)  # (B, D, emb)
        else:
            out = E.mean(dim=1).unsqueeze(1).expand(-1, num_drugs, -1)   # (B, D, emb)

        return self.layer_dropout(out)


class DrugDecoder(nn.Module):
    """Collaborative filtering decoder with learned drug embeddings.

    Computes a dot product between the encoder output and learned drug
    embeddings to produce a sensitivity logit for each (cell line, drug) pair.

    Args:
        num_drugs (int): Number of drugs (260 for GDSC).
        embedding_dim (int): Embedding dimension. Default: ``200``.
    """

    def __init__(self, num_drugs: int, embedding_dim: int = 200) -> None:
        super().__init__()
        self.layer_emb_drg = nn.Embedding(
            num_embeddings=num_drugs, embedding_dim=embedding_dim
        )
        self.drg_bias = nn.Parameter(torch.zeros(num_drugs))

    def forward(
        self, cell_repr: torch.Tensor, drg_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-drug logits via dot product.

        Args:
            cell_repr (torch.Tensor): Shape ``(B, D, emb)`` from encoder.
            drg_ids (torch.Tensor): Shape ``(1, D)`` — drug index range.

        Returns:
            torch.Tensor: Shape ``(B, D)`` — raw logits.
        """
        D = self.layer_emb_drg(drg_ids).expand(cell_repr.shape[0], -1, -1)
        logits = (cell_repr * D).sum(dim=2) + self.drg_bias.unsqueeze(0)
        return logits


class CADRE(nn.Module):
    """CADRE: Contextual Attention-based Drug REsponse prediction model.

    Combines :class:`ExpEncoder` (frozen Gene2Vec embeddings + multi-head
    contextual attention) with :class:`DrugDecoder` (collaborative filtering)
    for multi-task binary drug sensitivity prediction.

    Integrates with PyHealth via :class:`~pyhealth.datasets.GDSCDataset`
    and :class:`~pyhealth.tasks.DrugSensitivityPredictionGDSC`.

    Args:
        gene_embeddings (np.ndarray): Pre-trained Gene2Vec matrix ``(3001, 200)``.
        num_drugs (int): Number of drugs. ``260`` for GDSC.
        num_pathways (int): Number of unique drug target pathways. ``25`` for GDSC.
        drug_pathway_ids (List[int]): Pathway ID for each drug, length ``num_drugs``.
        embedding_dim (int): Gene/drug embedding dimension. Default: ``200``.
        attention_size (int): Attention hidden dimension. Default: ``128``.
        attention_head (int): Number of attention heads. Default: ``8``.
        dropout_rate (float): Dropout probability. Default: ``0.6``.
        use_attention (bool): Enable attention. Default: ``True``.
        use_cntx_attn (bool): Enable contextual (pathway) conditioning.
            Default: ``True``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.models import CADRE
        >>> gene_emb = np.zeros((3001, 200))
        >>> model = CADRE(
        ...     gene_embeddings=gene_emb,
        ...     num_drugs=260,
        ...     num_pathways=25,
        ...     drug_pathway_ids=list(range(260)),
        ... )
        >>> gene_indices = torch.randint(1, 3001, (4, 1500))  # batch=4
        >>> out = model(gene_indices)
        >>> out["probs"].shape
        torch.Size([4, 260])
    """

    def __init__(
        self,
        gene_embeddings: np.ndarray,
        num_drugs: int,
        num_pathways: int,
        drug_pathway_ids: List[int],
        embedding_dim: int = 200,
        attention_size: int = 128,
        attention_head: int = 8,
        dropout_rate: float = 0.6,
        use_attention: bool = True,
        use_cntx_attn: bool = True,
        freeze_gene_emb: bool = True,
    ) -> None:
        super().__init__()

        self.num_drugs = num_drugs
        self.embedding_dim = embedding_dim

        self.register_buffer("ptw_ids", torch.LongTensor([drug_pathway_ids]))
        self.register_buffer("drg_ids", torch.arange(num_drugs).unsqueeze(0))

        self.encoder = ExpEncoder(
            gene_embeddings=gene_embeddings,
            num_pathways=num_pathways,
            embedding_dim=embedding_dim,
            attention_size=attention_size,
            attention_head=attention_head,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            use_cntx_attn=use_cntx_attn,
            freeze_gene_emb=freeze_gene_emb,
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
            gene_indices (torch.Tensor): Shape ``(B, G)`` — active gene
                indices padded to the batch maximum.
            labels (torch.Tensor, optional): Shape ``(B, D)`` — binary
                drug sensitivity labels.  Required to compute ``loss``.
            mask (torch.Tensor, optional): Shape ``(B, D)`` — 1 if the
                drug was tested for that cell line, 0 otherwise.  Required
                to compute ``loss``.

        Returns:
            dict: Always contains:

            * ``"logits"`` — raw logits ``(B, D)``
            * ``"probs"`` — sigmoid probabilities ``(B, D)``

            When ``labels`` and ``mask`` are provided, also contains:

            * ``"loss"`` — masked BCE loss (scalar)
            * ``"y_true"`` — same as ``labels``
        """
        cell_repr = self.encoder(gene_indices, self.ptw_ids)
        logits = self.decoder(cell_repr, self.drg_ids)
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
        """Return the last batch's attention weights for interpretability.

        Returns:
            torch.Tensor or None: Shape ``(B, D, G)`` — attention weight
            per (drug, gene) pair in the last forward call.
        """
        return self.encoder.attention_weights


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate a list of sample dicts into a batched tensor dict.

    Pads ``gene_indices`` to the maximum length within the batch using
    the padding index (0). All other fields are stacked directly.

    Args:
        batch (List[dict]): Sample dicts from
            :class:`~pyhealth.datasets.SampleBaseDataset`.

    Returns:
        dict: Keys ``gene_indices`` (LongTensor), ``labels`` (FloatTensor),
        ``mask`` (FloatTensor), ``patient_ids`` (List[str]).
    """
    max_genes = max(len(s["gene_indices"]) for s in batch)
    gene_indices, labels, masks, patient_ids = [], [], [], []

    for s in batch:
        gi = s["gene_indices"]
        gene_indices.append(gi + [0] * (max_genes - len(gi)))
        labels.append(s["labels"])
        masks.append(s["mask"])
        patient_ids.append(s["patient_id"])

    return {
        "gene_indices": torch.LongTensor(gene_indices),
        "labels": torch.FloatTensor(labels),
        "mask": torch.FloatTensor(masks),
        "patient_ids": patient_ids,
    }
