"""DILA: Dictionary Label Attention for interpretable ICD coding.

Implements the full two-stage DILA pipeline from:
    DILA: Dictionary Label Attention for Interpretable ICD Coding

This module provides:
    - DILA: PyHealth BaseModel integrating the sparse autoencoder and attention head.
    - pretrain_sparse_autoencoder: Stage-1 SAE pretraining on PLM embeddings.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel
from pyhealth.models.dila_sparse_autoencoder import SparseAutoencoder
from pyhealth.models.dila_dict_label_attention import DictionaryLabelAttention

logger = logging.getLogger(__name__)


class DILA(BaseModel):
    """Dictionary Label Attention model for interpretable multi-label ICD coding.

    A two-stage model that replaces the standard nonlinear label attention with
    a sparse, dictionary-guided mechanism:

    1. A SparseAutoencoder decomposes dense PLM token embeddings into a sparse
       set of dictionary features.
    2. A DictionaryLabelAttention head projects those features through an
       ICD-initialized matrix to produce interpretable per-token attention
       weights, which are used to aggregate token representations per label.

    The model expects pre-encoded token-level embeddings (e.g., from a
    fine-tuned RoBERTa) as its input feature, rather than raw text.  Users are
    responsible for encoding clinical notes with a PLM and storing the resulting
    embeddings as the input feature in their SampleDataset.

    Combined training loss (Eq. 8):
        L = lambda_saenc * L_saenc + L_BCE

    Args:
        dataset: PyHealth SampleDataset used to determine num_labels and to
            follow the BaseModel convention.
        feature_key: Key in the dataset sample dict containing the dense
            embedding tensor of shape (seq_len, embedding_dim).
        label_key: Key in the dataset sample dict containing the multilabel
            target.
        embedding_dim: Dimensionality of the PLM token embeddings. Default: 768.
        dict_size: Number of dictionary features (m). Default: 4096.
        lambda_l1: L1 sparsity coefficient for the SAE. Default: 1e-4.
        lambda_l2: L2 regularization coefficient for the SAE. Default: 1e-5.
        lambda_saenc: Weight of the SAE loss in the combined training loss.
            Default: 1e-6.
        pretrained_autoencoder_path: Path to a saved SparseAutoencoder
            state_dict (from pretrain_sparse_autoencoder). When provided the
            autoencoder weights are loaded before training begins. Default: None.

    Examples:
        >>> # Stage 1: pretrain the sparse autoencoder on all PLM embeddings
        >>> sae = SparseAutoencoder(input_dim=768, dict_size=4096)
        >>> pretrain_sparse_autoencoder(sae, all_embeddings, epochs=10,
        ...                             save_path="sae.pt")
        >>>
        >>> # (Optional) initialize ICD projection from description text
        >>> proj_init = DictionaryLabelAttention.compute_icd_projection_init(
        ...     sae, icd_descs, tok, plm)
        >>>
        >>> # Stage 2: full DILA training via PyHealth Trainer
        >>> model = DILA(dataset, feature_key="embeddings",
        ...              label_key="icd_codes", pretrained_autoencoder_path="sae.pt")
        >>> model.dict_label_att.initialize_from_icd_descriptions(proj_init)
        >>>
        >>> trainer = Trainer(model)
        >>> trainer.train(train_loader, val_loader, test_loader, epochs=3)
    """

    def __init__(
        self,
        dataset: SampleDataset,
        feature_key: str,
        label_key: str,
        embedding_dim: int = 768,
        dict_size: int = 4096,
        lambda_l1: float = 1e-4,
        lambda_l2: float = 1e-5,
        lambda_saenc: float = 1e-6,
        pretrained_autoencoder_path: Optional[str] = None,
    ):
        super().__init__(dataset)
        self.feature_key = feature_key
        self.label_key = label_key
        self.embedding_dim = embedding_dim
        self.dict_size = dict_size
        self.lambda_saenc = lambda_saenc
        self.mode = "multilabel"

        num_labels = dataset.output_processors[label_key].size()

        self.autoencoder = SparseAutoencoder(
            input_dim=embedding_dim,
            dict_size=dict_size,
            lambda_l1=lambda_l1,
            lambda_l2=lambda_l2,
        )
        if pretrained_autoencoder_path is not None:
            state = torch.load(pretrained_autoencoder_path, map_location="cpu")
            self.autoencoder.load_state_dict(state)

        self.dict_label_att = DictionaryLabelAttention(
            autoencoder=self.autoencoder,
            num_labels=num_labels,
            input_dim=embedding_dim,
        )

    def forward(self, **kwargs) -> dict:
        """Forward pass.

        Args:
            **kwargs: Batch dict from the dataloader. Must contain
                ``feature_key`` with a dense embedding tensor of shape
                (batch, seq_len, embedding_dim). When ``label_key`` is
                present, the loss is computed and included in the output.

        Returns:
            dict with keys:
                - ``"logit"``: Raw logits of shape (batch, num_labels).
                - ``"y_prob"``: Sigmoid probabilities of shape (batch, num_labels).
                - ``"loss"`` (when labels present): Combined scalar loss.
                - ``"loss_bce"`` (when labels present): Binary cross-entropy loss.
                - ``"loss_saenc"`` (when labels present): SAE reconstruction loss.
                - ``"y_true"`` (when labels present): Ground-truth label tensor.
        """
        # Extract dense embedding tensor
        feature = kwargs[self.feature_key]
        x = feature[0] if isinstance(feature, tuple) else feature
        x = x.to(self.device)

        logits, aux_losses = self.dict_label_att(x)
        y_prob = torch.sigmoid(logits)

        results = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            label = kwargs[self.label_key]
            y_true = label[0] if isinstance(label, tuple) else label
            y_true = y_true.to(self.device)

            loss_bce = F.binary_cross_entropy_with_logits(logits, y_true)
            loss = self.lambda_saenc * aux_losses["loss_saenc"] + loss_bce

            results["loss"] = loss
            results["loss_bce"] = loss_bce
            results["loss_saenc"] = aux_losses["loss_saenc"]
            results["y_true"] = y_true

        return results


# ---------------------------------------------------------------------------
# Stage-1 utility: pretrain the sparse autoencoder
# ---------------------------------------------------------------------------


def pretrain_sparse_autoencoder(
    autoencoder: SparseAutoencoder,
    embeddings: Union[torch.Tensor, DataLoader],
    epochs: int = 10,
    lr: float = 5e-5,
    batch_size: int = 256,
    device: str = "cpu",
    save_path: Optional[str] = None,
) -> SparseAutoencoder:
    """Pretrain a SparseAutoencoder on a corpus of PLM embeddings (Stage 1).

    Runs the autoencoder training loop independently of the full DILA model,
    enabling the two-stage workflow described in Section 3 of the paper.  After
    pretraining the saved weights can be passed to DILA via
    ``pretrained_autoencoder_path``.

    Each optimizer step is followed by ``normalize_decoder()`` to keep decoder
    column norms at unity.

    Args:
        autoencoder: SparseAutoencoder instance to train (modified in-place).
        embeddings: Either a 2-D tensor of shape (N, embedding_dim) or a
            DataLoader whose batches are either tensors of shape (B, embedding_dim)
            or tuples whose first element is such a tensor.
        epochs: Number of full passes over the embedding corpus. Default: 10.
        lr: AdamW learning rate. Default: 5e-5.
        batch_size: Batch size used when ``embeddings`` is a raw tensor.
            Ignored when a DataLoader is provided. Default: 256.
        device: Target device string (e.g. "cpu", "cuda:0"). Default: "cpu".
        save_path: If provided, the trained autoencoder state_dict is saved to
            this path after training completes. Default: None.

    Returns:
        The trained SparseAutoencoder (same object as ``autoencoder``).
    """
    autoencoder = autoencoder.to(device)
    autoencoder.train()

    if isinstance(embeddings, torch.Tensor):
        dataset = TensorDataset(embeddings)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        loader = embeddings

    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_saenc = 0.0
        total_recon = 0.0
        total_l1 = 0.0
        n_batches = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            _, _, loss_dict = autoencoder(x)
            loss = loss_dict["loss_saenc"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            autoencoder.normalize_decoder()

            total_saenc += loss_dict["loss_saenc"].item()
            total_recon += loss_dict["loss_recon"].item()
            total_l1 += loss_dict["loss_l1"].item()
            n_batches += 1

        logger.info(
            "Epoch %d/%d — loss_saenc: %.6f  loss_recon: %.6f  loss_l1: %.6f",
            epoch,
            epochs,
            total_saenc / max(n_batches, 1),
            total_recon / max(n_batches, 1),
            total_l1 / max(n_batches, 1),
        )

    if save_path is not None:
        torch.save(autoencoder.state_dict(), save_path)
        logger.info("Autoencoder weights saved to %s", save_path)

    return autoencoder
