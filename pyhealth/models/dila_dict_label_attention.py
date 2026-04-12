"""Dictionary Label Attention module for DILA.

Implements equations 5-7 from:
    DILA: Dictionary Label Attention for Interpretable ICD Coding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.dila_sparse_autoencoder import SparseAutoencoder


class DictionaryLabelAttention(nn.Module):
    """Dictionary-guided label attention mechanism for multi-label ICD coding.

    Encodes each PLM token into a sparse dictionary feature vector, then
    projects through an ICD-initialized matrix to produce per-token attention
    weights over each ICD code.  The weighted token representations are
    aggregated per label and passed through a per-label classification head.

    Pipeline (Eq. 5-7):
        F_note  = encode(X_note)         ∈ ℝ^(s×m)   sparse token features
        A_laat  = softmax(F_note·A_ficd) ∈ ℝ^(s×c)   per-token label attention
        X_att   = A_laat^T · X_note      ∈ ℝ^(c×d)   attended representations
        logits  = diag(X_att · W_o^T) + b_o           per-label scores

    where A_ficd ∈ ℝ^(m×c) is the ICD projection matrix (stored transposed
    as icd_projection of shape (c, m) for efficient matmul).

    Args:
        autoencoder: SparseAutoencoder instance used to encode token embeddings.
            May be pretrained or jointly trained with this module.
        num_labels: Number of ICD codes (output classes, c).
        input_dim: Dimensionality of PLM token embeddings (d).

    Examples:
        >>> sae = SparseAutoencoder(input_dim=64, dict_size=256)
        >>> attn = DictionaryLabelAttention(sae, num_labels=10, input_dim=64)
        >>> x = torch.randn(4, 16, 64)   # (batch, seq_len, input_dim)
        >>> logits, losses = attn(x)
        >>> logits.shape
        torch.Size([4, 10])
    """

    def __init__(
        self,
        autoencoder: SparseAutoencoder,
        num_labels: int,
        input_dim: int,
    ):
        super().__init__()
        self.autoencoder = autoencoder
        self.num_labels = num_labels
        self.input_dim = input_dim

        # A_ficd stored as (num_labels, dict_size) so:
        # attn_logits = F_note @ icd_projection.T   (B, S, F) @ (F, C) → (B, S, C)
        self.icd_projection = nn.Parameter(
            torch.empty(num_labels, autoencoder.dict_size)
        )
        # Per-label output head — weight row c projects the attended representation
        # for label c down to a scalar logit, matching the reference code pattern
        self.output_head = nn.Linear(input_dim, num_labels)

        nn.init.normal_(self.icd_projection, mean=0.0, std=0.03)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.03)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute label attention and produce per-label logits.

        Args:
            x: PLM token embeddings of shape (batch, seq_len, input_dim).

        Returns:
            Tuple of:
                logits (Tensor): Per-label logits of shape (batch, num_labels).
                aux_losses (dict): Dict with key "loss_saenc" (scalar tensor)
                    containing the autoencoder reconstruction + sparsity loss.
        """
        batch, seq_len, _ = x.size()

        # Encode all tokens through the sparse autoencoder
        x_flat = x.reshape(batch * seq_len, self.input_dim)
        f_flat, _, loss_dict = self.autoencoder(x_flat)
        # f_note: (batch, seq_len, dict_size)
        f_note = f_flat.view(batch, seq_len, self.autoencoder.dict_size)

        # Per-token attention logits over ICD codes (Eq. 6)
        # (B, S, dict_size) @ (dict_size, C) → (B, S, num_labels)
        attn_logits = f_note @ self.icd_projection.t()

        # Softmax over seq_len: for each label, attention weights across tokens sum to 1
        a_laat = F.softmax(attn_logits, dim=1)  # (B, S, num_labels)

        # Attended representations per label (Eq. 7)
        # (B, num_labels, S) @ (B, S, input_dim) → (B, num_labels, input_dim)
        x_att = a_laat.transpose(1, 2) @ x

        # Per-label logits: for label c, dot output_head.weight[c] with x_att[:, c, :]
        # output_head.weight is (num_labels, input_dim); x_att is (B, num_labels, input_dim)
        # Element-wise multiply then sum over input_dim → (B, num_labels)
        logits = (
            self.output_head.weight.mul(x_att)
            .sum(dim=2)
            .add(self.output_head.bias)
        )

        return logits, {"loss_saenc": loss_dict["loss_saenc"]}

    def initialize_from_icd_descriptions(
        self, description_embeddings: torch.Tensor
    ) -> None:
        """Initialize icd_projection from precomputed ICD description embeddings.

        Implements Eq. 5: sets each column of A_ficd to the average-pooled
        sparse feature vector computed from the corresponding ICD code's
        textual description.

        Args:
            description_embeddings: Tensor of shape (num_labels, dict_size)
                containing one average-pooled sparse feature vector per ICD
                code, as returned by compute_icd_projection_init().

        Raises:
            ValueError: If description_embeddings has the wrong shape.
        """
        expected = (self.num_labels, self.autoencoder.dict_size)
        if tuple(description_embeddings.shape) != expected:
            raise ValueError(
                f"Expected shape {expected}, got {tuple(description_embeddings.shape)}"
            )
        with torch.no_grad():
            self.icd_projection.copy_(description_embeddings)

    @staticmethod
    def compute_icd_projection_init(
        autoencoder: SparseAutoencoder,
        icd_descriptions: list,
        tokenizer,
        plm_model: nn.Module,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Compute the A_ficd initialization matrix from ICD code description text.

        Implements Eq. 5 of the DILA paper.  For each ICD code:
            1. Tokenize and encode the description with the PLM → token embeddings.
            2. Pass every token embedding through the sparse autoencoder → sparse features.
            3. Average-pool sparse features over the description tokens → prototype vector.

        The resulting (num_labels, dict_size) tensor can be passed directly to
        ``initialize_from_icd_descriptions()``.

        Args:
            autoencoder: Trained (or pretrained) SparseAutoencoder instance.
            icd_descriptions: List of c description strings, one per ICD code, in
                the same order as the model's output labels.
            tokenizer: HuggingFace tokenizer compatible with ``plm_model``.
            plm_model: HuggingFace PLM (e.g., RoBERTa) used to embed description
                tokens.  Should already be in eval mode.
            device: Target device for computation. Default: "cpu".
            batch_size: Number of descriptions to process in one PLM forward pass.
                Default: 32.

        Returns:
            Tensor of shape (num_labels, dict_size) containing one average-pooled
            sparse feature prototype per ICD code.
        """
        autoencoder = autoencoder.to(device)
        plm_model = plm_model.to(device)
        autoencoder.eval()
        plm_model.eval()

        prototypes = []

        with torch.no_grad():
            # split by batch
            for i in range(0, len(icd_descriptions), batch_size):
                batch_descs = icd_descriptions[i : i + batch_size]

                encoding = tokenizer(
                    batch_descs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = encoding["input_ids"].to(device)
                attention_mask = encoding["attention_mask"].to(device)

                outputs = plm_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # token_embeddings: (batch_desc, seq_len, hidden_dim)
                token_embeddings = outputs.last_hidden_state

                for j in range(token_embeddings.size(0)):
                    # Select only non-padding positions
                    mask_j = attention_mask[j].bool()
                    valid_embeddings = token_embeddings[j][mask_j]  # (l, d)

                    # Encode each token through the sparse autoencoder
                    f_desc, _, _ = autoencoder(valid_embeddings)  # (l, dict_size)

                    # Average-pool over tokens → prototype for this ICD code
                    f_bar = f_desc.mean(dim=0)  # (dict_size,)
                    prototypes.append(f_bar.cpu())

        return torch.stack(prototypes, dim=0)  # (num_labels, dict_size)
