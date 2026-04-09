import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from pyhealth.models import BaseModel


class EBCLModel(BaseModel):
    """Event-Based Contrastive Learning (EBCL) Model for PyHealth.

    This model implements the pretraining phase of EBCL, utilizing a
    tokenized representation of Electronic Health Records (EHR) and a
    Transformer encoder to align pre-event and post-event patient trajectories.

    Example:
        >>> dataset = None  # Replace with instantiated PyHealth dataset
        >>> model = EBCLPyHealthModel(dataset=dataset, num_features=1000)
        >>> pre_data = torch.randn(16, 50, 3)  # (Batch, Seq_Len, 3)
        >>> post_data = torch.randn(16, 30, 3)
        >>> kwargs = {"pre": pre_data, "post": post_data}
        >>> output = model(**kwargs)
        >>> print(output["loss"].item())
    """

    def __init__(
        self,
        dataset: Optional[object] = None,
        num_features: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        projection_dim: int = 64,
        temperature: float = 0.1,
    ) -> None:
        """Initializes the EBCLPyHealthModel.

        Args:
            dataset: A PyHealth dataset object.
            num_features: Total number of unique categorical features (vocab size).
            d_model: Hidden dimension size for the Transformer and embeddings.
            n_heads: Number of attention heads in the Transformer.
            n_layers: Number of Transformer encoder layers.
            projection_dim: Output dimension of the contrastive projection heads.
            temperature: Temperature scaling factor for the InfoNCE loss.
        """
        super().__init__(dataset=dataset)

        self.temperature = temperature

        # -------------------------
        # Embeddings
        # -------------------------
        self.feature_emb = nn.Embedding(num_features, d_model)

        self.value_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # -------------------------
        # Transformer
        # -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # -------------------------
        # Separate projection heads
        # -------------------------
        self.pre_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

        self.post_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds the (time, feature, value) triplets into a dense representation.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 3), where the last
                dimension represents (time_delta, feature_id, continuous_value).

        Returns:
            A dense representation tensor of shape (batch_size, seq_len, d_model).
        """
        time = x[..., 0:1]
        feat = x[..., 1].long()
        value = x[..., 2:3]

        return (
            self.time_mlp(time)
            + self.feature_emb(feat)
            + self.value_mlp(value)
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes tokenized sequences using the Transformer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 3).

        Returns:
            A pooled representation of the sequence of shape (batch_size, d_model).
        """
        h = self.embed_tokens(x)
        h = self.transformer(h)
        return h.mean(dim=1)

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for PyHealth Trainer integration.

        Args:
            **kwargs: Dictionary containing the batched input tensors. Must
                include the keys "pre" and "post", which map to tensors of shape
                (batch_size, seq_len, 3).

        Returns:
            A dictionary containing:
                - "loss": The scalar contrastive loss value.
                - "z_pre": The normalized pre-event embeddings.
                - "z_post": The normalized post-event embeddings.
        """
        pre = kwargs["pre"]
        post = kwargs["post"]

        h_pre = self.encode(pre)
        h_post = self.encode(post)

        z_pre = F.normalize(self.pre_projector(h_pre), dim=-1)
        z_post = F.normalize(self.post_projector(h_post), dim=-1)

        loss = self.compute_loss(z_pre, z_post)

        return {
            "loss": loss,
            "z_pre": z_pre,
            "z_post": z_post,
        }

    def compute_loss(self, z_pre: torch.Tensor, z_post: torch.Tensor) -> torch.Tensor:
        """Computes the symmetric CLIP-style contrastive loss.

        Args:
            z_pre: Normalized pre-event embeddings of shape (batch_size, projection_dim).
            z_post: Normalized post-event embeddings of shape (batch_size, projection_dim).

        Returns:
            A scalar tensor representing the symmetric InfoNCE loss.
        """
        logits = torch.matmul(z_pre, z_post.T) / self.temperature

        labels = torch.arange(z_pre.size(0), device=z_pre.device)

        loss_1 = F.cross_entropy(logits, labels)
        loss_2 = F.cross_entropy(logits.T, labels)

        return (loss_1 + loss_2) / 2.0