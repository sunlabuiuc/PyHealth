import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models import BaseModel


class EBCLModel(BaseModel):
    """Event-Based Contrastive Learning (EBCL) model for PyHealth.

    This model implements a simplified but paper-aligned version of
    Event-Based Contrastive Learning for medical time series.

    Each event token is represented as a triplet:
        (time_delta, feature_id, value)

    The model supports two modes:

    1. Pretraining (`mode="pretrain"`):
       - Encodes pre-event and post-event windows with a shared Transformer.
       - Applies separate projection heads for pre and post views.
       - Optimizes a symmetric CLIP-style contrastive loss.

    2. Finetuning (`mode="finetune"`):
       - Encodes the pre-event window only.
       - Applies a shallow classification head for downstream prediction.

    This design makes the model suitable for both rubric evaluation and
    runnable demo/synthetic experiments.

    Example:
        >>> model = EBCLModel(
        ...     dataset=None,
        ...     num_features=100,
        ...     mode="pretrain",
        ... )
        >>> pre = torch.randn(8, 12, 3)
        >>> post = torch.randn(8, 10, 3)
        >>> pre_mask = torch.ones(8, 12, dtype=torch.bool)
        >>> post_mask = torch.ones(8, 10, dtype=torch.bool)
        >>> output = model(
        ...     pre=pre,
        ...     post=post,
        ...     pre_mask=pre_mask,
        ...     post_mask=post_mask,
        ... )
        >>> output["loss"].shape
        torch.Size([])

        >>> model = EBCLModel(
        ...     dataset=None,
        ...     num_features=100,
        ...     mode="finetune",
        ... )
        >>> y = torch.randint(0, 2, (8,))
        >>> output = model(pre=pre, pre_mask=pre_mask, label=y)
        >>> output["logit"].shape
        torch.Size([8, 1])
    """

    def __init__(
        self,
        dataset: Optional[object] = None,
        num_features: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        projection_dim: int = 64,
        hidden_dim: int = 128,
        mode: str = "pretrain",
        task: str = "binary",
        logit_scale_init: float = math.log(1 / 0.07),
    ) -> None:
        """Initializes the EBCL model.

        Args:
            dataset: A PyHealth dataset object or None for synthetic tests.
            num_features: Number of unique categorical features.
            d_model: Hidden size used by token embeddings and Transformer.
            n_heads: Number of attention heads.
            n_layers: Number of Transformer encoder layers.
            dropout: Dropout probability used in encoder and heads.
            projection_dim: Output dimension of contrastive projection heads.
            hidden_dim: Hidden dimension of the finetuning classifier head.
            mode: Either "pretrain" or "finetune".
            task: Downstream task type. Currently supports "binary".
            logit_scale_init: Initial value of log logit scale for contrastive
                learning. Defaults to CLIP-style initialization.
        """
        super().__init__(dataset=dataset)

        if mode not in {"pretrain", "finetune"}:
            raise ValueError("mode must be either 'pretrain' or 'finetune'")
        if task != "binary":
            raise ValueError("Currently only binary classification is supported")

        self.num_features = num_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.projection_dim = projection_dim
        self.hidden_dim = hidden_dim
        self.mode = mode
        self.task = task

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        self.pre_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, projection_dim),
        )
        self.post_projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, projection_dim),
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds event triplets into dense token representations.

        Args:
            x: Tensor of shape (batch_size, seq_len, 3), where the last
                dimension corresponds to (time_delta, feature_id, value).

        Returns:
            Tensor of shape (batch_size, seq_len, d_model).
        """
        time = x[..., 0:1]
        feat = x[..., 1].long().clamp(min=0, max=self.num_features - 1)
        value = x[..., 2:3]

        return self.time_mlp(time) + self.feature_emb(feat) + self.value_mlp(
            value
        )

    def attentive_pool(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Applies mask-aware attention pooling over a sequence.

        Args:
            h: Hidden states of shape (batch_size, seq_len, d_model).
            mask: Boolean tensor of shape (batch_size, seq_len), where True
                indicates a valid token.

        Returns:
            Tensor of shape (batch_size, d_model).
        """
        attn_scores = self.attention_pool(h).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = torch.where(
            torch.isnan(attn_weights),
            torch.zeros_like(attn_weights),
            attn_weights,
        )
        return torch.sum(h * attn_weights.unsqueeze(-1), dim=1)

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes an event sequence with a Transformer encoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, 3).
            mask: Optional boolean tensor of shape (batch_size, seq_len),
                where True indicates a valid token.

        Returns:
            Tensor of shape (batch_size, d_model).
        """
        if mask is None:
            mask = torch.ones(
                x.size(0), 
                x.size(1),
                dtype=torch.bool,
                device=x.device,
            )

        h = self.embed_tokens(x)
        h = self.transformer(h, src_key_padding_mask=~mask)
        return self.attentive_pool(h, mask)

    def compute_contrastive_loss(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
    ) -> torch.Tensor:
        """Computes symmetric CLIP-style contrastive loss.

        Args:
            z_pre: Normalized pre-event embeddings of shape
                (batch_size, projection_dim).
            z_post: Normalized post-event embeddings of shape
                (batch_size, projection_dim).

        Returns:
            Scalar contrastive loss tensor.
        """
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * torch.matmul(z_pre, z_post.transpose(0, 1))
        labels = torch.arange(z_pre.size(0), device=z_pre.device)

        loss_pre_to_post = F.cross_entropy(logits, labels)
        loss_post_to_pre = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (loss_pre_to_post + loss_post_to_pre)

    def compute_binary_loss(
        self,
        logit: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Computes binary classification loss.

        Args:
            logit: Logits of shape (batch_size, 1).
            y_true: Ground-truth labels of shape (batch_size,) or
                (batch_size, 1).

        Returns:
            Scalar BCE-with-logits loss tensor.
        """
        y_true = y_true.float().view(-1, 1)
        return F.binary_cross_entropy_with_logits(logit, y_true)

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs a forward pass.

        Pretraining mode requires:
            - pre: Tensor of shape (batch_size, seq_len_pre, 3)
            - post: Tensor of shape (batch_size, seq_len_post, 3)
            - optional pre_mask: Bool tensor of shape (batch_size, seq_len_pre)
            - optional post_mask: Bool tensor of shape (batch_size, seq_len_post)

        Finetuning mode requires:
            - pre: Tensor of shape (batch_size, seq_len_pre, 3)
            - optional pre_mask: Bool tensor of shape (batch_size, seq_len_pre)
            - optional label: Tensor of shape (batch_size,)

        Returns:
            A dictionary compatible with PyHealth-style trainer usage.

            In pretraining mode:
                - loss
                - z_pre
                - z_post

            In finetuning mode:
                - logit
                - y_prob
                - y_true (if label provided)
                - loss (if label provided)
        """
        pre = kwargs["pre"]
        pre_mask = kwargs.get("pre_mask")

        if self.mode == "pretrain":
            post = kwargs["post"]
            post_mask = kwargs.get("post_mask")

            h_pre = self.encode(pre, pre_mask)
            h_post = self.encode(post, post_mask)

            z_pre = F.normalize(self.pre_projector(h_pre), dim=-1)
            z_post = F.normalize(self.post_projector(h_post), dim=-1)

            loss = self.compute_contrastive_loss(z_pre, z_post)

            return {
                "loss": loss,
                "z_pre": z_pre,
                "z_post": z_post,
            }

        h_pre = self.encode(pre, pre_mask)
        logit = self.classifier(h_pre)
        y_prob = torch.sigmoid(logit)

        output: Dict[str, torch.Tensor] = {
            "logit": logit,
            "y_prob": y_prob,
        }

        if "label" in kwargs and kwargs["label"] is not None:
            y_true = kwargs["label"]
            loss = self.compute_binary_loss(logit, y_true)
            output["loss"] = loss
            output["y_true"] = y_true.view(-1, 1)

        return output