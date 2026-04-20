import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.models.base_model import BaseModel


class EBCLModel(BaseModel):
    """Event-Based Contrastive Learning model.

    This model implements an event-centered representation learning approach
    with two stages:

    1. Pretraining:
       Learns aligned representations of paired pre-event and post-event
       windows using a symmetric contrastive loss.

    2. Finetuning:
       Uses the pre-event representation for downstream binary prediction,
       such as mortality or 3-day length of stay.

    Each token in the input sequence is represented as a triplet:
        [time_value, feature_id, measurement_value]

    Therefore, the expected input tensor shape is:
        (batch_size, sequence_length, 3)

    Args:
        dataset: Optional PyHealth dataset object used to initialize the model.
        num_features: Number of unique feature ids in the token vocabulary.
            Default is 1000.
        d_model: Token embedding dimension. Default is 32.
        n_heads: Number of attention heads in the transformer encoder.
            Default is 4.
        n_layers: Number of transformer encoder layers. Default is 2.
        ff_hidden_dim: Hidden size of the feed-forward block inside each
            transformer layer. Default is 128.
        dropout: Dropout rate used in the transformer and classifier head.
            Default is 0.1.
        projection_dim: Output dimension of the pretraining projection heads.
            Default is 32.
        stage: Training stage, either "pretrain" or "finetune".
            Default is "pretrain".
        task: Prediction mode used for PyHealth compatibility. Only "binary"
            is currently supported. Default is "binary".
        logit_scale_init: Initial value of the learnable logit scale used in
            contrastive training. Default is log(1 / 0.07).

    Notes:
        - `stage` controls whether the model runs in pretraining or
          finetuning mode.
        - `mode` is reserved for PyHealth task compatibility and is set to
          "binary".

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models.event_contrastive import EBCLModel
        >>> samples = [
        ...     {
        ...         "patient_id": "p1",
        ...         "visit_id": "v1",
        ...         "pre": [[0.1, 1.0, 0.5], [0.0, 0.0, 0.0]],
        ...         "post": [[0.2, 2.0, 1.0], [0.0, 0.0, 0.0]],
        ...         "pre_mask": [1, 0],
        ...         "post_mask": [1, 0],
        ...         "label": 1,
        ...     }
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "pre": "tensor",
        ...         "post": "tensor",
        ...         "pre_mask": "tensor",
        ...         "post_mask": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ebcl_demo",
        ... )
        >>> model = EBCLModel(
        ...     dataset=dataset,
        ...     num_features=16,
        ...     stage="finetune",
        ...     task="binary",
        ... )
    """

    def __init__(
        self,
        dataset: Optional[object] = None,
        num_features: int = 1000,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_hidden_dim: int = 128,
        dropout: float = 0.1,
        projection_dim: int = 32,
        stage: str = "pretrain",
        task: str = "binary",
        logit_scale_init: float = math.log(1 / 0.07),
    ) -> None:
        """Initializes the EBCL model.

        Args:
            dataset: Optional PyHealth dataset object.
            num_features: Number of distinct feature ids.
            d_model: Token embedding dimension.
            n_heads: Number of attention heads in the transformer.
            n_layers: Number of transformer encoder layers.
            ff_hidden_dim: Feed-forward hidden size inside transformer blocks.
            dropout: Dropout rate.
            projection_dim: Dimension of the contrastive projection space.
            stage: Either "pretrain" or "finetune".
            task: Downstream prediction type. Only "binary" is supported.
            logit_scale_init: Initial log scale used in contrastive learning.
        """
        super().__init__(dataset=dataset)

        if stage not in {"pretrain", "finetune"}:
            raise ValueError("stage must be either 'pretrain' or 'finetune'")
        if task != "binary":
            raise ValueError("Currently only binary classification is supported")

        self.num_features = num_features
        self.d_model = d_model
        self.stage = stage

        # PyHealth expects `mode` to describe the prediction task type.
        self.mode = task

        # Embedding for categorical feature ids.
        self.feature_emb = nn.Embedding(num_features, d_model)

        # Small feed-forward networks for continuous time and value inputs.
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Shared transformer encoder for both pre-event and post-event windows.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Attention-based pooling layer used to compress a sequence into a
        # single vector representation.
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

        # Separate projection heads for contrastive pretraining.
        self.pre_projector = nn.Linear(d_model, projection_dim)
        self.post_projector = nn.Linear(d_model, projection_dim)

        # One-hidden-layer feed-forward classifier for finetuning.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Learnable temperature scale for contrastive logits.
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

    def embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds token triplets into dense vectors.

        Each token is represented as:
            [time_value, feature_id, measurement_value]

        The final embedding is the sum of:
            - time embedding
            - feature embedding
            - value embedding

        Args:
            x: Tensor of shape (batch_size, sequence_length, 3).

        Returns:
            Tensor of shape (batch_size, sequence_length, d_model).
        """
        time_value = x[..., 0:1]
        feature_id = x[..., 1].long().clamp(min=0, max=self.num_features - 1)
        measurement_value = x[..., 2:3]

        return (
            self.time_mlp(time_value)
            + self.feature_emb(feature_id)
            + self.value_mlp(measurement_value)
        )

    def attentive_pool(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Applies attention pooling over a sequence.

        Args:
            hidden_states: Tensor of shape
                (batch_size, sequence_length, d_model).
            mask: Boolean tensor of shape
                (batch_size, sequence_length), where True indicates a valid
                token and False indicates padding.

        Returns:
            Tensor of shape (batch_size, d_model).
        """
        attention_scores = self.attention_pool(hidden_states).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))

        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights,
        )

        pooled = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1),
            dim=1,
        )
        return pooled

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes a sequence into a single representation.

        This method:
            1. Embeds the input tokens
            2. Passes them through the shared transformer encoder
            3. Applies attention pooling

        Args:
            x: Tensor of shape (batch_size, sequence_length, 3).
            mask: Optional boolean tensor of shape
                (batch_size, sequence_length).

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
        else:
            mask = mask.to(dtype=torch.bool, device=x.device)

        hidden_states = self.embed_tokens(x)
        hidden_states = self.transformer(
            hidden_states,
            src_key_padding_mask=~mask,
        )
        return self.attentive_pool(hidden_states, mask)

    def compute_contrastive_loss(
        self,
        z_pre: torch.Tensor,
        z_post: torch.Tensor,
    ) -> torch.Tensor:
        """Computes symmetric contrastive loss for EBCL pretraining.

        Positive pairs are matched pre-event and post-event representations from
        the same sample. All other pairs in the batch act as negatives.

        Args:
            z_pre: Normalized projected pre-event embeddings of shape
                (batch_size, projection_dim).
            z_post: Normalized projected post-event embeddings of shape
                (batch_size, projection_dim).

        Returns:
            A scalar contrastive loss.
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
            logit: Predicted logits of shape (batch_size, 1).
            y_true: Ground-truth binary labels of shape
                (batch_size,) or (batch_size, 1).

        Returns:
            A scalar BCE-with-logits loss.
        """
        y_true = y_true.float().view(-1, 1)
        return F.binary_cross_entropy_with_logits(logit, y_true)

    def get_encoder_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Returns the shared encoder weights for transfer to finetuning.

        Returns:
            A dictionary containing the state dicts of the shared encoder
            components.
        """
        return {
            "feature_emb": self.feature_emb.state_dict(),
            "time_mlp": self.time_mlp.state_dict(),
            "value_mlp": self.value_mlp.state_dict(),
            "transformer": self.transformer.state_dict(),
            "attention_pool": self.attention_pool.state_dict(),
        }

    def load_encoder_state_dict(
        self,
        encoder_state: Dict[str, Dict[str, torch.Tensor]],
    ) -> None:
        """Loads shared encoder weights from a pretrained model.

        Args:
            encoder_state: Dictionary returned by `get_encoder_state_dict()`.
        """
        self.feature_emb.load_state_dict(encoder_state["feature_emb"])
        self.time_mlp.load_state_dict(encoder_state["time_mlp"])
        self.value_mlp.load_state_dict(encoder_state["value_mlp"])
        self.transformer.load_state_dict(encoder_state["transformer"])
        self.attention_pool.load_state_dict(encoder_state["attention_pool"])

    def forward(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Runs a forward pass.

        Pretraining mode expects:
            - pre
            - post
            - optional pre_mask
            - optional post_mask

        Finetuning mode expects:
            - pre
            - optional pre_mask
            - optional label

        Returns:
            In pretraining:
                {
                    "loss": ...,
                    "z_pre": ...,
                    "z_post": ...
                }

            In finetuning:
                {
                    "logit": ...,
                    "y_prob": ...,
                    "loss": ...,      # if label provided
                    "y_true": ...     # if label provided
                }
        """
        pre = kwargs["pre"]
        pre_mask = kwargs.get("pre_mask")
        if pre_mask is not None:
            pre_mask = pre_mask.bool()

        if self.stage == "pretrain":
            post = kwargs["post"]
            post_mask = kwargs.get("post_mask")
            if post_mask is not None:
                post_mask = post_mask.bool()

            pre_embedding = self.encode(pre, pre_mask)
            post_embedding = self.encode(post, post_mask)

            z_pre = F.normalize(self.pre_projector(pre_embedding), dim=-1)
            z_post = F.normalize(self.post_projector(post_embedding), dim=-1)

            loss = self.compute_contrastive_loss(z_pre, z_post)

            return {
                "loss": loss,
                "z_pre": z_pre,
                "z_post": z_post,
            }

        pre_embedding = self.encode(pre, pre_mask)
        logit = self.classifier(pre_embedding)
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