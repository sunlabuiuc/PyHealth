import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel

from .embedding import EmbeddingModel


class EBCL(BaseModel):
    """Event-Based Contrastive Learning model for structured medical time series.

    This implementation adapts the paper's core idea to the standard PyHealth
    model interface:

    - Fine-tuning mode uses the "pre-event" feature streams already present in a
      :class:`~pyhealth.datasets.SampleDataset`.
    - Contrastive mode is enabled when the dataset also includes matching
      ``post_*`` feature streams (for example ``conditions`` and
      ``post_conditions``). The model then learns paired representations around
      an index event using a symmetric InfoNCE / CLIP-style loss.

    The design intentionally follows PyHealth's existing multi-stream EHR
    models: each feature stream is embedded independently, encoded with a
    transformer, pooled to a feature representation, concatenated, and finally
    fused into a patient representation. This makes the model easy to test and
    reuse with existing PyHealth tasks while still preserving the paper's
    event-centered contrastive objective.

    Args:
        dataset: Processed sample dataset containing one label key and one or
            more input feature keys.
        embedding_dim: Shared embedding dimension for the input streams.
        hidden_dim: Hidden dimension of the pooled patient representation.
        projection_dim: Output dimension of the contrastive projection head.
        heads: Number of attention heads in each feature transformer.
        num_layers: Number of transformer layers per feature stream.
        dropout: Dropout applied in the transformer and MLP heads.
        temperature: Initial contrastive temperature.
        contrastive_weight: Multiplier for the contrastive loss when combined
            with supervised fine-tuning loss.
        post_prefix: Prefix used to identify post-event paired features.
        max_seq_len: Maximum number of steps kept per feature stream.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> samples = [
        ...     {
        ...         "patient_id": "patient-0",
        ...         "visit_id": "visit-0",
        ...         "conditions": ["A", "B", "C"],
        ...         "labs": [1.0, 2.0, 3.0],
        ...         "post_conditions": ["B", "C"],
        ...         "post_labs": [2.0, 3.0, 4.0],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "patient-1",
        ...         "visit_id": "visit-1",
        ...         "conditions": ["D", "E"],
        ...         "labs": [0.5, 1.5, 2.5],
        ...         "post_conditions": ["E"],
        ...         "post_labs": [1.0, 1.5, 2.0],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={
        ...         "conditions": "sequence",
        ...         "labs": "tensor",
        ...         "post_conditions": "sequence",
        ...         "post_labs": "tensor",
        ...     },
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ebcl_demo",
        ... )
        >>> model = EBCL(dataset=dataset)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        >>> output = model(**batch)
        >>> "contrastive_loss" in output and "supervised_loss" in output
        True
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        temperature: float = 0.07,
        contrastive_weight: float = 1.0,
        post_prefix: str = "post_",
        max_seq_len: int = 512,
    ):
        super().__init__(dataset=dataset)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim
        self.heads = heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.contrastive_weight = contrastive_weight
        self.post_prefix = post_prefix
        self.max_seq_len = max_seq_len

        if embedding_dim % heads != 0:
            raise ValueError("embedding_dim must be divisible by heads")

        assert len(self.label_keys) == 1, "Only one label key is supported if EBCL is initialized"
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        raw_feature_keys = list(self.feature_keys)
        self.raw_feature_keys = raw_feature_keys
        self.post_feature_keys: Dict[str, str] = {}
        self.base_feature_keys = []
        for feature_key in raw_feature_keys:
            if feature_key.startswith(self.post_prefix):
                self.post_feature_keys[feature_key[len(self.post_prefix):]] = feature_key
            else:
                self.base_feature_keys.append(feature_key)
        self.feature_keys = self.base_feature_keys

        if len(self.base_feature_keys) == 0:
            raise ValueError("EBCL requires at least one pre-event feature key")

        self.embedding_model = EmbeddingModel(dataset, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        self.feature_encoder = nn.ModuleDict()
        self.feature_projector = nn.ModuleDict()
        encoder_layer = lambda: nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=heads,
            dim_feedforward=max(hidden_dim, embedding_dim) * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        for feature_key in self.base_feature_keys:
            self.feature_encoder[feature_key] = nn.TransformerEncoder(
                encoder_layer(),
                num_layers=num_layers,
                norm=nn.LayerNorm(embedding_dim),
            )
            self.feature_projector[feature_key] = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        fused_dim = len(self.base_feature_keys) * hidden_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.fc = nn.Linear(hidden_dim, self.get_output_size())
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / max(temperature, 1e-6)), dtype=torch.float32)
        )

    def _get_branch_feature_map(self, use_post: bool = False) -> Dict[str, str]:
        if not use_post:
            return {feature_key: feature_key for feature_key in self.base_feature_keys}
        return {
            feature_key: self.post_feature_keys[feature_key]
            for feature_key in self.base_feature_keys
            if feature_key in self.post_feature_keys
        }

    def _prepare_branch_inputs(
        self,
        feature_map: Dict[str, str],
        kwargs: Dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
    ) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        inputs: Dict[str, torch.Tensor] = {}
        masks: Dict[str, torch.Tensor] = {}

        for actual_key in feature_map.values():
            feature = kwargs[actual_key]
            if isinstance(feature, torch.Tensor):
                feature = (feature,)

            schema = self.dataset.input_processors[actual_key].schema()
            value = feature[schema.index("value")] if "value" in schema else None
            mask = feature[schema.index("mask")] if "mask" in schema else None

            if value is None:
                raise ValueError(
                    f"Feature '{actual_key}' must contain 'value' in the schema."
                )

            inputs[actual_key] = value
            if mask is not None:
                masks[actual_key] = mask

        embedded, out_masks = self.embedding_model(
            inputs,
            masks=masks if masks else None,
            output_mask=True,
        )
        return embedded, out_masks

    def _collapse_sequence(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        original_x_dim = x.dim()
        while x.dim() > 3:
            x = x.sum(dim=2)
            if mask is not None and mask.dim() > 2:
                mask = mask.any(dim=-1)

        if x.dim() == 2:
            x = x.unsqueeze(1)

        if mask is None:
            mask = torch.any(torch.abs(x) > 0, dim=-1)
        else:
            mask = mask.to(self.device)
            while mask.dim() > 2:
                mask = mask.any(dim=-1)
            if mask.dim() == x.dim():
                mask = mask.any(dim=-1)
            if mask.dim() == 1:
                mask = mask.unsqueeze(1)
            elif original_x_dim == 2 and mask.dim() == 2:
                # Dense tensor features are a single event with multiple
                # dimensions, but EmbeddingModel emits a per-dimension mask.
                mask = mask.any(dim=-1, keepdim=True)
            elif mask.size(1) != x.size(1):
                # After pooling nested axes, reduce any stale mask axis to the
                # active sequence dimension.
                mask = mask.any(dim=-1, keepdim=True)

        if x.size(1) > self.max_seq_len:
            x = x[:, : self.max_seq_len, :]
            mask = mask[:, : self.max_seq_len]

        mask = mask.bool()
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True
        return x, mask

    def _encode_feature(
        self,
        feature_key: str,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=self.device)
        x = x + self.position_embedding(positions).unsqueeze(0)
        encoded = self.feature_encoder[feature_key](
            x,
            src_key_padding_mask=~mask,
        )
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (encoded * mask.unsqueeze(-1)).sum(dim=1) / denom
        return self.feature_projector[feature_key](pooled)

    def _encode_branch(
        self,
        feature_map: Dict[str, str],
        kwargs: Dict[str, torch.Tensor | tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded, masks = self._prepare_branch_inputs(feature_map, kwargs)
        feature_emb = []

        for base_key, actual_key in feature_map.items():
            x = embedded[actual_key]
            mask = masks.get(actual_key)
            x, mask = self._collapse_sequence(x, mask)
            feature_emb.append(self._encode_feature(base_key, x, mask))

        patient_emb = torch.cat(feature_emb, dim=-1)
        patient_emb = self.fusion(patient_emb)
        projection = F.normalize(self.projection_head(patient_emb), dim=-1)
        return patient_emb, projection

    def _compute_contrastive_loss(
        self,
        pre_projection: torch.Tensor,
        post_projection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = logit_scale * pre_projection @ post_projection.transpose(0, 1)
        labels = torch.arange(logits.size(0), device=self.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.transpose(0, 1), labels)
        )
        return loss, logits

    def forward(
        self,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        pre_feature_map = self._get_branch_feature_map(use_post=False)
        patient_emb, pre_projection = self._encode_branch(pre_feature_map, kwargs)

        logits = self.fc(patient_emb)
        y_prob = self.prepare_y_prob(logits)
        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device)
            supervised_loss = self.get_loss_function()(logits, y_true)
            results["supervised_loss"] = supervised_loss
            results["y_true"] = y_true
            results["loss"] = supervised_loss

        has_any_post = any(key in kwargs for key in self.post_feature_keys.values())
        has_all_post = all(key in kwargs for key in self.post_feature_keys.values()) and len(
            self.post_feature_keys
        ) == len(self.base_feature_keys)

        if has_any_post and not has_all_post:
            missing = sorted(
                key for key in self.post_feature_keys.values() if key not in kwargs
            )
            raise ValueError(
                "EBCL received an incomplete post-event batch. Missing keys: "
                + ", ".join(missing)
            )

        if has_all_post:
            post_feature_map = self._get_branch_feature_map(use_post=True)
            post_emb, post_projection = self._encode_branch(post_feature_map, kwargs)
            contrastive_loss, similarity = self._compute_contrastive_loss(
                pre_projection,
                post_projection,
            )
            results["contrastive_loss"] = contrastive_loss
            results["contrastive_logits"] = similarity
            results["post_embed"] = post_emb
            if "loss" in results:
                results["loss"] = results["loss"] + self.contrastive_weight * contrastive_loss
            else:
                results["loss"] = self.contrastive_weight * contrastive_loss

        if kwargs.get("embed", False):
            results["embed"] = patient_emb
        return results
