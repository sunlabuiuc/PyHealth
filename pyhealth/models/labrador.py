from typing import Dict, Optional, Tuple, cast

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset
from pyhealth.models import BaseModel


class LabradorModel(BaseModel):
    """Transformer-based model for tabular lab code/value inputs.

    The model consumes two aligned feature streams:
    1) categorical lab codes (token ids)
    2) continuous lab values

    Architecture:
    - code embedding
    - value projection
    - additive fusion -> linear -> ReLU -> LayerNorm
    - Transformer encoder (no positional encoding)
    - mean pooling over lab dimension
    - MLP classifier head

    Args:
        dataset: The dataset used by PyHealth trainers.
        code_feature_key: Input feature key containing lab code tokens.
        value_feature_key: Input feature key containing lab values.
        embed_dim: Hidden size for embeddings and transformer blocks.
        num_heads: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dropout: Dropout used by the transformer encoder layer.
        ff_hidden_dim: Feed-forward width inside each transformer layer.
        classifier_hidden_dim: Hidden width of classifier MLP head.

    Examples:
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.models import LabradorModel
        >>> samples = [
        ...     {
        ...         "patient_id": "p-0",
        ...         "visit_id": "v-0",
        ...         "lab_codes": ["lab-1", "lab-2"],
        ...         "lab_values": [0.2, 0.8],
        ...         "label": 1,
        ...     }
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"lab_codes": "sequence", "lab_values": "tensor"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="labrador_demo",
        ... )
        >>> model = LabradorModel(
        ...     dataset=dataset,
        ...     code_feature_key="lab_codes",
        ...     value_feature_key="lab_values",
        ... )
    """

    def __init__(
        self,
        dataset: SampleDataset,
        code_feature_key: str,
        value_feature_key: str,
        embed_dim: int = 128,
        num_heads: int = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        ff_hidden_dim: Optional[int] = None,
        classifier_hidden_dim: Optional[int] = None,
    ):
        super().__init__(dataset=dataset)

        assert len(self.label_keys) == 1, "Only one label key is supported"
        self.label_key = self.label_keys[0]

        if code_feature_key not in self.feature_keys:
            raise ValueError(
                f"code_feature_key='{code_feature_key}' not found in dataset input schema: "
                f"{self.feature_keys}"
            )
        if value_feature_key not in self.feature_keys:
            raise ValueError(
                f"value_feature_key='{value_feature_key}' not found in dataset input schema: "
                f"{self.feature_keys}"
            )

        self.code_feature_key = code_feature_key
        self.value_feature_key = value_feature_key
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        num_labs = self.dataset.input_processors[self.code_feature_key].size()
        if num_labs is None:
            raise ValueError(
                "LabradorModel requires a tokenized categorical code feature with known vocabulary size. "
                f"Feature '{self.code_feature_key}' returned size=None."
            )
        self.num_labs = int(num_labs)

        ff_hidden_dim = ff_hidden_dim or embed_dim
        classifier_hidden_dim = classifier_hidden_dim or embed_dim

        self.code_embedding = nn.Embedding(self.num_labs, embed_dim)
        self.value_projection = nn.Linear(1, embed_dim)
        self.value_fusion = nn.Linear(embed_dim, embed_dim)
        self.value_act = nn.ReLU()
        self.value_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        output_size = self.get_output_size()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Linear(classifier_hidden_dim, output_size),
        )

    def _extract_value_and_mask(
        self, feature_key: str, feature: torch.Tensor | Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Extract value tensor and optional mask from a processed feature.

        Args:
            feature_key: Name of the dataset feature being decoded.
            feature: Processor output for one feature. This may be either a
                tensor or a tuple containing tensors such as value and mask.

        Returns:
            A tuple ``(value, mask)`` where ``value`` is the feature tensor and
            ``mask`` is the optional validity mask if provided by the processor.

        Raises:
            ValueError: If the processor schema does not contain a ``value``
                field.
        """
        if isinstance(feature, torch.Tensor):
            feature_tuple: Tuple[torch.Tensor, ...] = (feature,)
        else:
            feature_tuple = feature

        schema = self.dataset.input_processors[feature_key].schema()

        value = feature_tuple[schema.index("value")] if "value" in schema else None
        if value is None:
            raise ValueError(
                f"Feature '{feature_key}' must contain 'value' in processor schema."
            )

        mask = feature_tuple[schema.index("mask")] if "mask" in schema else None
        if len(feature_tuple) == len(schema) + 1 and mask is None:
            mask = feature_tuple[-1]

        return value, mask

    @staticmethod
    def _ensure_2d(tensor: torch.Tensor, name: str) -> torch.Tensor:
        """Normalize tensor to ``[batch, num_labs]`` for aligned lab streams.

        Args:
            tensor: Input tensor representing lab codes, values, or masks.
            name: Human-readable tensor name used in error messages.

        Returns:
            A 2D tensor of shape ``[batch, num_labs]``.

        Raises:
            ValueError: If the input tensor cannot be interpreted as a 2D lab
                matrix.
        """
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() == 3 and tensor.size(-1) == 1:
            return tensor.squeeze(-1)
        raise ValueError(
            f"Expected {name} to have shape [batch, num_labs] (or [..., 1]), got {tuple(tensor.shape)}"
        )

    @staticmethod
    def _mean_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply masked mean pooling over the lab dimension.

        Args:
            x: Token representations of shape ``[batch, num_labs, embed_dim]``.
            mask: Float mask of shape ``[batch, num_labs]`` indicating valid
                lab positions.

        Returns:
            Pooled patient embeddings of shape ``[batch, embed_dim]``.
        """
        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        **kwargs: torch.Tensor | Tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            **kwargs: Keyword arguments containing the code feature,
                value feature, optional label, and optional ``embed`` flag.
                The two feature tensors must describe aligned lab sequences for
                the same samples.

        Returns:
            A dictionary containing model outputs. This always includes:
                - ``logit``: classification logits.
                - ``y_prob``: predicted probabilities.
            When labels are provided, the dictionary also includes:
                - ``loss``: supervised task loss.
                - ``y_true``: ground-truth labels.
            When ``embed=True`` is passed, the dictionary also includes:
                - ``embed``: pooled patient embedding.

        Raises:
            ValueError: If code and value features do not have matching shapes.
        """
        code_values, code_mask = self._extract_value_and_mask(
            self.code_feature_key, kwargs[self.code_feature_key]
        )
        lab_values, value_mask = self._extract_value_and_mask(
            self.value_feature_key, kwargs[self.value_feature_key]
        )

        codes = self._ensure_2d(code_values.to(self.device), "code feature").long()
        values = self._ensure_2d(lab_values.to(self.device), "value feature").float()

        if codes.shape != values.shape:
            raise ValueError(
                f"Code/value feature shapes must match, got codes={tuple(codes.shape)} "
                f"and values={tuple(values.shape)}"
            )

        if code_mask is not None:
            mask = self._ensure_2d(code_mask.to(self.device), "code mask").bool()
        elif value_mask is not None:
            mask = self._ensure_2d(value_mask.to(self.device), "value mask").bool()
        else:
            mask = codes != 0

        # Avoid all-masked rows to keep transformer behavior stable.
        invalid_rows = ~mask.any(dim=1)
        if invalid_rows.any():
            mask[invalid_rows, 0] = True

        code_emb = self.code_embedding(codes)
        value_emb = self.value_projection(values.unsqueeze(-1))

        x = code_emb + value_emb
        x = self.value_fusion(x)
        x = self.value_act(x)
        x = self.value_norm(x)

        x = self.transformer(x, src_key_padding_mask=~mask)
        patient_emb = self._mean_pool(x, mask.float())
        logits = self.classifier(patient_emb)
        y_prob = self.prepare_y_prob(logits)

        results: Dict[str, torch.Tensor] = {
            "logit": logits,
            "y_prob": y_prob,
        }

        if self.label_key in kwargs:
            y_true = cast(torch.Tensor, kwargs[self.label_key]).to(self.device)
            loss = self.get_loss_function()(logits, y_true)
            results["loss"] = loss
            results["y_true"] = y_true

        if kwargs.get("embed", False):
            results["embed"] = patient_emb

        return results