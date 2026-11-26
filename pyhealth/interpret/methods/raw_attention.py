"""Raw attention weights for transformer interpretability.

This module implements the raw-attention baseline: the attention weights of a
single layer, read off directly as token relevance. It is forward-pass only,
gradient-free and class-agnostic, and is the simplest attention baseline that
:class:`~pyhealth.interpret.methods.AttentionRollout` and
:class:`~pyhealth.interpret.methods.CheferRelevance` are compared against.

Paper:
    Chefer, Hila, Shir Gur, and Lior Wolf.
    "Transformer Interpretability Beyond Attention Visualization."
    Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
    Recognition (CVPR), 2021.
    https://arxiv.org/abs/2012.09838
"""

import torch

from pyhealth.models.base_model import BaseModel

from .base_interpreter import BaseInterpreter


class RawAttention(BaseInterpreter):
    """Raw attention weights as a token attribution baseline.

    Uses the attention weights of one layer (the last by default), averaged
    over heads, as each token's relevance. This is the "raw attention"
    baseline of Chefer et al., "Transformer Interpretability Beyond Attention
    Visualization" (2021), https://arxiv.org/abs/2012.09838.

    Unlike :class:`~pyhealth.interpret.methods.AttentionRollout`, which
    composes the attention of every layer, raw attention looks at a single
    layer and ignores how information mixed in the layers below it. Unlike
    :class:`~pyhealth.interpret.methods.CheferRelevance`, it uses no gradients
    and is **class-agnostic**. Comparing the three shows how much the
    cross-layer composition and the gradient weighting add.

    .. note::
        Do not call ``attribute`` inside ``torch.no_grad()``. No backward pass
        is run, but the shared attention-readout plumbing registers a gradient
        hook on the attention tensors during the forward pass, which raises a
        ``RuntimeError`` without autograd.

    This interpreter works with any model that exposes the attention-readout
    methods ``set_attention_hooks``, ``get_attention_layers``, and
    ``get_relevance_tensor`` (currently :class:`~pyhealth.models.Transformer`
    and :class:`~pyhealth.models.StageAttentionNet`). Compatibility is checked
    by duck-typing in ``__init__``.

    The algorithm, per feature key:

    1. Enable attention hooks via ``model.set_attention_hooks(True)`` and run a
       single forward pass (no backward pass).
    2. Take the attention map of layer ``layer`` from
       ``model.get_attention_layers()``.
    3. Fuse heads to get one ``[batch, seq, seq]`` matrix.
    4. Reduce to per-token scores via ``model.get_relevance_tensor()`` (the
       CLS-token row for the Transformer, the last valid visit's row for
       StageAttentionNet), then expand to raw input value shapes.

    With ``head_fusion="mean"`` each row is a softmax average, so the
    per-token relevance sums to 1 before the input-shape expansion.

    Args:
        model (BaseModel): A trained PyHealth model exposing the attention-
            readout methods listed above.
        layer (int): Index of the attention layer to read, as a Python list
            index into the model's layers. Defaults to ``-1``, the last layer,
            which is the baseline used in the paper.
        head_fusion (str): How to combine attention heads: ``"mean"`` (the
            paper's choice) or ``"max"``. Defaults to ``"mean"``.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods import RawAttention
        >>>
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v0",
        ...         "conditions": ["A05B", "A05C", "A06A"],
        ...         "procedures": ["P01", "P02"],
        ...         "label": 1,
        ...     },
        ...     {
        ...         "patient_id": "p0",
        ...         "visit_id": "v1",
        ...         "conditions": ["A05B"],
        ...         "procedures": ["P01"],
        ...         "label": 0,
        ...     },
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples=samples,
        ...     input_schema={"conditions": "sequence", "procedures": "sequence"},
        ...     output_schema={"label": "binary"},
        ...     dataset_name="ehr_example",
        ... )
        >>> model = Transformer(dataset=dataset)
        >>> # ... train the model ...
        >>>
        >>> interpreter = RawAttention(model)  # last layer, mean over heads
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>>
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns dict: {"conditions": tensor, "procedures": tensor}
        >>> print(attributions["conditions"].shape)  # [batch, num_tokens]
        >>>
        >>> # read the first layer instead
        >>> first = RawAttention(model, layer=0).attribute(**batch)
    """

    def __init__(
        self,
        model: BaseModel,
        layer: int = -1,
        head_fusion: str = "mean",
    ):
        if head_fusion not in ("mean", "max"):
            raise ValueError(
                f"Unsupported head_fusion='{head_fusion}'. "
                "Currently supported values: mean, max."
            )
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise TypeError(f"layer must be an int, got {type(layer).__name__}.")

        required_methods = [
            "set_attention_hooks",
            "get_attention_layers",
            "get_relevance_tensor",
        ]
        missing_methods = [m for m in required_methods if not hasattr(model, m)]

        if missing_methods:
            raise TypeError(
                "RawAttention requires a model that exposes the attention "
                "interpretability methods: "
                f"{', '.join(required_methods)}. "
                f"Missing: {', '.join(missing_methods)}."
            )

        super().__init__(model)
        self.layer = layer
        self.head_fusion = head_fusion

    def attribute(
        self,
        target_class_idx: int | None = None,
        **data,
    ) -> dict[str, torch.Tensor]:
        """Compute class-agnostic raw attention attributions.

        Args:
            target_class_idx: Accepted for API compatibility with class-specific
                interpreters. Raw attention is class-agnostic, so this argument
                is ignored.
            **data: Batch input passed directly to the model.

        Returns:
            Dict[str, torch.Tensor]: A dict keyed by the model's feature keys.
                Each value holds the chosen layer's head-fused attention,
                reduced to one score per token by
                ``model.get_relevance_tensor()`` and then expanded to the raw
                input value shape. For flat sequence features this is
                ``[batch, num_tokens]``; for nested sequences the per-visit
                score is replicated across the codes within each visit.
                Scores are non-negative.

        Raises:
            ValueError: If ``layer`` is out of range for a feature's layers.
            RuntimeError: If the model captured no attention map.

        Note:
            Do not call this method inside a ``torch.no_grad()`` context; see
            the class note.
        """

        self.model.set_attention_hooks(True)
        try:
            self.model(**data)
        finally:
            self.model.set_attention_hooks(False)

        attention_layers = self.model.get_attention_layers()
        R = {}

        for feature_key, layers in attention_layers.items():
            if not -len(layers) <= self.layer < len(layers):
                raise ValueError(
                    f"layer={self.layer} is out of range for feature "
                    f"'{feature_key}', which has {len(layers)} attention "
                    "layer(s)."
                )

            attn_map, _ = layers[self.layer]
            if attn_map is None:
                raise RuntimeError(
                    "RawAttention expected attention maps to be captured "
                    f"for feature '{feature_key}', but found None."
                )

            R[feature_key] = self._fuse_heads(attn_map.detach())

        attributions = self.model.get_relevance_tensor(R, **data)
        return self._map_to_input_shapes(attributions, data)

    def _fuse_heads(self, attn_map: torch.Tensor) -> torch.Tensor:
        """Fuse attention heads from [batch, heads, seq, seq] to [batch, seq, seq]."""
        if self.head_fusion == "max":
            return attn_map.max(dim=1).values
        return attn_map.mean(dim=1)

    def _map_to_input_shapes(
        self,
        attributions: dict[str, torch.Tensor],
        data: dict,
    ) -> dict[str, torch.Tensor]:
        """Expand attributions to match raw input value shapes.

        For nested sequences the attention operates on a pooled
        (visit-level) sequence, but downstream consumers (e.g. ablation
        metrics) expect attributions to match the raw input value shape.
        Per-visit relevance scores are replicated across all codes
        within each visit.

        Args:
            attributions: Per-feature attribution tensors returned by
                ``model.get_relevance_tensor()``.
            data: Original ``**data`` kwargs from the dataloader batch.

        Returns:
            Attributions expanded to raw input value shapes where needed.
        """
        result: dict[str, torch.Tensor] = {}
        for key, attr in attributions.items():
            feature = data.get(key)
            if feature is not None:
                if isinstance(feature, torch.Tensor):
                    val = feature
                else:
                    schema = self.model.dataset.input_processors[key].schema()
                    val = feature[schema.index("value")] if "value" in schema else None
                if val is not None and val.dim() > attr.dim():
                    for _ in range(val.dim() - attr.dim()):
                        attr = attr.unsqueeze(-1)
                    attr = attr.expand_as(val)
            result[key] = attr
        return result
