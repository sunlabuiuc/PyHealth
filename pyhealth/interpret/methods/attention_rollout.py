# Author: Felipe Amaral Bonchristiano
# NetID: felipea5
# Description: Attention rollout interpretability method implementation for PyHealth 2.0

from typing import Dict, Optional

import torch

from pyhealth.models.base_model import BaseModel
from .base_interpreter import BaseInterpreter


class AttentionRollout(BaseInterpreter):
    """Attention rollout for transformer interpretability.

    Implements the canonical attention rollout method of Abnar & Zuidema,
    "Quantifying Attention Flow in Transformers" (2020),
    https://arxiv.org/abs/2005.00928.

    Unlike :class:`~pyhealth.interpret.methods.CheferRelevance`, which is
    gradient-weighted and class-specific, rollout is **forward-pass only**,
    **gradient-free**, and **class-agnostic**: it quantifies how attention
    propagates information across layers, independent of any target class.
    It serves as the standard baseline that gradient-based attention methods
    are compared against.

    This interpreter works with any model that exposes the attention-readout
    methods ``set_attention_hooks``, ``get_attention_layers``, and
    ``get_relevance_tensor`` (currently :class:`~pyhealth.models.Transformer`
    and :class:`~pyhealth.models.StageAttentionNet`). Compatibility is checked
    by duck-typing in ``__init__`` rather than by requiring a named interface,
    since these methods are general attention readout and not specific to any
    one method.

    The algorithm, per feature key:

    1. Enable attention hooks via ``model.set_attention_hooks(True)`` and run a
       single forward pass (no backward pass).
    2. Retrieve per-layer attention maps via ``model.get_attention_layers()``,
       discarding the gradient element of each ``(attn_map, attn_grad)`` pair.
    3. Fuse heads (mean) to get one ``[batch, seq, seq]`` matrix per layer.
    4. Account for residual connections: ``A_hat = 0.5 * (A + I)``.
    5. Compose layers by matrix product: ``rollout = A_hat_L @ ... @ A_hat_1``.
    6. Reduce to per-token scores via ``model.get_relevance_tensor()``, then
       expand to raw input value shapes.

    Because each ``A_hat`` is row-stochastic, so is their product; the
    per-token relevance therefore forms a distribution over tokens (sums to 1
    before the input-shape expansion).

    Args:
        model (BaseModel): A trained PyHealth model exposing the attention-
            readout methods listed above.
        head_fusion (str): How to combine attention heads into a single matrix
            per layer. Currently only ``"mean"`` is supported (the canonical
            choice from the paper). Defaults to ``"mean"``.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods import AttentionRollout
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
        >>> interpreter = AttentionRollout(model)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>>
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns dict: {"conditions": tensor, "procedures": tensor}
        >>> print(attributions["conditions"].shape)  # [batch, num_tokens]
        >>>
        >>> # target_class_idx is accepted but ignored (rollout is class-agnostic)
        >>> same = interpreter.attribute(target_class_idx=1, **batch)
    """
        
    def __init__(self, model: BaseModel, head_fusion: str = "mean"):
        if head_fusion != "mean":
            raise ValueError(
                f"Unsupported head_fusion='{head_fusion}'. "
                "Currently supported values: mean."
            )

        required_methods = [
            "set_attention_hooks",
            "get_attention_layers",
            "get_relevance_tensor",
        ]
        missing_methods = [m for m in required_methods if not hasattr(model, m)]

        if missing_methods:
            raise TypeError(
                "AttentionRollout requires a model that exposes the attention "
                "interpretability methods: "
                f"{', '.join(required_methods)}. "
                f"Missing: {', '.join(missing_methods)}."
            )
        
        super().__init__(model)
        self.head_fusion = head_fusion


    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute class-agnostic attention rollout attributions.

        Args:
            target_class_idx: Accepted for API compatibility with class-specific
                interpreters. Attention rollout is class-agnostic, so this argument
                is ignored.
            **data: Batch input passed directly to the model.

        Returns:
            Dict[str, torch.Tensor]: A dict keyed by the model's feature keys.
                Each value holds the rollout relevance for that feature — the
                CLS-token row of the composed attention-rollout matrix, reduced
                to one score per token by ``model.get_relevance_tensor()`` and
                then expanded to the raw input value shape by
                ``_map_to_input_shapes``. For flat sequence features this is
                ``[batch, num_tokens]``; for nested sequences the per-visit
                score is replicated across the codes within each visit.
                Scores are non-negative and, before the input-shape expansion,
                sum to 1 across tokens (a consequence of composing
                row-stochastic matrices).
        """

        self.model.set_attention_hooks(True)
        try:
            self.model(**data)
        finally:
            self.model.set_attention_hooks(False)

        attention_layers = self.model.get_attention_layers()
        R = {}

        for feature_key, layers in attention_layers.items():
            rollout = None

            for attn_map, _ in layers:
                if attn_map is None:
                    raise RuntimeError(
                        "AttentionRollout expected attention maps to be captured "
                        f"for feature '{feature_key}', but found None."
                    )

                attn = self._fuse_heads(attn_map)
                attn = self._add_residual(attn)

                if rollout is None:
                    batch_size, seq_len, _ = attn.shape
                    rollout = torch.eye(
                        seq_len,
                        device=attn.device,
                        dtype=attn.dtype,
                    )
                    rollout = rollout.unsqueeze(0).expand(
                        batch_size,
                        seq_len,
                        seq_len,
                    )

                rollout = torch.bmm(attn, rollout)

            if rollout is None:
                raise RuntimeError(
                    "AttentionRollout expected at least one attention layer "
                    f"for feature '{feature_key}', but found none."
                )

            R[feature_key] = rollout

        attributions = self.model.get_relevance_tensor(R, **data)
        return self._map_to_input_shapes(attributions, data)
        
    def _fuse_heads(self, attn_map: torch.Tensor) -> torch.Tensor:
        """Fuse attention heads from [batch, heads, seq, seq] to [batch, seq, seq]."""

        if self.head_fusion == "mean":
            return attn_map.mean(dim=1)
        
        raise ValueError(
            f"Unsupported head_fusion='{self.head_fusion}'. "
            "Currently supported values: mean."
        )
     
    def _map_to_input_shapes(
        self,
        attributions: Dict[str, torch.Tensor],
        data: dict,
    ) -> Dict[str, torch.Tensor]:
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
        result: Dict[str, torch.Tensor] = {}
        for key, attr in attributions.items():
            feature = data.get(key)
            if feature is not None:
                if isinstance(feature, torch.Tensor):
                    val = feature
                else:
                    schema = self.model.dataset.input_processors[key].schema()
                    val = (
                        feature[schema.index("value")]
                        if "value" in schema
                        else None
                    )
                if val is not None and val.dim() > attr.dim():
                    for _ in range(val.dim() - attr.dim()):
                        attr = attr.unsqueeze(-1)
                    attr = attr.expand_as(val)
            result[key] = attr
        return result
    
    @staticmethod
    def _add_residual(attn: torch.Tensor) -> torch.Tensor:
        """
        Add canonical rollout residual connection: 0.5 * (A + I).
        0.5 * (A + I) stays row-stochastic only because A is (soft-max ouput).
        """

        batch, seq_len, _ = attn.shape
        identity = torch.eye(
            seq_len,
            device=attn.device,
            dtype=attn.dtype,
        ).unsqueeze(0)
        return 0.5 * (attn + identity)