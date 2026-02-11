"""Chefer's gradient-weighted attention relevance propagation.

This module implements the Chefer et al. relevance propagation method for
explaining transformer-family model predictions.  It relies on the
:class:`~pyhealth.interpret.api.CheferInterpretable` interface — any model
that implements that interface is automatically supported.

Paper:
    Chefer, Hila, Shir Gur, and Lior Wolf.
    "Generic Attention-model Explainability for Interpreting Bi-Modal and
    Encoder-Decoder Transformers."
    Proceedings of the IEEE/CVF International Conference on Computer Vision
    (ICCV), 2021.
"""

from typing import Dict, Optional, cast

import torch
import torch.nn.functional as F

from pyhealth.interpret.api import CheferInterpretable
from pyhealth.models.base_model import BaseModel

from .base_interpreter import BaseInterpreter, _CheferInterpretableModel


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def apply_self_attention_rules(R_ss, cam_ss):
    """Apply Chefer's self-attention rules for relevance propagation.

    Args:
        R_ss: Relevance matrix [batch, seq_len, seq_len].
        cam_ss: Attention weight matrix [batch, seq_len, seq_len].

    Returns:
        Updated relevance matrix after propagating through attention layer.
    """
    return torch.matmul(cam_ss, R_ss)


def avg_heads(cam, grad):
    """Average attention scores weighted by gradients across heads.

    Args:
        cam: Attention weights [batch, heads, seq_len, seq_len] or [batch, seq_len, seq_len].
        grad: Gradients w.r.t. attention weights. Same shape as cam.

    Returns:
        Gradient-weighted attention [batch, seq_len, seq_len].
    """
    if len(cam.size()) < 4 and len(grad.size()) < 4:
        return (grad * cam).clamp(min=0)
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=1)
    return cam.clone()


# ---------------------------------------------------------------------------
# Main interpreter
# ---------------------------------------------------------------------------

class CheferRelevance(BaseInterpreter):
    """Chefer's gradient-weighted attention method for transformer interpretability.

    This interpreter works with **any** model that implements the
    :class:`~pyhealth.interpret.api.CheferInterpretable` interface, which
    currently includes:

    * :class:`~pyhealth.models.Transformer`
    * :class:`~pyhealth.models.StageAttentionNet`

    The algorithm:

    1. Enable attention hooks via ``model.set_attention_hooks(True)``.
    2. Forward pass → capture attention maps and register gradient hooks.
    3. Backward pass from a one-hot target class.
    4. Retrieve ``(attn_map, attn_grad)`` pairs via ``model.get_attention_layers()``.
    5. Propagate relevance: ``R += clamp(attn * grad, min=0) @ R``.
    6. Reduce ``R`` to per-token vectors via ``model.get_relevance_tensor()``.

    Steps 1, 4 and 6 are delegated to the model through the
    ``CheferInterpretable`` interface, making this class fully
    model-agnostic.

    Args:
        model (BaseModel): A trained PyHealth model that implements
            :class:`~pyhealth.interpret.api.CheferInterpretable`.

    Example:
        >>> from pyhealth.datasets import create_sample_dataset, get_dataloader
        >>> from pyhealth.models import Transformer
        >>> from pyhealth.interpret.methods import CheferRelevance
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
        >>> interpreter = CheferRelevance(model)
        >>> batch = next(iter(get_dataloader(dataset, batch_size=2)))
        >>>
        >>> # Default: attribute to predicted class
        >>> attributions = interpreter.attribute(**batch)
        >>> # Returns dict: {"conditions": tensor, "procedures": tensor}
        >>> print(attributions["conditions"].shape)  # [batch, num_tokens]
        >>>
        >>> # Optional: attribute to a specific class (e.g., class 1)
        >>> attributions = interpreter.attribute(class_index=1, **batch)
    """

    def __init__(self, model: _CheferInterpretableModel):
        super().__init__(model)
        self.model = cast(_CheferInterpretableModel, model)

        if not isinstance(model, CheferInterpretable):
            raise ValueError(
                f"CheferRelevance requires a model implementing "
                f"CheferInterpretable, got {type(model).__name__}."
            )

    def attribute(
        self,
        class_index: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute relevance scores for each input token.

        Args:
            class_index: Target class index to compute attribution for.
                If None (default), uses the model's predicted class.
            **data: Input data from dataloader batch containing feature
                keys and label key.

        Returns:
            Dict[str, torch.Tensor]: Dictionary keyed by feature keys,
                where each tensor has shape ``[batch, seq_len]`` with
                per-token attribution scores.
        """
        # --- 1. Forward with attention hooks enabled ---
        self.model.set_attention_hooks(True)
        try:
            logits = self.model(**data)["logit"]
        finally:
            self.model.set_attention_hooks(False)

        # --- 2. Backward from target class ---
        if class_index is None:
            class_index_t = torch.argmax(logits, dim=-1)
        elif isinstance(class_index, int):
            class_index_t = torch.tensor(class_index)
        else:
            class_index_t = class_index

        one_hot = F.one_hot(
            class_index_t.detach().clone(), logits.size(1)
        ).float()
        one_hot = one_hot.requires_grad_(True)
        scalar = torch.sum(one_hot.to(logits.device) * logits)
        self.model.zero_grad()
        scalar.backward(retain_graph=True)

        # --- 3. Retrieve (attn_map, attn_grad) pairs per feature key ---
        attention_layers = self.model.get_attention_layers()

        batch_size = logits.shape[0]
        device = logits.device

        # --- 4. Relevance propagation per feature key ---
        R_dict: dict[str, torch.Tensor] = {}
        for key, layers in attention_layers.items():
            num_tokens = layers[0][0].shape[-1]
            R = (
                torch.eye(num_tokens, device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
            for cam, grad in layers:
                cam = avg_heads(cam, grad)
                R = R + apply_self_attention_rules(R, cam).detach()
            R_dict[key] = R

        # --- 5. Reduce R matrices to per-token vectors ---
        return self.model.get_relevance_tensor(R_dict, **data)

    # ------------------------------------------------------------------
    # Backward compatibility aliases
    # ------------------------------------------------------------------

    def get_relevance_matrix(self, **data):
        """Alias for attribute(). Deprecated."""
        return self.attribute(**data)


# ======================================================================
# LEGACY REFERENCE IMPLEMENTATIONS
# ======================================================================
# The functions below are the original model-specific implementations
# that existed before the CheferInterpretable API was introduced. They
# are kept here ONLY as a reference for future developers and are NOT
# called by any production code.  They may be removed in a future
# release.
#
# For ViT models, _reference_attribute_vit is the only implementation
# until ViT models implement CheferInterpretable.
# ======================================================================

def _reference_attribute_transformer(
    model,
    class_index=None,
    **data,
) -> Dict[str, torch.Tensor]:
    """[REFERENCE ONLY] Original Transformer-specific Chefer attribution.

    This was the body of ``CheferRelevance._attribute_transformer()``
    before the CheferInterpretable API was introduced.  It accesses
    model internals (``model.transformer[key].transformer``) directly.
    """
    data["register_hook"] = True

    logits = model(**data)["logit"]
    if class_index is None:
        class_index = torch.argmax(logits, dim=-1)

    if isinstance(class_index, torch.Tensor):
        one_hot = F.one_hot(class_index.detach().clone(), logits.size()[1]).float()
    else:
        one_hot = F.one_hot(torch.tensor(class_index), logits.size()[1]).float()
    one_hot = one_hot.requires_grad_(True)
    one_hot = torch.sum(one_hot.to(logits.device) * logits)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    feature_keys = model.feature_keys
    num_tokens = {}
    for key in feature_keys:
        feature_transformer = model.transformer[key].transformer
        for block in feature_transformer:
            num_tokens[key] = block.attention.get_attn_map().shape[-1]

    batch_size = logits.shape[0]
    attn = {}
    for key in feature_keys:
        R = (
            torch.eye(num_tokens[key])
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(logits.device)
        )
        for blk in model.transformer[key].transformer:
            grad = blk.attention.get_attn_grad()
            cam = blk.attention.get_attn_map()
            cam = avg_heads(cam, grad)
            R += apply_self_attention_rules(R, cam).detach()
        attn[key] = R[:, 0]

    return attn


def _reference_attribute_stageattn(
    model,
    class_index=None,
    **data,
) -> Dict[str, torch.Tensor]:
    """[REFERENCE ONLY] Original StageAttentionNet-specific Chefer attribution.

    This was the body of ``CheferRelevance._attribute_stageattn()``
    before the CheferInterpretable API was introduced.  It accesses
    model internals (``model.stagenet[key]``, ``model.embedding_model``)
    directly.
    """
    data["register_attn_hook"] = True

    logits = model(**data)["logit"]
    if class_index is None:
        class_index = torch.argmax(logits, dim=-1)

    if isinstance(class_index, torch.Tensor):
        one_hot = F.one_hot(class_index.detach().clone(), logits.size()[1]).float()
    else:
        one_hot = F.one_hot(torch.tensor(class_index), logits.size()[1]).float()
    one_hot = one_hot.requires_grad_(True)
    one_hot = torch.sum(one_hot.to(logits.device) * logits)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    batch_size = logits.shape[0]
    feature_keys = model.feature_keys
    attn = {}

    for key in feature_keys:
        layer = model.stagenet[key]
        cam = layer.get_attn_map()
        grad = layer.get_attn_grad()
        num_tokens = cam.shape[-1]

        R = (
            torch.eye(num_tokens)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
            .to(logits.device)
        )
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R, cam).detach()

        feature = data[key]
        if isinstance(feature, tuple) and len(feature) == 2:
            _, x_val = feature
        else:
            x_val = feature

        embedded = model.embedding_model({key: x_val})
        emb = embedded[key]
        if emb.dim() == 4:
            emb = emb.sum(dim=2)
        mask = (emb.sum(dim=-1) != 0).long().to(logits.device)

        last_idx = mask.sum(dim=1) - 1
        attn[key] = R[torch.arange(batch_size, device=logits.device), last_idx]

    return attn


def _reference_attribute_vit(
    model,
    interpolate: bool = True,
    class_index=None,
    **data,
) -> Dict[str, torch.Tensor]:
    """[REFERENCE ONLY] Original ViT-specific Chefer attribution.

    ViT models do not yet implement CheferInterpretable.  This code
    shows the ViT-specific flow that will be needed when ViT support is
    added to the unified API.
    """
    feature_key = model.feature_keys[0]
    x = data.get(feature_key)
    if x is None:
        raise ValueError(
            f"Expected feature key '{feature_key}' in data. "
            f"Available keys: {list(data.keys())}"
        )

    x = x.to(model.device)
    input_size = x.shape[-1]

    model.zero_grad()
    logits, attention_maps = model.forward_with_attention(x, register_hook=True)

    target_class = class_index
    if target_class is None:
        target_class = logits.argmax(dim=-1)

    one_hot = torch.zeros_like(logits)
    if isinstance(target_class, int):
        one_hot[:, target_class] = 1
    else:
        if target_class.dim() == 0:
            target_class = target_class.unsqueeze(0)
        one_hot.scatter_(1, target_class.unsqueeze(1), 1)

    one_hot = one_hot.requires_grad_(True)
    (logits * one_hot).sum().backward(retain_graph=True)

    attention_gradients = model.get_attention_gradients()
    batch_size = attention_maps[0].shape[0]
    num_tokens = attention_maps[0].shape[-1]
    device = attention_maps[0].device

    R = torch.eye(num_tokens, device=device)
    R = R.unsqueeze(0).expand(batch_size, -1, -1).clone()

    for attn, grad in zip(attention_maps, attention_gradients):
        cam = avg_heads(attn, grad)
        R = R + apply_self_attention_rules(R.detach(), cam.detach())

    patches_attr = R[:, 0, 1:]

    h_patches, w_patches = model.get_num_patches(input_size)
    attr_map = patches_attr.reshape(batch_size, 1, h_patches, w_patches)

    if interpolate:
        attr_map = F.interpolate(
            attr_map,
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )

    return {feature_key: attr_map}