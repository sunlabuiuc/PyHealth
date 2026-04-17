"""Layer-wise Relevance Propagation (LRP) for PyHealth models.

Implements epsilon-rule and alphabeta-rule LRP for computing feature
attributions. Supports embedding-based models (discrete medical codes)
and CNN-based models (images).

References:
    Binder et al., "Layer-wise Relevance Propagation for Neural Networks
    with Local Renormalization Layers", arXiv:1604.00825, 2016.
"""

import contextlib
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal

from pyhealth.models import BaseModel
from pyhealth.interpret.api import Interpretable
from pyhealth.interpret.methods.base_interpreter import BaseInterpreter
from pyhealth.interpret.methods.lrp_base import (
    stabilize_denominator,
    LRPHandlerRegistry,
    LinearLRPHandler,
    ReLULRPHandler,
    Conv2dLRPHandler,
    MaxPool2dLRPHandler,
    AvgPool2dLRPHandler,
    AdaptiveAvgPool2dLRPHandler,
    BatchNorm2dLRPHandler,
    FlattenLRPHandler,
    DropoutLRPHandler,
    RNNLRPHandler,
)

logger = logging.getLogger(__name__)


class _LRPHookContext(contextlib.AbstractContextManager):
    """Owns all mutable hook state for one LRP forward+backward pass.

    Creating a fresh context per :meth:`LayerwiseRelevancePropagation.attribute`
    call eliminates the concurrency / re-entrancy hazard that arises when hook
    state is stored as instance attributes on the interpreter.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.activations: Dict[str, dict] = {}
        self.hooks: list = []
        self.residual_blocks: Dict[str, dict] = {}

    def __enter__(self) -> "_LRPHookContext":
        self._register_hooks()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()
        self.residual_blocks.clear()
        return False

    def _register_hooks(self) -> None:  # noqa: C901
        def save_activation(name: str):
            def hook(module, input, output):
                in_t = input[0] if isinstance(input, tuple) else input
                out_t = output[0] if isinstance(output, tuple) else output
                entry: dict = {"input": in_t, "output": out_t, "module": module}
                # Pre-compute pool indices so MaxPool2dLRPHandler can unpool
                if isinstance(module, nn.MaxPool2d):
                    _, indices = F.max_pool2d(
                        in_t,
                        kernel_size=module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                        dilation=module.dilation,
                        return_indices=True,
                    )
                    entry["indices"] = indices
                self.activations[name] = entry
            return hook

        for name, module in self.model.named_modules():
            if isinstance(
                module,
                (
                    nn.Linear, nn.ReLU, nn.LSTM, nn.GRU,
                    nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d, nn.BatchNorm2d,
                    nn.Flatten, nn.Dropout,
                ),
            ):
                handle = module.register_forward_hook(save_activation(name))
                self.hooks.append(handle)

        try:
            from torchvision.models.resnet import BasicBlock, Bottleneck

            def save_residual_block(block_name: str):
                def hook(module, input, output):
                    x = input[0] if isinstance(input, tuple) else input
                    self.residual_blocks[block_name] = {
                        "input": x,
                        "has_downsample": module.downsample is not None,
                    }
                return hook

            for name, module in self.model.named_modules():
                if isinstance(module, (BasicBlock, Bottleneck)):
                    handle = module.register_forward_hook(save_residual_block(name))
                    self.hooks.append(handle)
        except (ImportError, AttributeError):
            pass


class LayerwiseRelevancePropagation(BaseInterpreter):
    """LRP attribution method for PyHealth models.

    Decomposes a model's prediction into per-feature relevance scores
    via backward propagation. Satisfies the conservation property:
    sum of relevances ~ f(x).

    Args:
        model: Trained PyHealth model to interpret.
        rule: Propagation rule ('epsilon' or 'alphabeta').
        epsilon: Stabilizer for epsilon-rule (default: 0.01).
        alpha: Positive weight for alphabeta-rule (default: 1.0).
        beta: Negative weight for alphabeta-rule (default: 0.0).
        use_embeddings: If True, propagate from embedding layer.

    Examples:
        >>> lrp = LayerwiseRelevancePropagation(model, rule="epsilon")
        >>> attributions = lrp.attribute(**test_batch)
        >>> for key, rel in attributions.items():
        ...     print(f"{key}: sum={rel.sum().item():.4f}")
    """

    def __init__(
        self,
        model: BaseModel,
        rule: Literal["epsilon", "alphabeta"] = "epsilon",
        epsilon: float = 0.01,
        alpha: float = 1.0,
        beta: float = 0.0,
        use_embeddings: bool = True,
    ):
        super().__init__(model)
        if use_embeddings and not isinstance(model, Interpretable):
            raise ValueError(
                "Model must implement Interpretable interface when "
                "use_embeddings=True"
            )
        self.rule = rule
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.use_embeddings = use_embeddings

        self._registry = LRPHandlerRegistry()
        for handler in [
            LinearLRPHandler(),
            ReLULRPHandler(),
            Conv2dLRPHandler(),
            MaxPool2dLRPHandler(),
            AvgPool2dLRPHandler(),
            AdaptiveAvgPool2dLRPHandler(),
            BatchNorm2dLRPHandler(),
            FlattenLRPHandler(),
            DropoutLRPHandler(),
            RNNLRPHandler(),
        ]:
            self._registry.register(handler)

        if use_embeddings:
            assert hasattr(model, "forward_from_embedding"), (
                f"Model {type(model).__name__} must implement "
                "forward_from_embedding() for embedding-level LRP. "
                "Set use_embeddings=False for continuous features only."
            )

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute LRP attributions for input features.

        Args:
            target_class_idx: Target class for attribution (None = predicted).
            **data: Input data from a dataloader batch.

        Returns:
            Dict mapping each feature key to its relevance tensor.
        """
        feature_keys = getattr(self.model, "feature_keys", list(data.keys()))
        device = next(self.model.parameters()).device
        has_processors = hasattr(self.model, "dataset") and hasattr(
            getattr(self.model, "dataset", None), "input_processors"
        )

        # Keep original input tuples (matching processor schema) and extract values
        inputs, values, label_data = {}, {}, {}
        for key in feature_keys:
            if key not in data:
                continue
            v = data[key]
            if isinstance(v, torch.Tensor):
                v = (v,)
            inputs[key] = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in v
            )
            if has_processors and key in self.model.dataset.input_processors:
                schema = self.model.dataset.input_processors[key].schema()
                values[key] = inputs[key][schema.index("value")]
            else:
                # Fallback for models without processor schema
                values[key] = inputs[key][0]

        # Build and re-append padding masks so sequence models receive them (Issue 3)
        if has_processors:
            masks: Dict[str, torch.Tensor] = {}
            for k in list(inputs.keys()):
                if k not in self.model.dataset.input_processors:
                    continue
                schema = self.model.dataset.input_processors[k].schema()
                if "mask" in schema:
                    masks[k] = inputs[k][schema.index("mask")]
                else:
                    processor = self.model.dataset.input_processors[k]
                    val = values[k]
                    if processor.is_token():
                        # For nested tokens (val is 3D [B, seq, inner]), produce a
                        # visit-level mask [B, seq] so temporal models receive 2D masks
                        raw = (val != 0)
                        if raw.dim() == 3:
                            masks[k] = raw.any(dim=-1).int()
                        else:
                            masks[k] = raw.int()
                    elif val.dim() >= 3:
                        masks[k] = (val.abs().sum(dim=-1) != 0).int()
                    else:
                        masks[k] = (val != 0).int()
            for k in list(inputs.keys()):
                if k in masks and k in self.model.dataset.input_processors:
                    schema = self.model.dataset.input_processors[k].schema()
                    if "mask" not in schema:
                        inputs[k] = (*inputs[k], masks[k])

        for key in getattr(self.model, "label_keys", []):
            if key in data:
                val = data[key]
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val)
                label_data[key] = val.to(device)

        if self.use_embeddings:
            return self._compute_from_embeddings(
                inputs, values, target_class_idx, label_data
            )
        return self._compute_from_inputs(inputs, values, target_class_idx, label_data)

    def visualize(self, plt, image, relevance, title=None, method="overlay", **kwargs):
        """Visualize LRP relevance maps using SaliencyVisualizer."""
        from pyhealth.interpret.methods.saliency_visualization import visualize_attribution

        if title is None:
            title = f"LRP Attribution ({self.rule}-rule)"
        visualize_attribution(plt, image, relevance, title=title, method=method, **kwargs)

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def _compute_from_embeddings(self, inputs, values, target_class_idx, label_data):
        """LRP starting from embedding layer (for discrete inputs)."""
        embedding_model = self.model.get_embedding_model()
        assert embedding_model is not None, (
            "Model must have an embedding model for embedding-based LRP. "
            "Set use_embeddings=False for continuous features only."
        )
        input_embeddings, input_shapes = {}, {}

        for key in values:
            input_shapes[key] = values[key].shape
            embedded = embedding_model({key: values[key]})
            x = embedded[key]
            if x.dim() == 4:
                x = x.sum(dim=2)
            input_embeddings[key] = x

        # Reconstruct input tuples with embedded values spliced in
        forward_kwargs = {}
        has_processors = hasattr(self.model, "dataset") and hasattr(
            getattr(self.model, "dataset", None), "input_processors"
        )
        for key in inputs:
            if has_processors and key in self.model.dataset.input_processors:
                schema = self.model.dataset.input_processors[key].schema()
                val_idx = schema.index("value")
                original = inputs[key]
                forward_kwargs[key] = (
                    *original[:val_idx],
                    input_embeddings[key],
                    *original[val_idx + 1:],
                )
            else:
                # Fallback: pass bare embedding tensor
                forward_kwargs[key] = input_embeddings[key]
        if label_data:
            forward_kwargs.update(label_data)

        with _LRPHookContext(self.model) as ctx:
            with torch.no_grad():
                output = self.model.forward_from_embedding(**forward_kwargs)
            logits = output["logit"]
            output_relevance = self._init_output_relevance(logits, target_class_idx)
            relevance_at_emb = self._propagate_relevance_backward(
                output_relevance, input_embeddings, ctx
            )

        result = {}
        for key in input_embeddings:
            rel = relevance_at_emb.get(key)
            if rel is None:
                continue
            # Sum over embedding dim to get per-token relevance
            if rel.dim() >= 2:
                result[key] = rel.sum(dim=-1)
            else:
                result[key] = rel
            # Expand to match original input shape if needed
            orig = input_shapes[key]
            if result[key].shape != orig and len(orig) == 3 and result[key].dim() == 2:
                result[key] = result[key].unsqueeze(-1).expand(orig)
        return result

    def _compute_from_inputs(self, inputs, values, target_class_idx, label_data):
        """LRP starting from continuous inputs (e.g., images)."""
        self.model.eval()
        forward_kwargs = {**values}
        if label_data:
            forward_kwargs.update(label_data)
        with _LRPHookContext(self.model) as ctx:
            with torch.no_grad():
                output = self.model(**forward_kwargs)
            logits = self._extract_logits(output, ctx)
            output_relevance = self._init_output_relevance(logits, target_class_idx)
            result = self._propagate_relevance_backward(output_relevance, values, ctx)
        if not isinstance(result, dict):
            result = {list(values.keys())[0]: result}
        return result

    def _extract_logits(
        self, output: dict, ctx: "_LRPHookContext"
    ) -> torch.Tensor:
        """Return raw pre-softmax logits for LRP initialisation.

        Priority:
        1. ``output["logit"]`` — model explicitly exposes raw logits.
        2. Last ``nn.Linear`` hook in *ctx* — the final classification head
           fires its hook *before* ``prepare_y_prob`` applies softmax, so
           its cached output is the true pre-softmax logit even for models
           like ``TorchvisionModel`` that do not return a "logit" key.
        3. ``output["y_prob"]`` — last-resort fallback; conservation will
           not hold because softmax collapses the logit scale.
        """
        if "logit" in output:
            return output["logit"]

        # Walk the activations dict (insertion = forward-execution order).
        # Keeping the last Linear hit gives the final classification head.
        last_linear_output: Optional[torch.Tensor] = None
        for info in ctx.activations.values():
            if isinstance(info["module"], nn.Linear):
                last_linear_output = info["output"]

        if last_linear_output is not None:
            logger.debug(
                "Extracted raw logits from the final Linear layer hook "
                "(model does not return a 'logit' key)."
            )
            return last_linear_output

        if "y_prob" in output:
            logger.debug(
                "Model forward() did not return a 'logit' key; falling back "
                "to 'y_prob'. LRP conservation property may not hold."
            )
            return output["y_prob"]

        raise KeyError(
            "Model forward() must return a 'logit' key for LRP. "
            f"Got keys: {list(output.keys())}"
        )

    def _init_output_relevance(self, logits, target_class_idx):
        """Initialize relevance at the output layer using prediction-mode dispatch."""
        try:
            mode = self._prediction_mode()
        except Exception:
            # Fall back to shape-based heuristic for models that don't expose
            # output schema (e.g. TorchvisionModel with use_embeddings=False).
            mode = "multiclass" if logits.dim() == 2 and logits.size(-1) > 1 else "binary"

        if mode == "binary":
            # Binary logits have shape [B] or [B, 1].
            # target_class_idx=1 (positive) → use logit as-is.
            # target_class_idx=0 (negative) → negate so LRP attributes for the
            # "not-positive" direction (relevance conservation still holds for -logit).
            if target_class_idx is not None:
                idx = (
                    target_class_idx
                    if isinstance(target_class_idx, int)
                    else int(target_class_idx.item())
                )
                if idx == 0:
                    return -logits
            return logits
        if mode in ("multiclass", "multilabel"):
            if target_class_idx is None:
                target_class_idx = torch.argmax(logits, dim=-1)
            elif not isinstance(target_class_idx, torch.Tensor):
                target_class_idx = torch.tensor(
                    target_class_idx, device=logits.device
                )
            batch_size = logits.size(0)
            output_relevance = torch.zeros_like(logits)
            output_relevance[range(batch_size), target_class_idx] = logits[
                range(batch_size), target_class_idx
            ]
            return output_relevance
        # Regression or unknown mode: return logits unchanged.
        return logits

    # ------------------------------------------------------------------
    # Backward propagation
    # ------------------------------------------------------------------

    def _get_block_for_layer(self, layer_name: str, ctx: _LRPHookContext) -> Optional[str]:
        """Return the residual block prefix if this layer belongs to a tracked block."""
        for bname in ctx.residual_blocks:
            if layer_name.startswith(bname + "."):
                return bname
        return None

    def _propagate_through_residual_block(
        self,
        block_name: str,
        block_layer_names_rev: list,
        r_in: torch.Tensor,
        ctx: _LRPHookContext,
    ) -> torch.Tensor:
        """Propagate relevance through a residual block with correct LRP skip-connection split.

        At each residual addition ``out = main_path + identity`` the epsilon rule requires
        splitting the incoming relevance proportionally:
            R_main     = (main_path / stabilise(main_path + identity)) * R_in
            R_identity = (identity  / stabilise(main_path + identity)) * R_in

        R_main is then propagated back through the block's convolutional layers while
        R_identity bypasses them (or is propagated through the downsample layers when
        the block includes a projection shortcut).
        """
        has_downsample = ctx.residual_blocks[block_name]["has_downsample"]
        block_input = ctx.residual_blocks[block_name]["input"]

        # Partition layers into main path vs. downsample path
        main_layers = [l for l in block_layer_names_rev if ".downsample." not in l]
        ds_layers = [l for l in block_layer_names_rev if ".downsample." in l]

        # main_out = output of the last main-path BN (first encountered in reversed order,
        # e.g. bn2 for BasicBlock, bn3 for Bottleneck)
        main_out = ctx.activations[main_layers[0]]["output"] if main_layers else None

        # identity = output of the final downsample BN (projection shortcut) OR raw input
        if has_downsample and ds_layers:
            identity = ctx.activations[ds_layers[0]]["output"]
        else:
            identity = block_input

        # LRP-epsilon split at the residual addition
        if (main_out is not None
                and identity is not None
                and main_out.shape == identity.shape
                and r_in.shape == main_out.shape):
            total = main_out + identity
            sign = total.sign()
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            denom = total + self.epsilon * sign
            r_for_main = (main_out / denom) * r_in
            r_for_identity = (identity / denom) * r_in
        else:
            r_for_main = r_in
            r_for_identity = None

        # Propagate r_for_main backward through the main-path layers
        r_at_input = r_for_main
        for ln in main_layers:
            ai = ctx.activations[ln]
            if r_at_input.shape != ai["output"].shape:
                r_at_input = self._match_shapes(r_at_input, ai["output"].shape)
            r_at_input = self._propagate_through_layer(ai["module"], ai, r_at_input)

        # Propagate r_for_identity through downsample layers (projection shortcut)
        if has_downsample and ds_layers and r_for_identity is not None:
            r_ds = r_for_identity
            for ln in ds_layers:
                ai = ctx.activations[ln]
                if r_ds.shape != ai["output"].shape:
                    r_ds = self._match_shapes(r_ds, ai["output"].shape)
                r_ds = self._propagate_through_layer(ai["module"], ai, r_ds)
            if r_ds.shape == r_at_input.shape:
                r_at_input = r_at_input + r_ds
        elif not has_downsample and r_for_identity is not None:
            # No projection: identity goes directly to block input
            if r_for_identity.shape == r_at_input.shape:
                r_at_input = r_at_input + r_for_identity

        return r_at_input

    def _propagate_relevance_backward(self, output_relevance, input_embeddings, ctx: _LRPHookContext):
        """Propagate relevance from output back to input embeddings.

        Iterates layers in reverse-forward order.  When a layer belongs to a
        tracked residual block (BasicBlock / Bottleneck), all sub-layers of that
        block are handled by ``_propagate_through_residual_block`` which applies
        a proper LRP-epsilon split at the skip-connection addition.
        """
        current_relevance = output_relevance
        layer_names = list(reversed(list(ctx.activations.keys())))

        feature_relevances = {}
        concat_detected = False
        blocks_handled: set = set()

        # Pre-compute expected combined embedding size for shape-based concat detection
        # (Issue 8). This is reliable when all features have consistent embedding dims.
        expected_concat_size: Optional[int] = None
        n_features = 0
        if isinstance(input_embeddings, dict) and hasattr(self.model, "feature_keys"):
            n_features = len(
                [k for k in self.model.feature_keys if k in input_embeddings]
            )
            if n_features > 1:
                try:
                    expected_concat_size = sum(
                        emb.size(-1) for emb in input_embeddings.values()
                    )
                except Exception:
                    expected_concat_size = None

        idx = 0
        while idx < len(layer_names):
            layer_name = layer_names[idx]

            # ---- Residual block handling ----
            block_name = self._get_block_for_layer(layer_name, ctx)
            if block_name is not None and block_name not in blocks_handled:
                # Collect all consecutive layers that belong to this block
                block_layers = []
                j = idx
                while j < len(layer_names) and self._get_block_for_layer(layer_names[j], ctx) == block_name:
                    block_layers.append(layer_names[j])
                    j += 1

                current_relevance = self._propagate_through_residual_block(
                    block_name, block_layers, current_relevance, ctx
                )
                blocks_handled.add(block_name)
                idx = j
                continue

            # ---- Regular layer handling ----
            activation_info = ctx.activations[layer_name]
            module = activation_info["module"]
            output_tensor = activation_info["output"]

            # Detect concatenation point (multi-feature MLP pattern)
            if not concat_detected and isinstance(module, nn.Linear) and n_features > 1:
                # Primary signal: shape matches the sum of all embedding dims
                if (
                    expected_concat_size is not None
                    and current_relevance.dim() == 2
                    and current_relevance.size(1) == expected_concat_size
                ):
                    concat_detected = True
                # Fallback: string heuristic (warns to aid debugging)
                elif (
                    idx + 1 < len(layer_names)
                    and hasattr(self.model, "feature_keys")
                ):
                    next_name = layer_names[idx + 1]
                    if "mlp." in next_name and any(
                        f in next_name for f in self.model.feature_keys
                    ):
                        logger.warning(
                            "LRP concat detection fell back to string heuristic for "
                            "layer '%s'; consider ensuring all embedding dims are "
                            "consistent so shape-based detection triggers instead.",
                            layer_name,
                        )
                        concat_detected = True

            if current_relevance.shape != output_tensor.shape:
                current_relevance = self._match_shapes(
                    current_relevance, output_tensor.shape
                )

            current_relevance = self._propagate_through_layer(
                module, activation_info, current_relevance
            )

            if concat_detected and current_relevance.dim() == 2:
                n_feat = len(self.model.feature_keys)
                dim = current_relevance.size(1) // n_feat
                for i, fk in enumerate(self.model.feature_keys):
                    feature_relevances[fk] = current_relevance[
                        :, i * dim : (i + 1) * dim
                    ]
                # Process remaining feature-specific layers
                for fk in self.model.feature_keys:
                    cur = feature_relevances[fk]
                    for ln in layer_names[idx + 1:]:
                        if fk not in ln:
                            continue
                        ai = ctx.activations[ln]
                        m = ai["module"]
                        if cur.shape != ai["output"].shape:
                            cur = self._match_shapes(cur, ai["output"].shape)
                        cur = self._propagate_through_layer(m, ai, cur)
                    feature_relevances[fk] = cur
                return self._split_relevance_to_features(
                    feature_relevances, input_embeddings
                )

            idx += 1

        return self._split_relevance_to_features(
            current_relevance, input_embeddings
        )

    def _invoke_handler(
        self,
        handler,
        module: nn.Module,
        activation_info: dict,
        relevance: torch.Tensor,
    ) -> torch.Tensor:
        """Populate *handler*'s activation cache from *activation_info* and call it.

        The handlers in ``lrp_base`` use ``self._get_cached(layer)`` internally.
        We bridge that by pre-populating the cache entry before the call and
        removing it immediately after, so the handler never observes stale data.
        """
        if isinstance(module, nn.Flatten):
            cache_entry = {
                "input_shape": activation_info["input"].shape,
                "output": activation_info["output"],
            }
        else:
            cache_entry = {k: v for k, v in activation_info.items() if k != "module"}
        handler.activations_cache[id(module)] = cache_entry
        try:
            return handler.backward_relevance(
                module,
                relevance,
                rule=self.rule,
                epsilon=self.epsilon,
                alpha=self.alpha,
                beta=self.beta,
            )
        finally:
            handler.activations_cache.pop(id(module), None)

    def _propagate_through_layer(self, module, activation_info, relevance):
        """Route relevance propagation to the correct handler via the registry."""
        handler = self._registry.get_handler(module)
        if handler is None:
            return relevance
        return self._invoke_handler(handler, module, activation_info, relevance)

    # ------------------------------------------------------------------
    # Shape utilities
    # ------------------------------------------------------------------

    def _match_shapes(self, relevance, target_shape):
        if relevance.shape == target_shape:
            return relevance
        batch_size = relevance.shape[0]

        # 2D -> 4D
        if relevance.dim() == 2 and len(target_shape) == 4:
            total = target_shape[1] * target_shape[2] * target_shape[3]
            if relevance.shape[1] == total:
                return relevance.view(batch_size, *target_shape[1:])
            return (relevance.sum(dim=1, keepdim=True) / total).view(
                batch_size, 1, 1, 1
            ).expand(batch_size, *target_shape[1:])

        # 4D -> 4D
        if relevance.dim() == 4 and len(target_shape) == 4:
            if relevance.shape[1] != target_shape[1]:
                relevance = relevance.mean(dim=1, keepdim=True).expand(
                    -1, target_shape[1], -1, -1
                )
            if relevance.shape[2:] != target_shape[2:]:
                relevance = F.interpolate(
                    relevance, size=target_shape[2:], mode="bilinear",
                    align_corners=False,
                )
            return relevance

        # 3D -> 4D
        if relevance.dim() == 3 and len(target_shape) == 4:
            relevance = relevance.unsqueeze(1).expand(
                -1, target_shape[1], -1, -1
            )
            if relevance.shape[2:] != target_shape[2:]:
                relevance = F.interpolate(
                    relevance, size=target_shape[2:], mode="bilinear",
                    align_corners=False,
                )
            return relevance

        return relevance

    def _split_relevance_to_features(self, relevance, input_embeddings):
        """Split combined relevance back to individual features."""
        result = {}

        # Already split per feature
        if isinstance(relevance, dict):
            for key, rel in relevance.items():
                if key not in input_embeddings:
                    continue
                emb = input_embeddings[key]
                if emb.dim() == 3 and rel.dim() == 2:
                    feat_dim = emb.size(-1)
                    if rel.size(1) > feat_dim:
                        # Strip appended dimensions (e.g. StageNet concatenates
                        # a 1-D time interval to the embedding before its
                        # kernel Linear; LRP returns relevance for the full
                        # concatenated input, so discard the time columns).
                        rel = rel[:, :feat_dim]
                    rel = rel.unsqueeze(1).expand_as(emb)
                result[key] = rel
            return result

        # Compute per-feature sizes (after pooling)
        feature_sizes = {}
        for key, emb in input_embeddings.items():
            if emb.dim() == 3:
                feature_sizes[key] = emb.size(2)
            elif emb.dim() == 2:
                feature_sizes[key] = emb.size(1)
            else:
                feature_sizes[key] = emb.numel() // emb.size(0)

        total = sum(feature_sizes.values())
        if relevance.dim() != 2 or relevance.size(1) != total:
            # Fallback: distribute equally
            for key in input_embeddings:
                result[key] = relevance / len(input_embeddings)
            return result

        idx = 0
        for key in self.model.feature_keys:
            if key not in input_embeddings:
                continue
            size = feature_sizes[key]
            chunk = relevance[:, idx : idx + size]
            emb = input_embeddings[key]
            if emb.dim() == 3:
                chunk = chunk.unsqueeze(1).expand_as(emb)
            result[key] = chunk
            idx += size

        return result