from __future__ import annotations

import math
import contextlib
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from pyhealth.models import BaseModel

from .base_interpreter import BaseInterpreter


def _iter_child_modules(module: torch.nn.Module):
    for name, child in module.named_children():
        yield module, name, child
        yield from _iter_child_modules(child)


class _SoftmaxTSG(torch.nn.Module):
    """Swap nn.Softmax with temperature-adjusted backward for GIM."""

    def __init__(self, dim: int | None, temperature: float):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  
        return _TemperatureSoftmaxFn.apply(tensor, self.dim, self.temperature) # type: ignore[override]


class _FrozenLayerNorm(torch.nn.Module):
    """LayerNorm replacement that treats normalization statistics as constants.

    Implements the LayerNorm freeze rule from GIM Sec. 4.2: in the forward
    pass the output is identical to ``nn.LayerNorm``, but the backward pass
    treats the mean and variance as fixed constants (i.e. their Jacobian
    contributions are detached).
    """

    def __init__(self, original: torch.nn.LayerNorm):
        super().__init__()
        self.original = original

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        out = _FrozenLayerNormFn.apply(
            x,
            self.original.normalized_shape,
            self.original.weight,
            self.original.bias,
            self.original.eps,
        )
        assert isinstance(out, torch.Tensor)
        return out


class _FrozenLayerNormFn(torch.autograd.Function):
    """Custom autograd: forward == LayerNorm, backward freezes statistics."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        normalized_shape: Tuple[int, ...],
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        eps: float,
    ) -> torch.Tensor:
        # Compute the standard LayerNorm output.
        dims = tuple(range(-len(normalized_shape), 0))
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, unbiased=False, keepdim=True)
        x_hat = (x - mean) / torch.sqrt(var + eps)
        out = x_hat
        if weight is not None:
            out = out * weight
        if bias is not None:
            out = out + bias
        # Save what we need for backward – mean and std are treated as
        # constants, so we only need x_hat, weight, and 1/std.
        inv_std = 1.0 / torch.sqrt(var + eps)
        ctx.save_for_backward(x_hat, weight, inv_std)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        x_hat, weight, inv_std = ctx.saved_tensors
        # Treat mean/var as frozen constants → ∂out/∂x = weight / std
        # (no correction terms from differentiating through mean/var).
        if weight is not None:
            grad_x = grad_output * weight * inv_std
        else:
            grad_x = grad_output * inv_std

        # Gradients for affine parameters (weight, bias) are standard.
        grad_weight = None
        if weight is not None:
            grad_weight = (grad_output * x_hat).flatten(end_dim=-len(weight.shape) - 1).sum(0)
        grad_bias = None
        if ctx.saved_tensors[1] is not None:  # bias was provided
            grad_bias = grad_output.flatten(end_dim=-len(weight.shape) - 1).sum(0)

        return grad_x, None, grad_weight, grad_bias, None


class _MatMulNorm(torch.autograd.Function):
    """matmul whose backward divides grad by fan-in (=2 for a binary product).

    Implements the uniform division rule from GIM Sec. 4.2 for a single
    matrix multiplication.  Used inside :class:`_AttentionGIM` to normalise
    gradients flowing through Q·K^T and attn·V products.

    Because the /2 is applied *per matmul*, the effective normalisation
    compounds across the two sequential multiplications in attention:

    * **V** participates in one matmul (attn·V)  → effective /2.
    * **Q** passes through two matmuls (Q·K^T → softmax → attn·V) →
      effective /4  (the /2 from attn·V propagates through softmax's
      linear Jacobian, then a second /2 comes from Q·K^T).
    * **K** — same as Q → effective /4.

    This matches the reference implementation (JoakimEdin/gim,
    ``_grad_normalize``: key÷4, query÷4, value÷2).
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(a, b)
        return a @ b

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        a, b = ctx.saved_tensors
        grad_a = (grad_output @ b.transpose(-2, -1)) / 2.0
        grad_b = (a.transpose(-2, -1) @ grad_output) / 2.0
        return grad_a, grad_b


class _AttentionGIM(torch.nn.Module):
    """Drop-in replacement for ``Attention`` that applies GIM rules 1 & 3.

    1. **TSG** – the softmax in the forward pass is computed normally, but
       the backward Jacobian uses a higher temperature (Sec. 4.1).
    3. **Gradient normalisation** – both ``matmul(Q, K^T)`` and
       ``matmul(attn, V)`` use the uniform division rule so that each
       factor receives half of the incoming gradient (Sec. 4.2).

    This module mirrors the signature of
    ``pyhealth.models.transformer.Attention`` so it can be swapped in and
    out by :class:`_GIMSwapContext` without touching any global state.
    """

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature = max(float(temperature), 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dropout: Optional[torch.nn.Module] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- scores = Q · K^T / sqrt(d_k) with gradient normalisation ---
        qk = _MatMulNorm.apply(query, key.transpose(-2, -1))
        assert isinstance(qk, torch.Tensor)
        scores = qk / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # --- softmax with TSG ---
        p_attn: torch.Tensor = _TemperatureSoftmaxFn.apply(scores, -1, self.temperature)  # type: ignore[assignment]

        if mask is not None:
            p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)

        # --- attn · V with gradient normalisation ---
        out = _MatMulNorm.apply(p_attn, value)
        assert isinstance(out, torch.Tensor)
        return out, p_attn


class _GIMSwapContext(contextlib.AbstractContextManager):
    """Temporarily replace Attention, Softmax and LayerNorm modules with GIM-aware versions."""

    def __init__(self, model: BaseModel, temperature: float):
        self.model = model
        self.temperature = temperature
        self._swapped: List[Tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def __enter__(self) -> "_GIMSwapContext":
        for parent, name, child in _iter_child_modules(self.model):
            # Swap Attention modules inside MultiHeadedAttention –
            # this subsumes both the softmax (TSG) and the matmul
            # (gradient normalisation) rules for attention.
            if self._is_attention_module(child):
                wrapper = _AttentionGIM(temperature=self.temperature)
                setattr(parent, name, wrapper)
                self._swapped.append((parent, name, child))
            # Swap remaining standalone nn.Softmax modules (e.g. StageNet's
            # cumulative softmax) that live outside of Attention.
            elif isinstance(child, torch.nn.Softmax):
                wrapper = _SoftmaxTSG(dim=child.dim, temperature=self.temperature)
                setattr(parent, name, wrapper)
                self._swapped.append((parent, name, child))
            # Swap nn.LayerNorm modules (LN freeze rule).
            elif isinstance(child, torch.nn.LayerNorm):
                wrapper = _FrozenLayerNorm(child)
                setattr(parent, name, wrapper)
                self._swapped.append((parent, name, child))
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        for parent, name, original in reversed(self._swapped):
            setattr(parent, name, original)
        self._swapped.clear()
        return False

    @staticmethod
    def _is_attention_module(module: torch.nn.Module) -> bool:
        """Return True for PyHealth's ``Attention`` (scaled dot-product helper)."""
        cls = type(module)
        return (
            cls.__name__ == "Attention"
            and hasattr(module, "softmax")
            and isinstance(getattr(module, "softmax"), torch.nn.Softmax)
        )


class _TemperatureSoftmaxFn(torch.autograd.Function):
    """Custom autograd op implementing temperature-adjusted softmax gradients.

    Implements the Temperature-Scaled Gradients (TSG) rule from GIM Sec. 4.1 by
    recomputing the backward Jacobian with a higher temperature while leaving
    the forward softmax unchanged.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        dim: int,
        temperature: float,
    ) -> torch.Tensor:
        ctx.dim = dim
        ctx.temperature = float(temperature)
        ctx.save_for_backward(input_tensor)
        return torch.softmax(input_tensor, dim=dim)

    @staticmethod
    def backward( # type: ignore[return]
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None, None]: 
        (input_tensor,) = ctx.saved_tensors
        dim = ctx.dim
        temperature = max(ctx.temperature, 1.0)

        if temperature == 1.0:
            probs = torch.softmax(input_tensor, dim=dim)
            dot = (grad_output * probs).sum(dim=dim, keepdim=True)
            grad_input = probs * (grad_output - dot)
            return grad_input, None, None

        # TSG: recompute softmax at higher temperature, then use the
        # *standard* softmax Jacobian formula evaluated at the
        # temperature-adjusted distribution.  Crucially, we do NOT
        # multiply by 1/T (the chain-rule factor for x/T) — TSG is
        # defined as "change the point at which the Jacobian is
        # evaluated", not "compute the full derivative of softmax(x/T)".
        # This matches the reference implementation (softmax_tsg in
        # JoakimEdin/gim, utils.py).
        adjusted = torch.softmax(input_tensor / temperature, dim=dim)
        dot = (grad_output * adjusted).sum(dim=dim, keepdim=True)
        grad_input = adjusted * (grad_output - dot)
        return grad_input, None, None


class _GIMHookContext(contextlib.AbstractContextManager):
    """Context manager that installs all GIM backward-pass modifications.

    Activates three mechanisms when entered (all via module swapping):
    1. Temperature-adjusted softmax (TSG) — ``nn.Softmax`` and ``Attention``.
    2. Frozen LayerNorm — ``nn.LayerNorm``.
    3. Gradient normalisation for Q·K^T and attn·V — ``Attention``.
    """

    def __init__(self, model: BaseModel, temperature: float):
        self.model = model
        self.temperature = temperature
        self._swap_ctx = _GIMSwapContext(model, temperature=max(float(temperature), 1.0))

    def __enter__(self) -> "_GIMHookContext":
        self._swap_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self._swap_ctx.__exit__(exc_type, exc, exc_tb)
        return False


class GIM(BaseInterpreter):
    """Gradient Interaction Modifications for StageNet-style and Transformer models.

    This interpreter adapts the Gradient Interaction Modifications (GIM)
    technique (Edin et al., 2025) to PyHealth. It supports both
    recurrent models such as StageNet (where cumulative softmax can
    exhibit self-repair) and Transformer / attention-based architectures
    (where LayerNorm and Q·K^T interactions require special treatment).

    The implementation follows three rules from the paper:

    1. **Temperature-adjusted softmax gradients (TSG):** All ``nn.Softmax``
       modules are temporarily replaced so the backward Jacobian is
       recomputed at a higher temperature, exposing interactions hidden
       by softmax redistribution (Sec. 4.1).
    2. **LayerNorm freeze:** ``nn.LayerNorm`` modules are replaced with a
       variant that treats the running mean and variance as frozen
       constants during backpropagation. For models without LayerNorm
       (e.g. StageNet) this is a no-op (Sec. 4.2).
    3. **Gradient normalization (uniform division):** ``torch.matmul``
       calls inside attention layers (the Q·K^T product) are wrapped so
       that gradients flowing through the binary product are divided by 2.
       Thanks to composition across the two matmuls in attention, Q and K
       effectively receive /4 and V receives /2, matching the reference
       implementation.  For models without multi-head attention (e.g.
       StageNet) this is a no-op (Sec. 4.2).

    .. note::
       The paper also mentions a third multiplicative interaction (MLP
       gate-projection) that is relevant for gated FFNs (e.g. SwiGLU).
       PyHealth's ``PositionwiseFeedForward`` uses a standard two-layer
       FFN with GELU (no element-wise gate), so this normalisation is not
       needed and is intentionally omitted.

    Args:
        model: Trained PyHealth model supporting ``forward_from_embedding``
            and ``get_embedding_model()``. Currently tested with StageNet,
            StageNetMHA, and Transformer.
        temperature: Softmax temperature used exclusively for the backward
            pass. A value of ``2.0`` matches the paper's best setting.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import get_dataloader
        >>> from pyhealth.interpret.methods.gim import GIM
        >>> from pyhealth.models import StageNet
        >>>
        >>> # Assume ``sample_dataset`` and trained StageNet weights are available.
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> model = StageNet(dataset=sample_dataset)
        >>> model = model.to(device).eval()
        >>> test_loader = get_dataloader(sample_dataset, batch_size=1, shuffle=False)
        >>> gim = GIM(model, temperature=2.0)
        >>>
        >>> batch = next(iter(test_loader))
        >>> attributions = gim.attribute(**batch)
        >>> print({k: v.shape for k, v in attributions.items()})
    """

    def __init__(
        self,
        model: BaseModel,
        temperature: float = 2.0,
    ):
        super().__init__(model)
        if not hasattr(model, "forward_from_embedding"):
            raise AssertionError(
                "GIM requires models that implement `forward_from_embedding`."
            )
        embedding_model = model.get_embedding_model()
        if embedding_model is None:
            raise AssertionError(
                "GIM requires a model with an embedding model "
                "accessible via `get_embedding_model()`."
            )
        self.temperature = max(float(temperature), 1.0)

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **kwargs: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> Dict[str, torch.Tensor]:
        """Compute GIM attributions for a batch.

        Args:
            target_class_idx: Target class index for attribution. If None,
                uses the model's predicted class.
            **kwargs: Input data dictionary from a dataloader batch containing
                feature tensors or tuples of tensors for each modality, plus
                optional label tensors.

        Returns:
            Dictionary mapping feature keys to attribution tensors with the
            same shape as the raw input values.
        """
        device = next(self.model.parameters()).device

        # Filter kwargs to only include model feature keys and ensure tuples
        inputs = {
            k: (v,) if isinstance(v, torch.Tensor) else v
            for k, v in kwargs.items()
            if k in self.model.feature_keys
        }

        # Disassemble inputs to get values and masks via processor schema
        values: dict[str, torch.Tensor] = {}
        masks: dict[str, torch.Tensor] = {}
        for k, v in inputs.items():
            schema = self.model.dataset.input_processors[k].schema()
            values[k] = v[schema.index("value")]
            if "mask" in schema:
                masks[k] = v[schema.index("mask")]
            else:
                masks[k] = (v[schema.index("value")] != 0).int()

        # Append input masks to inputs for models that expect them
        for k, v in inputs.items():
            if "mask" not in self.model.dataset.input_processors[k].schema():
                inputs[k] = (*v, masks[k])

        # Save raw shapes before embedding for later mapping
        shapes = {k: v.shape for k, v in values.items()}

        # Determine target class from original input
        with torch.no_grad():
            base_logits = self.model.forward(**inputs)["logit"]

        mode = self._prediction_mode()
        if mode == "binary":
            if target_class_idx is not None:
                target = torch.tensor([target_class_idx], device=device)
            else:
                target = (torch.sigmoid(base_logits) > 0.5).long()
        elif mode == "multiclass":
            if target_class_idx is not None:
                target = F.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=base_logits.shape[-1],
                ).float()
            else:
                target = torch.argmax(base_logits, dim=-1)
                target = F.one_hot(
                    target, num_classes=base_logits.shape[-1]
                ).float()
        elif mode == "multilabel":
            if target_class_idx is not None:
                target = F.one_hot(
                    torch.tensor(target_class_idx, device=device),
                    num_classes=base_logits.shape[-1],
                ).float()
            else:
                target = (torch.sigmoid(base_logits) > 0.5).float()
        else:
            raise ValueError(
                "Unsupported prediction mode for GIM attribution."
            )

        # Embed values and detach for gradient attribution.
        # Split features by type using is_token():
        # - Token features (discrete): embed before gradient computation,
        #   since raw indices are not differentiable. Gradients are computed
        #   w.r.t. embeddings, then summed over the embedding dim.
        # - Continuous features: keep raw so each raw dimension gets its own
        #   gradient-based attribution. The embedding happens inside the
        #   forward pass via the embedding model.
        embedding_model = self.model.get_embedding_model()
        assert embedding_model is not None

        token_keys = {
            k for k in values
            if self.model.dataset.input_processors[k].is_token()
        }
        continuous_keys = set(values.keys()) - token_keys

        # Embed token features
        if token_keys:
            token_embedded = embedding_model({k: values[k] for k in token_keys})
        else:
            token_embedded = {}

        # Prepare gradient targets: embeddings for tokens, raw values for continuous
        embeddings: dict[str, torch.Tensor] = {}
        for key in sorted(values.keys()):
            if key in token_keys:
                emb = token_embedded[key]
            else:
                emb = values[key].to(device).float()
            emb = emb.detach().requires_grad_(True)
            emb.retain_grad()
            embeddings[key] = emb

        # Insert gradient targets back into input tuples.
        # For continuous features, we also need to embed them for
        # forward_from_embedding, but we keep the raw tensor as the
        # gradient target so attributions have per-raw-feature granularity.
        forward_inputs = inputs.copy()
        for k in forward_inputs.keys():
            schema = self.model.dataset.input_processors[k].schema()
            val_idx = schema.index("value")
            if k in continuous_keys:
                # Embed the raw tensor through the embedding model;
                # autograd will track gradients back to the raw tensor.
                embedded_val = embedding_model({k: embeddings[k]})[k]
                forward_inputs[k] = (
                    *forward_inputs[k][:val_idx],
                    embedded_val,
                    *forward_inputs[k][val_idx + 1:],
                )
            else:
                forward_inputs[k] = (
                    *forward_inputs[k][:val_idx],
                    embeddings[k],
                    *forward_inputs[k][val_idx + 1:],
                )

        # Clear stale gradients before the attribution pass.
        self.model.zero_grad(set_to_none=True)

        # All three GIM rules are applied via _GIMHookContext:
        #   Step 1 (TSG): nn.Softmax → temperature-adjusted backward.
        #   Step 2 (LayerNorm freeze): nn.LayerNorm → frozen statistics.
        #   Step 3 (Gradient normalization): torch.matmul → uniform division
        #            for Q·K^T in attention layers.
        # The context manager detects which rules are applicable to the model
        # and only activates the relevant ones.
        with _GIMHookContext(self.model, self.temperature):
            output = self.model.forward_from_embedding(**forward_inputs)

            logits = output["logit"]  # type: ignore[assignment]
            target_output = self._compute_target_output(logits, target)

            # Clear stale gradients, then backpropagate through the
            # GIM-modified computational graph.
            self.model.zero_grad(set_to_none=True)
            for emb in embeddings.values():
                if emb.grad is not None:
                    emb.grad.zero_()

            target_output.backward()

        attributions: dict[str, torch.Tensor] = {}
        for key, emb in embeddings.items():
            grad = emb.grad
            if grad is None:
                grad = torch.zeros_like(emb)
            # Sum embedding dimension to get per-token attribution
            # (only for token features that were embedded before gradient computation)
            attr = grad.detach()
            if key in token_keys and attr.dim() >= 3:
                attr = attr.sum(dim=-1)
            attributions[key] = attr

        return self._map_to_input_shapes(attributions, shapes)

    # ------------------------------------------------------------------
    # Target output computation
    # ------------------------------------------------------------------
    def _compute_target_output(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scalar target output for backpropagation.

        Creates a differentiable scalar from the model logits that,
        when differentiated, gives the gradient of the target class
        logit w.r.t. the input.

        Args:
            logits: Model output logits, shape [batch, num_classes] or
                [batch, 1].
            target: Target tensor. For binary: [batch] or [1] with 0/1
                class indices. For multiclass/multilabel: [batch, num_classes]
                one-hot or multi-hot tensor.

        Returns:
            Scalar tensor for backpropagation.
        """
        target_f = target.to(logits.device).float()
        mode = self._prediction_mode()

        if mode == "binary":
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(-1)
            target_f = target_f.expand_as(logits)
            signs = 2.0 * target_f - 1.0
            return (signs * logits).sum()
        else:
            # multiclass or multilabel: target is one-hot/multi-hot
            while target_f.dim() < logits.dim():
                target_f = target_f.unsqueeze(0)
            target_f = target_f.expand_as(logits)
            return (target_f * logits).sum()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_to_input_shapes(
        attr_values: Dict[str, torch.Tensor],
        input_shapes: dict,
    ) -> Dict[str, torch.Tensor]:
        """Map attributions back to original input tensor shapes.

        For embedding-based attributions, the embedding dimension has
        already been summed out. This method handles any remaining
        shape mismatches (e.g., expanding scalar attributions to match
        multi-dimensional inputs).

        Args:
            attr_values: Dictionary of attribution tensors.
            input_shapes: Dictionary of original input shapes.

        Returns:
            Dictionary of attributions reshaped to match original inputs.
        """
        mapped: dict[str, torch.Tensor] = {}
        for key, values in attr_values.items():
            if key not in input_shapes:
                mapped[key] = values
                continue

            orig_shape = input_shapes[key]

            if values.shape == orig_shape:
                mapped[key] = values
                continue

            reshaped = values
            while len(reshaped.shape) < len(orig_shape):
                reshaped = reshaped.unsqueeze(-1)

            if reshaped.shape != orig_shape:
                reshaped = reshaped.expand(orig_shape)

            mapped[key] = reshaped

        return mapped
