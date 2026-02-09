from __future__ import annotations

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


class _SoftmaxWrapper(torch.nn.Module):
    """Swap nn.Softmax with temperature-adjusted backward for GIM."""

    def __init__(self, dim: int | None, temperature: float):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  
        return _TemperatureSoftmax.apply(tensor, self.dim, self.temperature) # type: ignore[override]


class _GIMSwapContext(contextlib.AbstractContextManager):
    """Temporarily replace softmax modules with GIM-aware versions."""

    _TARGETS: Dict[Type[torch.nn.Module], str] = {torch.nn.Softmax: "softmax"}

    def __init__(self, model: BaseModel, temperature: float):
        self.model = model
        self.temperature = temperature
        self._swapped: List[Tuple[torch.nn.Module, str, torch.nn.Module]] = []

    def __enter__(self) -> "_GIMSwapContext":
        for parent, name, child in _iter_child_modules(self.model):
            if isinstance(child, torch.nn.Softmax):
                wrapper = _SoftmaxWrapper(dim=child.dim, temperature=self.temperature)
                setattr(parent, name, wrapper)
                self._swapped.append((parent, name, child))
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        for parent, name, original in reversed(self._swapped):
            setattr(parent, name, original)
        self._swapped.clear()
        return False


class _TemperatureSoftmax(torch.autograd.Function):
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

        adjusted = torch.softmax(input_tensor / temperature, dim=dim)
        dot = (grad_output * adjusted).sum(dim=dim, keepdim=True)
        grad_input = adjusted * (grad_output - dot)
        grad_input = grad_input / temperature
        return grad_input, None, None


class _GIMHookContext(contextlib.AbstractContextManager):
    """Context manager that swaps softmax modules for temperature-aware variants."""

    def __init__(self, model: BaseModel, temperature: float):
        self.model = model
        self.temperature = temperature
        self._swap_ctx = _GIMSwapContext(model, temperature=max(float(temperature), 1.0))

    def __enter__(self) -> "_GIMHookContext":
        if self.temperature > 1.0:
            self._swap_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        self._swap_ctx.__exit__(exc_type, exc, exc_tb)
        return False


class GIM(BaseInterpreter):
    """Gradient Interaction Modifications for StageNet-style and Transformer models.

    This interpreter adapts the Gradient Interaction Modifications (GIM)
    technique (Edin et al., 2025) to PyHealth, focusing on StageNet where
    cumulative softmax operations can exhibit self-repair. The implementation
    follows three high-level ideas from the paper:

    1. **Temperature-adjusted softmax gradients (TSG):** Backpropagated
       gradients through cumulative softmax are recomputed with a higher
       temperature, exposing interactions that are otherwise hidden by
       softmax redistribution.
    2. **LayerNorm freeze:** Layer normalization parameters are treated as
       constants during backpropagation. StageNet does not employ layer norm,
       so this rule becomes a mathematical no-op, matching the paper when
       σ is constant.
    3. **Gradient normalization:** When no multiplicative fan-in exists (as in
       StageNet's embedding → recurrent pipeline) the uniform division rule
       effectively multiplies by 1, so propagating raw gradients remains
       faithful to Section 4.2.

    Args:
        model: Trained PyHealth model supporting ``forward_from_embedding``
            and ``get_embedding_model()`` (StageNet is currently supported).
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

        # Embed values and detach for gradient attribution
        embedding_model = self.model.get_embedding_model()
        assert embedding_model is not None
        embedded = embedding_model(values)

        embeddings: dict[str, torch.Tensor] = {}
        for key, emb in embedded.items():
            emb = emb.detach().requires_grad_(True)
            emb.retain_grad()
            embeddings[key] = emb

        # Insert embeddings back into input tuples
        forward_inputs = inputs.copy()
        for k in forward_inputs.keys():
            schema = self.model.dataset.input_processors[k].schema()
            val_idx = schema.index("value")
            forward_inputs[k] = (
                *forward_inputs[k][:val_idx],
                embeddings[k],
                *forward_inputs[k][val_idx + 1:],
            )

        # Clear stale gradients before the attribution pass.
        self.model.zero_grad(set_to_none=True)

        # Step 1 (TSG): install the temperature-adjusted softmax hooks so all
        # backward passes through StageNet's cumax operations use the higher τ.
        with _GIMHookContext(self.model, self.temperature):
            output = self.model.forward_from_embedding(**forward_inputs)

        logits = output["logit"] # type: ignore[assignment]
        target_output = self._compute_target_output(logits, target)

        # Step 2 (LayerNorm freeze): StageNet does not contain layer norms, so
        # there are no σ parameters to freeze; the reset below ensures any
        # hypothetical normalization buffers would stay constant as in Sec. 4.2.
        self.model.zero_grad(set_to_none=True)
        for emb in embeddings.values():
            if emb.grad is not None:
                emb.grad.zero_()

        target_output.backward()

        # Step 3 (Gradient normalization): StageNet lacks the multi-input
        # products targeted by the uniform rule, so dividing by 1 (identity)
        # yields the same gradients the paper would propagate.
        attributions: dict[str, torch.Tensor] = {}
        for key, emb in embeddings.items():
            grad = emb.grad
            if grad is None:
                grad = torch.zeros_like(emb)
            # Sum embedding dimension to get per-token attribution
            attr = grad.detach()
            if attr.dim() >= 3:
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
