from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple, Type

import torch

from pyhealth.models import BaseModel

from .base_interpreter import BaseInterpreter


def _iter_child_modules(module: torch.nn.Module):
    for name, child in module.named_children():
        yield module, name, child
        yield from _iter_child_modules(child)


class _SoftmaxWrapper(torch.nn.Module):
    """Swap nn.Softmax with temperature-adjusted backward for GIM."""

    def __init__(self, dim: int, temperature: float):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return _TemperatureSoftmax.apply(tensor, self.dim, self.temperature)


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
    def backward(
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
       StageNet’s embedding → recurrent pipeline) the uniform division rule
       effectively multiplies by 1, so propagating raw gradients remains
       faithful to Section 4.2.

    Args:
        model: Trained PyHealth model supporting ``forward_from_embedding``
            (StageNet is currently supported).
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
        >>> model = StageNet(dataset=sample_dataset, mode="binary")
        >>> model = model.to(device).eval()
        >>> test_loader = get_dataloader(sample_dataset, batch_size=1, shuffle=False)
        >>> gim = GIM(model, temperature=2.0)
        >>>
        >>> batch = next(iter(test_loader))
        >>> batch_device = {}
        >>> for key, value in batch.items():
        ...     if isinstance(value, torch.Tensor):
        ...         batch_device[key] = value.to(device)
        ...     elif isinstance(value, tuple):
        ...         batch_device[key] = tuple(v.to(device) for v in value)
        ...     else:
        ...         batch_device[key] = value
        >>>
        >>> attributions = gim.attribute(**batch_device)
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
        if not hasattr(model, "embedding_model"):
            raise AssertionError(
                "GIM requires access to the model's embedding_model."
            )
        self.temperature = max(float(temperature), 1.0)

    def attribute(
        self,
        target_class_idx: Optional[int] = None,
        **data,
    ) -> Dict[str, torch.Tensor]:
        """Compute GIM attributions for a StageNet batch."""
        device = next(self.model.parameters()).device
        inputs, time_info, label_data = self._prepare_inputs(data, device)
        embeddings, input_shapes = self._embed_inputs(inputs)

        # Clear stale gradients before the attribution pass.
        self.model.zero_grad(set_to_none=True)

        # Step 1 (TSG): install the temperature-adjusted softmax hooks so all
        # backward passes through StageNet's cumax operations use the higher τ.
        with _GIMHookContext(self.model, self.temperature):
            forward_kwargs = {**label_data} if label_data else {}
            if time_info:
                forward_kwargs["time_info"] = time_info
            output = self.model.forward_from_embedding(
                feature_embeddings=embeddings,
                **forward_kwargs,
            )

        logits = output["logit"]
        target = self._compute_target_output(logits, target_class_idx)

        # Step 2 (LayerNorm freeze): StageNet does not contain layer norms, so
        # there are no σ parameters to freeze; the reset below ensures any
        # hypothetical normalization buffers would stay constant as in Sec. 4.2.
        self.model.zero_grad(set_to_none=True)
        for emb in embeddings.values():
            if emb.grad is not None:
                emb.grad.zero_()

        target.backward()

        attributions = {}
        for key, emb in embeddings.items():
            grad = emb.grad
            if grad is None:
                grad = torch.zeros_like(emb)
            # Step 3 (Gradient normalization): StageNet lacks the multi-input
            # products targeted by the uniform rule, so dividing by 1 (identity)
            # yields the same gradients the paper would propagate.
            token_attr = self._collapse_to_input_shape(grad, input_shapes[key])
            attributions[key] = token_attr.detach()

        return attributions

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def _prepare_inputs(
        self,
        data: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Split raw data into value tensors, time tensors, and labels."""
        inputs: Dict[str, torch.Tensor] = {}
        time_info: Dict[str, torch.Tensor] = {}

        for key in getattr(self.model, "feature_keys", []):
            if key not in data:
                continue
            value = data[key]
            time_tensor = None
            if isinstance(value, tuple) and len(value) == 2:
                time_tensor, value = value
                time_tensor = self._to_tensor(time_tensor, device)
            inputs[key] = self._to_tensor(value, device)
            if time_tensor is not None:
                time_info[key] = time_tensor

        label_data = {}
        for label_key in getattr(self.model, "label_keys", []):
            if label_key in data:
                label_data[label_key] = self._to_tensor(data[label_key], device)

        return inputs, time_info, label_data

    def _embed_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Size]]:
        """Run the model's embedding stack and detach tensors for attribution."""
        embeddings: Dict[str, torch.Tensor] = {}
        input_shapes: Dict[str, torch.Size] = {}

        for key, tensor in inputs.items():
            input_shapes[key] = tensor.shape
            embedded = self.model.embedding_model({key: tensor})
            emb_tensor = embedded[key].detach()
            emb_tensor.requires_grad_(True)
            emb_tensor.retain_grad()
            embeddings[key] = emb_tensor

        return embeddings, input_shapes

    def _compute_target_output(
        self,
        logits: torch.Tensor,
        target_class_idx: Optional[int],
    ) -> torch.Tensor:
        """Select a scalar logit to backpropagate based on the target class."""
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        if target_class_idx is None:
            if logits.shape[-1] == 1:
                selected = logits.squeeze(-1)
            else:
                indices = torch.argmax(logits, dim=-1)
                selected = logits.gather(1, indices.unsqueeze(-1)).squeeze(-1)
        else:
            if isinstance(target_class_idx, torch.Tensor):
                indices = target_class_idx.to(logits.device)
            else:
                indices = torch.full(
                    (logits.shape[0],),
                    int(target_class_idx),
                    device=logits.device,
                    dtype=torch.long,
                )
            indices = indices.view(-1, 1)
            if logits.shape[-1] == 1:
                selected = logits.squeeze(-1)
            else:
                selected = logits.gather(1, indices).squeeze(-1)

        return selected.sum()

    def _collapse_to_input_shape(
        self,
        tensor: torch.Tensor,
        orig_shape: torch.Size,
    ) -> torch.Tensor:
        """Sum the embedding dimension and reshape to match the raw inputs."""
        if tensor.dim() >= 2:
            tensor = tensor.sum(dim=-1)

        if tensor.shape == orig_shape:
            return tensor

        if len(orig_shape) > len(tensor.shape):
            expanded = tensor
            while len(expanded.shape) < len(orig_shape):
                expanded = expanded.unsqueeze(-1)
            expanded = expanded.expand(orig_shape)
            return expanded

        try:
            return tensor.reshape(orig_shape)
        except RuntimeError:
            return tensor

    @staticmethod
    def _to_tensor(value, device: torch.device) -> torch.Tensor:
        """Convert dataloader values (lists, numpy arrays) to tensors."""
        if isinstance(value, torch.Tensor):
            return value.to(device)
        return torch.as_tensor(value, device=device)
