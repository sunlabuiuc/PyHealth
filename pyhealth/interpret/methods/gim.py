from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import torch

from pyhealth.models import BaseModel

from .base_interpreter import BaseInterpreter


class _TemperatureSoftmax(torch.autograd.Function):
    """Custom autograd op implementing temperature-adjusted softmax gradients."""

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


class _GIMActivationHooks:
    """Router that swaps selected activations for GIM-aware variants."""

    def __init__(self, temperature: float = 2.0):
        self.temperature = temperature

    def apply(self, name: str, tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        if name == "softmax" and self.temperature is not None:
            dim = kwargs.get("dim", -1)
            temp = max(float(self.temperature), 1.0)
            return _TemperatureSoftmax.apply(tensor, dim, temp)
        fn = getattr(torch, name)
        return fn(tensor, **kwargs)


class _GIMHookContext(contextlib.AbstractContextManager):
    """Context manager that wires GIM hooks if the model supports them."""

    def __init__(self, model: BaseModel, temperature: float):
        self.model = model
        self.temperature = temperature
        self.hooks: Optional[_GIMActivationHooks] = None
        self._set_fn = None
        self._clear_fn = None

        # Prefer explicit GIM hooks if the model exposes them, otherwise
        # reuse the DeepLIFT hook wiring which StageNet already supports.
        if hasattr(model, "set_gim_hooks") and hasattr(model, "clear_gim_hooks"):
            self._set_fn = model.set_gim_hooks
            self._clear_fn = model.clear_gim_hooks
        elif hasattr(model, "set_deeplift_hooks") and hasattr(model, "clear_deeplift_hooks"):
            self._set_fn = model.set_deeplift_hooks
            self._clear_fn = model.clear_deeplift_hooks

    def __enter__(self) -> "_GIMHookContext":
        if self._set_fn is not None and self.temperature > 1.0:
            self.hooks = _GIMActivationHooks(temperature=self.temperature)
            self._set_fn(self.hooks)
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        if self._clear_fn is not None and self.hooks is not None:
            self._clear_fn()
        self.hooks = None
        return False


class GIM(BaseInterpreter):
    """Gradient Interaction Modifications for StageNet-style models.

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
       so this step is effectively a no-op but kept for API parity.
    3. **Gradient normalization:** Gradients are reported as gradient-timesinput and collapsed onto the original token axes, ensuring consistent scale
       across visits. This mirrors the normalization heuristic proposed in
       the paper for multiplicative interactions.

    Args:
        model: Trained PyHealth model supporting ``forward_from_embedding``
            (StageNet is currently supported).
        temperature: Softmax temperature used exclusively for the backward
            pass. A value of ``2.0`` matches the paper's best setting.
        multiply_by_input: Whether to return gradient×input (default) or raw
            gradients in embedding space.
    """

    def __init__(
        self,
        model: BaseModel,
        temperature: float = 2.0,
        multiply_by_input: bool = True,
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
        self.multiply_by_input = multiply_by_input

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

        time_kwarg = time_info if time_info else None
        with _GIMHookContext(self.model, self.temperature):
            forward_kwargs = {**label_data} if label_data else {}
            output = self.model.forward_from_embedding(
                feature_embeddings=embeddings,
                time_info=time_kwarg,
                **forward_kwargs,
            )

        logits = output["logit"]
        target = self._compute_target_output(logits, target_class_idx)

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
            attr = grad
            if self.multiply_by_input:
                attr = attr * emb
            token_attr = self._collapse_to_input_shape(attr, input_shapes[key])
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
