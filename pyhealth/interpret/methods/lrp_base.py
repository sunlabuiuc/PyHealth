"""Base classes and layer handlers for Layer-wise Relevance Propagation (LRP).

Provides abstract base classes, a handler registry, and concrete LRP handlers
for common layer types (Linear, Conv2d, pooling, normalization, embedding).

References:
    Binder et al., "Layer-wise Relevance Propagation for Neural Networks
    with Local Renormalization Layers", arXiv:1604.00825, 2016.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Utility functions
# ============================================================================


def stabilize_denominator(
    z: torch.Tensor,
    epsilon: float = 1e-2,
    rule: str = "epsilon",
) -> torch.Tensor:
    """Stabilize denominator to prevent division by zero.

    Args:
        z: Tensor to stabilize.
        epsilon: Stabilization parameter.
        rule: LRP rule ('epsilon' or 'z+').

    Returns:
        Stabilized tensor safe for division.
    """
    if rule == "epsilon":
        sign = z.sign()
        # sign(0) == 0 would make the denominator zero; treat zero as positive
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return z + epsilon * sign
    elif rule == "z+":
        return torch.clamp(z, min=epsilon)
    return z + epsilon


def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check tensor for NaN or Inf values."""
    if torch.isnan(tensor).any().item():
        logger.error(f"{name} contains NaN values!")
        return False
    if torch.isinf(tensor).any().item():
        logger.error(f"{name} contains Inf values!")
        return False
    return True


# ============================================================================
# Abstract base class and registry
# ============================================================================


class LRPLayerHandler(ABC):
    """Abstract base class for layer-specific LRP propagation rules.

    Each handler implements forward activation caching and backward
    relevance propagation for a specific layer type.
    """

    def __init__(self, name: str):
        self.name = name
        self.activations_cache = {}

    @abstractmethod
    def supports(self, layer: nn.Module) -> bool:
        """Return True if this handler can process the given layer type."""

    def forward_hook(
        self, module: nn.Module, input: Tuple, output: torch.Tensor
    ) -> None:
        """Cache input/output activations during forward pass."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        self.activations_cache[id(module)] = {
            "input": input_tensor.detach(),
            "output": output.detach(),
        }

    @abstractmethod
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs,
    ) -> torch.Tensor:
        """Propagate relevance backward through the layer.

        Args:
            layer: PyTorch module to propagate through.
            relevance_output: Relevance scores at layer output.
            rule: LRP rule ('epsilon', 'alphabeta').
            **kwargs: Rule-specific parameters (epsilon, alpha, beta).

        Returns:
            Relevance scores at layer input.
        """

    def clear_cache(self):
        self.activations_cache.clear()

    def _get_cached(self, layer: nn.Module) -> Dict:
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError(
                f"forward_hook not called for {type(layer).__name__}. "
                f"Run forward pass before backward_relevance."
            )
        return self.activations_cache[module_id]


class LRPHandlerRegistry:
    """Registry that maps layer types to their LRP handlers."""

    def __init__(self):
        self._handlers: List[LRPLayerHandler] = []
        self._cache: Dict[type, LRPLayerHandler] = {}

    def register(self, handler: LRPLayerHandler) -> None:
        if not isinstance(handler, LRPLayerHandler):
            raise TypeError(f"Expected LRPLayerHandler, got {type(handler)}")
        self._handlers.append(handler)
        self._cache.clear()

    def get_handler(self, layer: nn.Module) -> Optional[LRPLayerHandler]:
        layer_type = type(layer)
        if layer_type in self._cache:
            return self._cache[layer_type]
        for handler in self._handlers:
            if handler.supports(layer):
                self._cache[layer_type] = handler
                return handler
        return None

    def clear(self) -> None:
        self._handlers.clear()
        self._cache.clear()


# ============================================================================
# Embedding & activation handlers
# ============================================================================


class LinearLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Linear layers.

    Epsilon rule: R_i = sum_j (z_ij / (z_j + eps*sign(z_j))) * R_j
    AlphaBeta rule: R_i = sum_j [alpha*z_ij+ / z_j+ - beta*z_ij- / z_j-] * R_j
    """

    def __init__(self):
        super().__init__(name="LinearHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Linear)

    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        epsilon: float = 1e-2,
        alpha: float = 2.0,
        beta: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._get_cached(layer)
        x = cache["input"]
        if rule == "epsilon":
            return self._epsilon_rule(layer, x, relevance_output, epsilon)
        elif rule == "alphabeta":
            return self._alphabeta_rule(
                layer, x, relevance_output, alpha, beta, epsilon
            )
        raise ValueError(f"Unsupported rule: {rule}")

    def _epsilon_rule(self, layer, x, relevance_output, epsilon):
        w, b = layer.weight, layer.bias if layer.bias is not None else 0.0
        z = F.linear(x, w, b)
        z = stabilize_denominator(z, epsilon, rule="epsilon")
        return x * torch.mm(relevance_output / z, w)

    def _alphabeta_rule(self, layer, x, relevance_output, alpha, beta, epsilon):
        w = layer.weight
        b = layer.bias if layer.bias is not None else 0.0
        w_pos, w_neg = torch.clamp(w, min=0), torch.clamp(w, max=0)

        z_pos = F.linear(x, w_pos, torch.clamp(b, min=0)) + epsilon
        z_neg = F.linear(x, w_neg, torch.clamp(b, max=0)) - epsilon

        r_pos = alpha * x * torch.mm(relevance_output / z_pos, w_pos)
        r_neg = beta * x * torch.mm(relevance_output / z_neg, w_neg)
        return r_pos + r_neg


class ReLULRPHandler(LRPLayerHandler):
    """LRP handler for nn.ReLU — relevance passes through unchanged."""

    def __init__(self):
        super().__init__(name="ReLUHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.ReLU)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output


class EmbeddingLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Embedding — sum over embedding dim."""

    def __init__(self):
        super().__init__(name="EmbeddingHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Embedding)

    def forward_hook(self, module, input, output):
        input_tensor = input[0] if isinstance(input, tuple) else input
        self.activations_cache[id(module)] = {
            "indices": input_tensor.detach(),
            "output": output.detach(),
        }

    def backward_relevance(self, layer, relevance_output, **kwargs):
        if relevance_output.dim() == 3:
            return relevance_output.sum(dim=-1)
        return relevance_output


# ============================================================================
# CNN layer handlers
# ============================================================================


class Conv2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Conv2d layers (epsilon and alphabeta rules)."""

    def __init__(self):
        super().__init__(name="Conv2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Conv2d)

    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        epsilon: float = 1e-2,
        alpha: float = 2.0,
        beta: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._get_cached(layer)
        x = cache["input"]
        if rule == "epsilon":
            return self._epsilon_rule(layer, x, relevance_output, epsilon)
        elif rule == "alphabeta":
            return self._alphabeta_rule(
                layer, x, relevance_output, alpha, beta, epsilon
            )
        raise ValueError(f"Unsupported rule: {rule}")

    @staticmethod
    def _output_padding(layer, z_shape, x_shape):
        """Compute output_padding for conv_transpose2d to match input size."""
        pads = []
        for i in range(2):
            expected = (
                (z_shape[2 + i] - 1) * layer.stride[i]
                - 2 * layer.padding[i]
                + layer.kernel_size[i]
            )
            pads.append(max(0, x_shape[2 + i] - expected))
        return tuple(pads)

    def _epsilon_rule(self, layer, x, relevance_output, epsilon):
        conv_kw = dict(
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        z = F.conv2d(x, layer.weight, layer.bias, **conv_kw)
        z = stabilize_denominator(z, epsilon, rule="epsilon")
        s = relevance_output / z

        out_pad = self._output_padding(layer, z.shape, x.shape)
        c = F.conv_transpose2d(
            s, layer.weight, None, output_padding=out_pad, **conv_kw
        )
        return x * c

    def _alphabeta_rule(self, layer, x, relevance_output, alpha, beta, epsilon):
        conv_kw = dict(
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
        )
        w_pos = torch.clamp(layer.weight, min=0)
        w_neg = torch.clamp(layer.weight, max=0)
        b_pos = torch.clamp(layer.bias, min=0) if layer.bias is not None else None
        b_neg = torch.clamp(layer.bias, max=0) if layer.bias is not None else None

        z_pos = F.conv2d(x, w_pos, b_pos, **conv_kw) + epsilon
        z_neg = F.conv2d(x, w_neg, b_neg, **conv_kw) - epsilon

        out_pad = self._output_padding(layer, z_pos.shape, x.shape)
        trans_kw = dict(output_padding=out_pad, **conv_kw)

        r_pos = alpha * F.conv_transpose2d(
            relevance_output / z_pos * z_pos, w_pos, None, **trans_kw
        ) * x / (x + epsilon)
        r_neg = beta * F.conv_transpose2d(
            relevance_output / z_neg * z_neg, w_neg, None, **trans_kw
        ) * x / (x - epsilon)
        return r_pos + r_neg


class MaxPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.MaxPool2d — winner-take-all relevance routing."""

    def __init__(self):
        super().__init__(name="MaxPool2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.MaxPool2d)

    def forward_hook(self, module, input, output):
        input_tensor = input[0] if isinstance(input, tuple) else input
        _, indices = F.max_pool2d(
            input_tensor,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=True,
        )
        self.activations_cache[id(module)] = {
            "input": input_tensor.detach(),
            "output": output.detach(),
            "indices": indices,
        }

    def backward_relevance(self, layer, relevance_output, **kwargs):
        cache = self._get_cached(layer)
        input_shape = cache["input"].shape
        try:
            return F.max_unpool2d(
                relevance_output,
                cache["indices"],
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                output_size=input_shape,
            )
        except RuntimeError:
            return F.interpolate(
                relevance_output, size=input_shape[2:], mode="nearest"
            )


class AvgPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.AvgPool2d — uniform relevance distribution."""

    def __init__(self):
        super().__init__(name="AvgPool2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.AvgPool2d)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        input_shape = self._get_cached(layer)["input"].shape
        return F.interpolate(
            relevance_output, size=input_shape[2:], mode="nearest"
        )


class AdaptiveAvgPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.AdaptiveAvgPool2d — uniform distribution."""

    def __init__(self):
        super().__init__(name="AdaptiveAvgPool2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.AdaptiveAvgPool2d)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        cache = self._get_cached(layer)
        input_shape = cache["input"].shape
        output_shape = cache["output"].shape

        # Handle flattened relevance (e.g. after a Flatten layer)
        if relevance_output.dim() == 2 and len(output_shape) == 4:
            relevance_output = relevance_output.view(output_shape)

        in_h, in_w = input_shape[2], input_shape[3]
        out_h, out_w = relevance_output.shape[2], relevance_output.shape[3]

        if out_h == 1 and out_w == 1:
            return relevance_output.expand(-1, -1, in_h, in_w) / (in_h * in_w)

        result = torch.zeros_like(cache["input"])
        stride_h, stride_w = in_h / out_h, in_w / out_w
        for i in range(out_h):
            for j in range(out_w):
                h_s, h_e = int(i * stride_h), int((i + 1) * stride_h)
                w_s, w_e = int(j * stride_w), int((j + 1) * stride_w)
                region_size = (h_e - h_s) * (w_e - w_s)
                result[:, :, h_s:h_e, w_s:w_e] = (
                    relevance_output[:, :, i : i + 1, j : j + 1] / region_size
                )
        return result


class FlattenLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Flatten — reshape relevance back."""

    def __init__(self):
        super().__init__(name="FlattenHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Flatten)

    def forward_hook(self, module, input, output):
        input_tensor = input[0] if isinstance(input, tuple) else input
        self.activations_cache[id(module)] = {
            "input_shape": input_tensor.shape,
            "output": output.detach(),
        }

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output.view(self._get_cached(layer)["input_shape"])


class BatchNorm2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.BatchNorm2d — pass relevance through."""

    def __init__(self):
        super().__init__(name="BatchNorm2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.BatchNorm2d)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output


class DropoutLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Dropout — identity in eval mode."""

    def __init__(self):
        super().__init__(name="DropoutHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Dropout)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output


class AdditionLRPHandler(LRPLayerHandler):
    """LRP handler for skip connections: splits relevance proportionally."""

    def __init__(self):
        super().__init__(name="AdditionHandler")
        self.branch_cache = {}

    def supports(self, layer: nn.Module) -> bool:
        return False  # Manually invoked

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output

    def cache_branches(self, op_id: int, branch_a: torch.Tensor, branch_b: torch.Tensor):
        self.branch_cache[op_id] = {
            "branch_a": branch_a.detach(),
            "branch_b": branch_b.detach(),
        }

    def backward_relevance_split(
        self, op_id: int, relevance_output: torch.Tensor, epsilon: float = 1e-9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if op_id not in self.branch_cache:
            raise RuntimeError(f"No cached branches for operation {op_id}")
        a = self.branch_cache[op_id]["branch_a"]
        b = self.branch_cache[op_id]["branch_b"]
        z = stabilize_denominator(a + b, epsilon, rule="epsilon")
        return (a / z) * relevance_output, (b / z) * relevance_output


# ============================================================================
# Default registry
# ============================================================================


def create_default_registry() -> LRPHandlerRegistry:
    """Create a registry with handlers for all supported layer types."""
    registry = LRPHandlerRegistry()
    registry.register(LinearLRPHandler())
    registry.register(ReLULRPHandler())
    registry.register(EmbeddingLRPHandler())
    registry.register(Conv2dLRPHandler())
    registry.register(MaxPool2dLRPHandler())
    registry.register(AvgPool2dLRPHandler())
    registry.register(AdaptiveAvgPool2dLRPHandler())
    registry.register(FlattenLRPHandler())
    registry.register(BatchNorm2dLRPHandler())
    registry.register(DropoutLRPHandler())
    return registry