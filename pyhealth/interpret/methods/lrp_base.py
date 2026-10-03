"""Base classes and layer handlers for Layer-wise Relevance Propagation (LRP).

Provides abstract base classes, a handler registry, and concrete LRP handlers
for common layer types (Linear, Conv2d, pooling, normalization, embedding).

References:
    Binder et al., "Layer-wise Relevance Propagation for Neural Networks
    with Local Renormalization Layers", arXiv:1604.00825, 2016.
"""

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn

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

    Examples:
        >>> import torch
        >>> from pyhealth.interpret.methods.lrp_base import stabilize_denominator
        >>> z = torch.tensor([[2.0, -2.0, 0.0]])
        >>> out = stabilize_denominator(z, epsilon=0.1, rule="epsilon")
        >>> bool(torch.allclose(out, torch.tensor([[2.1, -2.1, 0.1]])))
        True
        >>> bool((stabilize_denominator(z, 0.1, rule="z+") >= 0.1).all())
        True
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
    """Check tensor for NaN or Inf values.

    Examples:
        >>> import torch
        >>> from pyhealth.interpret.methods.lrp_base import check_tensor_validity
        >>> check_tensor_validity(torch.ones(2, 2), "relevance")
        True
        >>> import logging
        >>> logging.disable(logging.ERROR)  # silence the diagnostic log line
        >>> check_tensor_validity(torch.tensor([float("nan")]), "relevance")
        False
        >>> logging.disable(logging.NOTSET)
    """
    if torch.isnan(tensor).any().item():
        logger.error(f"{name} contains NaN values!")
        return False
    if torch.isinf(tensor).any().item():
        logger.error(f"{name} contains Inf values!")
        return False
    return True


def pad_to_match(
    x: torch.Tensor, expected_size: int
) -> tuple[torch.Tensor, int]:
    """Pad or truncate *x* along dim=1 to *expected_size*.

    Returns:
        Tuple of (adjusted_x, original_size).

    Examples:
        >>> import torch
        >>> from pyhealth.interpret.methods.lrp_base import pad_to_match
        >>> padded, original = pad_to_match(torch.ones(1, 3), 5)
        >>> padded.shape, original
        (torch.Size([1, 5]), 3)
        >>> truncated, original = pad_to_match(torch.ones(1, 6), 4)
        >>> truncated.shape, original
        (torch.Size([1, 4]), 6)
    """
    orig = x.size(1)
    if orig == expected_size:
        return x, orig
    if orig < expected_size:
        pad = torch.zeros(
            x.size(0), expected_size - orig, device=x.device, dtype=x.dtype
        )
        return torch.cat([x, pad], dim=1), orig
    return x[:, :expected_size], orig


def match_relevance_dim(
    relevance: torch.Tensor, target_size: int
) -> torch.Tensor:
    """Pad or truncate *relevance* along dim=1 to *target_size*.

    Examples:
        >>> import torch
        >>> from pyhealth.interpret.methods.lrp_base import match_relevance_dim
        >>> match_relevance_dim(torch.ones(1, 5), 3).shape
        torch.Size([1, 3])
        >>> match_relevance_dim(torch.ones(1, 2), 4).shape
        torch.Size([1, 4])
    """
    current = relevance.size(1)
    if current == target_size:
        return relevance
    if current < target_size:
        pad = torch.zeros(
            relevance.size(0),
            target_size - current,
            device=relevance.device,
            dtype=relevance.dtype,
        )
        return torch.cat([relevance, pad], dim=1)
    return relevance[:, :target_size]


def conv_output_padding(
    layer: nn.Module, z_shape: tuple, x_shape: tuple
) -> tuple:
    """Compute output_padding for conv_transpose2d to recover input spatial size.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import conv_output_padding
        >>> layer = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        >>> x = torch.randn(1, 1, 8, 8)
        >>> conv_output_padding(layer, layer(x).shape, x.shape)
        (1, 1)
    """
    pads = []
    for i in range(2):
        s = layer.stride[i] if isinstance(layer.stride, tuple) else layer.stride
        p = layer.padding[i] if isinstance(layer.padding, tuple) else layer.padding
        k = layer.weight.shape[2 + i]
        expected = (z_shape[2 + i] - 1) * s - 2 * p + k
        pads.append(max(0, x_shape[2 + i] - expected))
    return tuple(pads)


def crop_spatial(tensor: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """Crop or zero-pad a 4-D tensor to match *target_shape* in the spatial dims.

    Examples:
        >>> import torch
        >>> from pyhealth.interpret.methods.lrp_base import crop_spatial
        >>> crop_spatial(torch.ones(1, 1, 8, 8), (1, 1, 5, 5)).shape
        torch.Size([1, 1, 5, 5])
        >>> crop_spatial(torch.ones(1, 1, 3, 3), (1, 1, 4, 4)).shape
        torch.Size([1, 1, 4, 4])
    """
    if tensor.shape[2:] == target_shape[2:]:
        return tensor
    if tensor.shape[2] > target_shape[2] or tensor.shape[3] > target_shape[3]:
        return tensor[:, :, : target_shape[2], : target_shape[3]]
    pad_h = target_shape[2] - tensor.shape[2]
    pad_w = target_shape[3] - tensor.shape[3]
    return F.pad(tensor, (0, pad_w, 0, pad_h))


# ============================================================================
# Abstract base class and registry
# ============================================================================


class LRPLayerHandler(ABC):
    """Abstract base class for layer-specific LRP propagation rules.

    Each handler implements forward activation caching and backward
    relevance propagation for a specific layer type.

    Examples:
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import (
        ...     LinearLRPHandler,
        ...     LRPLayerHandler,
        ... )
        >>> issubclass(LinearLRPHandler, LRPLayerHandler)
        True
        >>> LinearLRPHandler().supports(nn.Linear(4, 2))
        True
    """

    def __init__(self, name: str):
        self.name = name
        self.activations_cache = {}

    @abstractmethod
    def supports(self, layer: nn.Module) -> bool:
        """Return True if this handler can process the given layer type."""

    def forward_hook(
        self, module: nn.Module, input: tuple, output: torch.Tensor
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

    def _get_cached(self, layer: nn.Module) -> dict:
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError(
                f"forward_hook not called for {type(layer).__name__}. "
                f"Run forward pass before backward_relevance."
            )
        return self.activations_cache[module_id]


class LRPHandlerRegistry:
    """Registry that maps layer types to their LRP handlers.

    Examples:
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import (
        ...     LinearLRPHandler,
        ...     LRPHandlerRegistry,
        ... )
        >>> registry = LRPHandlerRegistry()
        >>> registry.register(LinearLRPHandler())
        >>> registry.get_handler(nn.Linear(4, 2)).name
        'LinearHandler'
        >>> registry.get_handler(nn.LayerNorm(4)) is None
        True
    """

    def __init__(self):
        self._handlers: list[LRPLayerHandler] = []
        self._cache: dict[type, LRPLayerHandler] = {}

    def register(self, handler: LRPLayerHandler) -> None:
        if not isinstance(handler, LRPLayerHandler):
            raise TypeError(f"Expected LRPLayerHandler, got {type(handler)}")
        self._handlers.append(handler)
        self._cache.clear()

    def get_handler(self, layer: nn.Module) -> LRPLayerHandler | None:
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

    Handles higher-dimensional inputs by flattening to 2-D and pads/truncates
    when the input width does not perfectly match the weight matrix width.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import LinearLRPHandler
        >>> layer, handler = nn.Linear(4, 2), LinearLRPHandler()
        >>> x = torch.randn(1, 4)
        >>> handler.forward_hook(layer, (x,), layer(x))
        >>> handler.backward_relevance(layer, torch.ones(1, 2), rule="epsilon").shape
        torch.Size([1, 4])
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
        alpha: float = 1.0,
        beta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._get_cached(layer)
        x = cache["input"]
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        if rule == "epsilon":
            return self._epsilon_rule(layer, x, relevance_output, epsilon)
        elif rule == "alphabeta":
            return self._alphabeta_rule(layer, x, relevance_output, alpha, beta)
        raise ValueError(f"Unsupported rule: {rule}")

    def _epsilon_rule(self, layer, x, relevance_output, epsilon):
        x_p, orig = pad_to_match(x, layer.weight.size(1))
        z = F.linear(x_p, layer.weight, layer.bias)
        relevance_output = match_relevance_dim(relevance_output, z.size(1))
        z = stabilize_denominator(z, epsilon, rule="epsilon")
        c = torch.einsum("bo,oi->bi", relevance_output / z, layer.weight)
        result = x_p * c
        if result.size(1) != orig:
            result = result[:, :orig]
        return result

    def _alphabeta_rule(self, layer, x, relevance_output, alpha, beta):
        # The alpha-beta rule splits each *contribution* z_ij = x_i * w_ij into
        # its positive and negative parts -- not the weights alone. For x_i < 0 a
        # positive weight produces a negative contribution, so pairing W+ with a
        # signed input mixes the two paths and lets relevance change sign even
        # when beta == 0. Split the input as well:
        #     z+ = x+ @ W+ + x- @ W-      (both products positive)
        #     z- = x+ @ W- + x- @ W+      (both products negative)
        x_p, orig = pad_to_match(x, layer.weight.size(1))
        W_pos = torch.clamp(layer.weight, min=0)
        W_neg = torch.clamp(layer.weight, max=0)
        b_pos = torch.clamp(layer.bias, min=0) if layer.bias is not None else None
        b_neg = torch.clamp(layer.bias, max=0) if layer.bias is not None else None
        x_pos = torch.clamp(x_p, min=0)
        x_neg = torch.clamp(x_p, max=0)

        z_pos = F.linear(x_pos, W_pos, b_pos) + F.linear(x_neg, W_neg, None) + 1e-9
        z_neg = F.linear(x_pos, W_neg, b_neg) + F.linear(x_neg, W_pos, None) - 1e-9

        relevance_output = match_relevance_dim(relevance_output, z_pos.size(1))
        s_pos = relevance_output / z_pos
        s_neg = relevance_output / z_neg

        c_pos = x_pos * torch.einsum("bo,oi->bi", s_pos, W_pos) + x_neg * torch.einsum(
            "bo,oi->bi", s_pos, W_neg
        )
        c_neg = x_pos * torch.einsum("bo,oi->bi", s_neg, W_neg) + x_neg * torch.einsum(
            "bo,oi->bi", s_neg, W_pos
        )

        result = alpha * c_pos - beta * c_neg
        if result.size(1) != orig:
            result = result[:, :orig]
        return result


class ReLULRPHandler(LRPLayerHandler):
    """LRP handler for nn.ReLU — relevance passes through unchanged.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import ReLULRPHandler
        >>> handler = ReLULRPHandler()
        >>> handler.supports(nn.ReLU())
        True
        >>> relevance = torch.tensor([[1.0, -2.0]])
        >>> bool(torch.equal(handler.backward_relevance(nn.ReLU(), relevance), relevance))
        True
    """

    def __init__(self):
        super().__init__(name="ReLUHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.ReLU)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output


class EmbeddingLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Embedding — sum over embedding dim.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import EmbeddingLRPHandler
        >>> handler = EmbeddingLRPHandler()
        >>> handler.supports(nn.Embedding(10, 4))
        True
        >>> handler.backward_relevance(nn.Embedding(10, 4), torch.ones(1, 3, 4)).shape
        torch.Size([1, 3])
    """

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
    """LRP handler for nn.Conv2d layers (epsilon and alphabeta rules).

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import Conv2dLRPHandler
        >>> layer, handler = nn.Conv2d(1, 2, kernel_size=3, padding=1), Conv2dLRPHandler()
        >>> x = torch.randn(1, 1, 8, 8)
        >>> z = layer(x)
        >>> handler.forward_hook(layer, (x,), z)
        >>> handler.backward_relevance(layer, torch.ones_like(z)).shape
        torch.Size([1, 1, 8, 8])
    """

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
        alpha: float = 1.0,
        beta: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        cache = self._get_cached(layer)
        x = cache["input"]
        if rule == "epsilon":
            return self._epsilon_rule(layer, x, relevance_output, epsilon)
        elif rule == "alphabeta":
            return self._alphabeta_rule(layer, x, relevance_output, alpha, beta, epsilon)
        raise ValueError(f"Unsupported rule: {rule}")

    def _epsilon_rule(self, layer, x, relevance_output, epsilon):
        conv_kw = {
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "groups": layer.groups,
        }
        z = F.conv2d(x, layer.weight, layer.bias, **conv_kw)
        z = stabilize_denominator(z, epsilon, rule="epsilon")
        s = relevance_output / z
        out_pad = conv_output_padding(layer, z.shape, x.shape)
        c = F.conv_transpose2d(
            s, layer.weight, None,
            stride=layer.stride, padding=layer.padding,
            output_padding=out_pad, dilation=layer.dilation, groups=layer.groups,
        )
        return x * crop_spatial(c, x.shape)

    def _alphabeta_rule(self, layer, x, relevance_output, alpha, beta, epsilon):
        conv_kw = {
            "stride": layer.stride,
            "padding": layer.padding,
            "dilation": layer.dilation,
            "groups": layer.groups,
        }
        W_pos = torch.clamp(layer.weight, min=0)
        W_neg = torch.clamp(layer.weight, max=0)
        b_pos = torch.clamp(layer.bias, min=0) if layer.bias is not None else None
        b_neg = torch.clamp(layer.bias, max=0) if layer.bias is not None else None

        # Split the input as well as the weights, so that each path collects only
        # same-signed contributions z_ij = x_i * w_ij (see LinearLRPHandler):
        #   z+ = conv(x+, W+) + conv(x-, W-)
        #   z- = conv(x+, W-) + conv(x-, W+)
        # Pairing W+ with a signed input would mix the two and let relevance flip
        # sign even when beta == 0.
        x_pos = torch.clamp(x, min=0)
        x_neg = torch.clamp(x, max=0)

        z_pos = F.conv2d(x_pos, W_pos, b_pos, **conv_kw) + F.conv2d(
            x_neg, W_neg, None, **conv_kw
        )
        z_neg = F.conv2d(x_pos, W_neg, b_neg, **conv_kw) + F.conv2d(
            x_neg, W_pos, None, **conv_kw
        )
        denom_pos = z_pos + epsilon
        denom_neg = z_neg - epsilon

        out_pad = conv_output_padding(layer, z_pos.shape, x.shape)
        trans_kw = {
            "stride": layer.stride, "padding": layer.padding,
            "output_padding": out_pad, "dilation": layer.dilation,
            "groups": layer.groups,
        }
        s_pos = relevance_output / denom_pos
        s_neg = relevance_output / denom_neg

        def _back(s, W):
            return crop_spatial(
                F.conv_transpose2d(s, W, None, **trans_kw), x.shape
            )

        c_pos = x_pos * _back(s_pos, W_pos) + x_neg * _back(s_pos, W_neg)
        c_neg = x_pos * _back(s_neg, W_neg) + x_neg * _back(s_neg, W_pos)
        return alpha * c_pos - beta * c_neg


class MaxPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.MaxPool2d — winner-take-all relevance routing.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import MaxPool2dLRPHandler
        >>> layer, handler = nn.MaxPool2d(kernel_size=2, stride=2), MaxPool2dLRPHandler()
        >>> x = torch.randn(1, 1, 4, 4)
        >>> z = layer(x)
        >>> handler.forward_hook(layer, (x,), z)
        >>> handler.backward_relevance(layer, torch.ones_like(z)).shape
        torch.Size([1, 1, 4, 4])
    """

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
    """LRP handler for nn.AvgPool2d — uniform relevance distribution via transposed conv.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import AvgPool2dLRPHandler
        >>> layer, handler = nn.AvgPool2d(kernel_size=2, stride=2), AvgPool2dLRPHandler()
        >>> x = torch.randn(1, 1, 4, 4)
        >>> z = layer(x)
        >>> handler.forward_hook(layer, (x,), z)
        >>> handler.backward_relevance(layer, torch.ones_like(z)).shape
        torch.Size([1, 1, 4, 4])
    """

    def __init__(self):
        super().__init__(name="AvgPool2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.AvgPool2d)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        cache = self._get_cached(layer)
        x = cache["input"]
        channels = relevance_output.size(1)
        ks = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        st = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
        pd = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
        weight = torch.ones(channels, 1, *ks, device=x.device, dtype=x.dtype) / (ks[0] * ks[1])
        result = F.conv_transpose2d(relevance_output, weight, stride=st, padding=pd, groups=channels)
        return crop_spatial(result, x.shape)


class AdaptiveAvgPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.AdaptiveAvgPool2d — uniform distribution.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import AdaptiveAvgPool2dLRPHandler
        >>> layer = nn.AdaptiveAvgPool2d((1, 1))
        >>> handler = AdaptiveAvgPool2dLRPHandler()
        >>> x = torch.randn(1, 2, 6, 6)
        >>> z = layer(x)
        >>> handler.forward_hook(layer, (x,), z)
        >>> handler.backward_relevance(layer, torch.ones_like(z)).shape
        torch.Size([1, 2, 6, 6])
    """

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
    """LRP handler for nn.Flatten — reshape relevance back.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import FlattenLRPHandler
        >>> layer, handler = nn.Flatten(), FlattenLRPHandler()
        >>> x = torch.randn(1, 2, 3, 3)
        >>> z = layer(x)
        >>> handler.forward_hook(layer, (x,), z)
        >>> handler.backward_relevance(layer, torch.ones_like(z)).shape
        torch.Size([1, 2, 3, 3])
    """

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
    """LRP handler for nn.BatchNorm2d — pass relevance through.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import BatchNorm2dLRPHandler
        >>> handler = BatchNorm2dLRPHandler()
        >>> handler.supports(nn.BatchNorm2d(3))
        True
        >>> relevance = torch.ones(1, 3, 4, 4)
        >>> out = handler.backward_relevance(nn.BatchNorm2d(3), relevance)
        >>> bool(torch.equal(out, relevance))
        True
    """

    def __init__(self):
        super().__init__(name="BatchNorm2dHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.BatchNorm2d)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output


class DropoutLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Dropout — identity in eval mode.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import DropoutLRPHandler
        >>> handler = DropoutLRPHandler()
        >>> handler.supports(nn.Dropout(0.5))
        True
        >>> relevance = torch.ones(1, 4)
        >>> bool(torch.equal(handler.backward_relevance(nn.Dropout(), relevance), relevance))
        True
    """

    def __init__(self):
        super().__init__(name="DropoutHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, nn.Dropout)

    def backward_relevance(self, layer, relevance_output, **kwargs):
        return relevance_output


class RNNLRPHandler(LRPLayerHandler):
    """LRP handler for nn.LSTM and nn.GRU.

    LRP for recurrent layers is an active research area; this implementation
    uses the simple heuristic of distributing the hidden-state relevance
    uniformly across all input time-steps, which is consistent with the
    epsilon-rule interpretation of the recurrent read-out.

    Examples:
        >>> import torch
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import RNNLRPHandler
        >>> layer = nn.LSTM(input_size=4, hidden_size=3, batch_first=True)
        >>> handler = RNNLRPHandler()
        >>> x = torch.randn(1, 5, 4)
        >>> out, _ = layer(x)
        >>> handler.forward_hook(layer, (x,), out)
        >>> handler.backward_relevance(layer, torch.ones(1, 3)).shape
        torch.Size([1, 5, 3])
    """

    def __init__(self):
        super().__init__(name="RNNHandler")

    def supports(self, layer: nn.Module) -> bool:
        return isinstance(layer, (nn.LSTM, nn.GRU))

    def backward_relevance(self, layer, relevance_output, **kwargs):
        cache = self._get_cached(layer)
        x = cache["input"]
        if x.dim() == 3:
            # relevance_output: [B, H] → distribute equally over T timesteps.
            # Dividing by T preserves the total relevance sum (Eq. 2 conservation).
            T = x.shape[1]
            return (relevance_output.unsqueeze(1) / T).expand(x.shape[0], T, -1).contiguous()
        return relevance_output


# ============================================================================
# Default registry
# ============================================================================


def create_default_registry() -> LRPHandlerRegistry:
    """Create a registry with handlers for all supported layer types.

    Examples:
        >>> from torch import nn
        >>> from pyhealth.interpret.methods.lrp_base import create_default_registry
        >>> registry = create_default_registry()
        >>> registry.get_handler(nn.Linear(4, 2)).name
        'LinearHandler'
        >>> registry.get_handler(nn.Conv2d(1, 1, 3)).name
        'Conv2dHandler'
    """
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
    registry.register(RNNLRPHandler())
    return registry
