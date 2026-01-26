"""
Base classes and infrastructure for Layer-wise Relevance Propagation (LRP).

This module provides the core abstract classes and utilities for building
a unified LRP implementation that supports both CNNs (image data) and
embedding-based models (discrete medical codes).

Classes:
    LRPLayerHandler: Abstract base class for layer-specific LRP rules
    LRPHandlerRegistry: Registry for managing layer handlers
    ConservationValidator: Utility for validating relevance conservation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


# Configure logging
logger = logging.getLogger(__name__)


class LRPLayerHandler(ABC):
    """Abstract base class for layer-specific LRP propagation rules.
    
    Each concrete handler implements the LRP backward propagation rule
    for a specific layer type (e.g., Linear, Conv2d, MaxPool2d).
    
    The core LRP principle: relevance conservation
        Sum of input relevances ≈ Sum of output relevances
        
    Different rules (epsilon, alpha-beta, z+) provide different trade-offs
    between stability, sharpness, and interpretability.
    
    Attributes:
        name (str): Human-readable name for this handler
        supported_layers (List[type]): List of layer types this handler supports
    """
    
    def __init__(self, name: str):
        """Initialize the handler.
        
        Args:
            name: Descriptive name for this handler (e.g., "LinearHandler")
        """
        self.name = name
        self.activations_cache = {}
        logger.debug(f"Initialized {self.name}")
    
    @abstractmethod
    def supports(self, layer: nn.Module) -> bool:
        """Check if this handler supports a given layer.
        
        Args:
            layer: PyTorch module to check
            
        Returns:
            True if this handler can process this layer type
        """
        pass
    
    @abstractmethod
    def forward_hook(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """Forward hook to capture activations during forward pass.
        
        This is called automatically during the forward pass and should
        store any information needed for backward relevance propagation.
        
        Args:
            module: The layer being hooked
            input: Input tensor(s) to the layer
            output: Output tensor from the layer
        """
        pass
    
    @abstractmethod
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance backward through the layer.
        
        This is the core LRP computation. Given relevance at the layer's
        output (R_j), compute relevance at the layer's input (R_i).
        
        Conservation property: sum(R_i) ≈ sum(R_j)
        
        Args:
            layer: The PyTorch module to propagate through
            relevance_output: Relevance scores at layer output [R_j]
            rule: LRP rule to apply ('epsilon', 'alphabeta', 'z+', etc.)
            **kwargs: Rule-specific parameters:
                - epsilon: Stabilizer for epsilon rule (default: 1e-2)
                - alpha: Weight for positive contributions (default: 2.0)
                - beta: Weight for negative contributions (default: 1.0)
                
        Returns:
            relevance_input: Relevance scores at layer input [R_i]
            
        Raises:
            ValueError: If rule is not supported by this handler
            RuntimeError: If forward_hook wasn't called before this
        """
        pass
    
    def clear_cache(self):
        """Clear cached activations to free memory."""
        self.activations_cache.clear()
    
    def validate_conservation(
        self,
        relevance_input: torch.Tensor,
        relevance_output: torch.Tensor,
        tolerance: float = 0.01,
        layer_name: str = "unknown"
    ) -> Tuple[bool, float]:
        """Validate that conservation property holds.
        
        Checks: |sum(R_in) - sum(R_out)| / |sum(R_out)| < tolerance
        
        Args:
            relevance_input: Input relevance tensor
            relevance_output: Output relevance tensor
            tolerance: Maximum allowed relative error (default: 1%)
            layer_name: Name for logging
            
        Returns:
            Tuple of (is_valid, error_percentage)
        """
        sum_in = relevance_input.sum().item()
        sum_out = relevance_output.sum().item()
        
        if abs(sum_out) < 1e-10:
            logger.warning(f"{layer_name}: Output relevance near zero ({sum_out:.6e})")
            return True, 0.0
        
        error = abs(sum_in - sum_out)
        error_pct = error / abs(sum_out)
        
        is_valid = error_pct <= tolerance
        
        if not is_valid:
            logger.warning(
                f"{layer_name} [{self.name}]: Conservation violated! "
                f"Error: {error_pct*100:.2f}% "
                f"(in={sum_in:.6f}, out={sum_out:.6f})"
            )
        else:
            logger.debug(
                f"{layer_name} [{self.name}]: ✓ Conservation OK "
                f"(error: {error_pct*100:.4f}%)"
            )
        
        return is_valid, error_pct * 100


class LRPHandlerRegistry:
    """Registry for managing LRP layer handlers.
    
    This class maintains a registry of handlers for different layer types
    and provides automatic handler selection based on layer type.
    
    Usage:
        >>> registry = LRPHandlerRegistry()
        >>> registry.register(LinearLRPHandler())
        >>> registry.register(Conv2dLRPHandler())
        >>> 
        >>> # Automatic handler lookup
        >>> layer = nn.Linear(10, 5)
        >>> handler = registry.get_handler(layer)
        >>> print(handler.name)  # "LinearHandler"
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._handlers: List[LRPLayerHandler] = []
        self._layer_type_cache: Dict[type, LRPLayerHandler] = {}
        logger.info("Initialized LRP handler registry")
    
    def register(self, handler: LRPLayerHandler) -> None:
        """Register a new layer handler.
        
        Args:
            handler: Handler instance to register
            
        Raises:
            TypeError: If handler is not an LRPLayerHandler instance
        """
        if not isinstance(handler, LRPLayerHandler):
            raise TypeError(
                f"Handler must be an LRPLayerHandler, got {type(handler)}"
            )
        
        self._handlers.append(handler)
        self._layer_type_cache.clear()  # Invalidate cache
        logger.info(f"Registered handler: {handler.name}")
    
    def get_handler(self, layer: nn.Module) -> Optional[LRPLayerHandler]:
        """Get appropriate handler for a given layer.
        
        Args:
            layer: PyTorch module to find handler for
            
        Returns:
            Handler instance if found, None otherwise
        """
        # Check cache first
        layer_type = type(layer)
        if layer_type in self._layer_type_cache:
            return self._layer_type_cache[layer_type]
        
        # Search for compatible handler
        for handler in self._handlers:
            if handler.supports(layer):
                self._layer_type_cache[layer_type] = handler
                logger.debug(f"Handler for {layer_type.__name__}: {handler.name}")
                return handler
        
        logger.warning(f"No handler found for layer type: {layer_type.__name__}")
        return None
    
    def list_handlers(self) -> List[str]:
        """Get list of registered handler names.
        
        Returns:
            List of handler names
        """
        return [h.name for h in self._handlers]
    
    def clear(self) -> None:
        """Remove all registered handlers."""
        self._handlers.clear()
        self._layer_type_cache.clear()
        logger.info("Cleared all handlers")


class ConservationValidator:
    """Utility class for validating LRP conservation property.
    
    The conservation property states that relevance should be conserved
    at each layer: sum(R_input) ≈ sum(R_output).
    
    This validator tracks conservation across all layers and provides
    diagnostic information when violations occur.
    
    Usage:
        >>> validator = ConservationValidator(tolerance=0.01)
        >>> 
        >>> # Check conservation at each layer
        >>> is_valid = validator.validate(
        ...     layer_name="fc1",
        ...     relevance_input=R_in,
        ...     relevance_output=R_out
        ... )
        >>> 
        >>> # Get summary report
        >>> validator.print_summary()
    """
    
    def __init__(self, tolerance: float = 0.01, strict: bool = False):
        """Initialize validator.
        
        Args:
            tolerance: Maximum allowed relative error (default: 1%)
            strict: If True, raise exception on violations (default: False)
        """
        self.tolerance = tolerance
        self.strict = strict
        self.violations: List[Dict[str, Any]] = []
        self.validations: List[Dict[str, Any]] = []
        logger.info(f"Conservation validator initialized (tolerance: {tolerance*100}%)")
    
    def validate(
        self,
        layer_name: str,
        relevance_input: torch.Tensor,
        relevance_output: torch.Tensor,
        layer_type: str = "unknown"
    ) -> bool:
        """Validate conservation at a single layer.
        
        Args:
            layer_name: Name of the layer being validated
            relevance_input: Relevance at layer input
            relevance_output: Relevance at layer output
            layer_type: Type of layer (for diagnostics)
            
        Returns:
            True if conservation holds within tolerance
            
        Raises:
            RuntimeError: If strict=True and conservation is violated
        """
        sum_in = relevance_input.sum().item()
        sum_out = relevance_output.sum().item()
        
        # Handle near-zero output
        if abs(sum_out) < 1e-10:
            logger.warning(
                f"{layer_name}: Output relevance near zero, skipping validation"
            )
            return True
        
        error = abs(sum_in - sum_out)
        error_pct = error / abs(sum_out)
        
        record = {
            'layer_name': layer_name,
            'layer_type': layer_type,
            'sum_input': sum_in,
            'sum_output': sum_out,
            'error': error,
            'error_pct': error_pct * 100,
            'valid': error_pct <= self.tolerance
        }
        
        self.validations.append(record)
        
        if not record['valid']:
            self.violations.append(record)
            logger.error(
                f"❌ {layer_name} ({layer_type}): Conservation violated! "
                f"Error: {error_pct*100:.2f}% (tolerance: {self.tolerance*100}%)\n"
                f"   Input sum:  {sum_in:12.6f}\n"
                f"   Output sum: {sum_out:12.6f}\n"
                f"   Difference: {error:12.6f}"
            )
            
            if self.strict:
                raise RuntimeError(
                    f"Conservation property violated at {layer_name}: "
                    f"{error_pct*100:.2f}% error"
                )
        else:
            logger.debug(
                f"✓ {layer_name} ({layer_type}): Conservation OK "
                f"(error: {error_pct*100:.4f}%)"
            )
        
        return record['valid']
    
    def reset(self):
        """Clear validation history."""
        self.violations.clear()
        self.validations.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all validations.
        
        Returns:
            Dictionary containing:
                - total_validations: Number of layers validated
                - violations_count: Number of violations
                - max_error_pct: Maximum error percentage observed
                - avg_error_pct: Average error percentage
                - violation_rate: Percentage of layers with violations
        """
        if not self.validations:
            return {
                'total_validations': 0,
                'violations_count': 0,
                'max_error_pct': 0.0,
                'avg_error_pct': 0.0,
                'violation_rate': 0.0
            }
        
        errors = [v['error_pct'] for v in self.validations]
        
        return {
            'total_validations': len(self.validations),
            'violations_count': len(self.violations),
            'max_error_pct': max(errors),
            'avg_error_pct': sum(errors) / len(errors),
            'violation_rate': 100 * len(self.violations) / len(self.validations)
        }
    
    def print_summary(self):
        """Print human-readable summary to console."""
        summary = self.get_summary()
        
        print("=" * 80)
        print("LRP CONSERVATION PROPERTY VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total layers validated: {summary['total_validations']}")
        print(f"Violations found:       {summary['violations_count']}")
        print(f"Violation rate:         {summary['violation_rate']:.1f}%")
        print(f"Average error:          {summary['avg_error_pct']:.4f}%")
        print(f"Maximum error:          {summary['max_error_pct']:.2f}%")
        print(f"Tolerance threshold:    {self.tolerance*100}%")
        
        if self.violations:
            print("\n" + "=" * 80)
            print("VIOLATIONS DETAIL")
            print("=" * 80)
            for v in self.violations:
                print(f"\n{v['layer_name']} ({v['layer_type']}):")
                print(f"  Error:      {v['error_pct']:.2f}%")
                print(f"  Input sum:  {v['sum_input']:12.6f}")
                print(f"  Output sum: {v['sum_output']:12.6f}")
        else:
            print("\n✓ All layers passed conservation check!")
        
        print("=" * 80)


def stabilize_denominator(
    z: torch.Tensor,
    epsilon: float = 1e-2,
    rule: str = "epsilon"
) -> torch.Tensor:
    """Apply stabilization to denominator to prevent division by zero.
    
    Different rules use different stabilization strategies:
    - epsilon: z + ε·sign(z)
    - alphabeta: Separate handling of positive/negative contributions
    - z+: Only positive values, with epsilon
    
    Args:
        z: Tensor to stabilize (typically forward contributions)
        epsilon: Stabilization parameter
        rule: Which LRP rule is being applied
        
    Returns:
        Stabilized tensor safe for division
    """
    if rule == "epsilon":
        # Add epsilon with same sign as z
        return z + epsilon * torch.sign(z)
    elif rule == "z+":
        # Only positive contributions, clamp to epsilon minimum
        return torch.clamp(z, min=epsilon)
    else:
        # Default: simple epsilon addition
        return z + epsilon


def check_tensor_validity(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check tensor for NaN, Inf, or other numerical issues.
    
    Args:
        tensor: Tensor to check
        name: Name for logging
        
    Returns:
        True if tensor is valid, False otherwise
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan:
        logger.error(f"{name} contains NaN values!")
        return False
    
    if has_inf:
        logger.error(f"{name} contains Inf values!")
        return False
    
    return True


# ============================================================================
# Layer-Specific LRP Handlers
# ============================================================================


class LinearLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Linear (fully connected) layers.
    
    Implements both epsilon and alpha-beta rules for Linear layers.
    
    Epsilon rule:
        R_i = Σ_j (z_ij / (z_j + ε·sign(z_j))) · R_j
        where z_ij = x_i · w_ij and z_j = Σ_k z_kj + b_j
    
    Alpha-beta rule:
        R_i = Σ_j [(α·z_ij^+ / z_j^+) - (β·z_ij^- / z_j^-)] · R_j
        where z^+ and z^- are positive and negative contributions
    """
    
    def __init__(self):
        super().__init__(name="LinearHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.Linear."""
        return isinstance(layer, nn.Linear)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store input and output activations."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
        
        logger.debug(
            f"Linear layer: input shape {input_tensor.shape}, "
            f"output shape {output.shape}"
        )
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        epsilon: float = 1e-2,
        alpha: float = 2.0,
        beta: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance backward through Linear layer."""
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError(
                f"forward_hook not called for this layer. "
                f"Make sure to run forward pass before backward_relevance."
            )
        
        cache = self.activations_cache[module_id]
        x = cache['input']
        
        check_tensor_validity(x, "Linear input")
        check_tensor_validity(relevance_output, "Linear relevance_output")
        
        if rule == "epsilon":
            return self._epsilon_rule(layer, x, relevance_output, epsilon)
        elif rule == "alphabeta":
            return self._alphabeta_rule(layer, x, relevance_output, alpha, beta, epsilon)
        else:
            raise ValueError(f"Unsupported rule for LinearLRPHandler: {rule}")
    
    def _epsilon_rule(
        self,
        layer: nn.Linear,
        x: torch.Tensor,
        relevance_output: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Apply epsilon rule for Linear layer."""
        w = layer.weight
        b = layer.bias if layer.bias is not None else 0.0
        
        z = F.linear(x, w, b)
        z_stabilized = stabilize_denominator(z, epsilon, rule="epsilon")
        
        relevance_fractions = relevance_output / z_stabilized
        relevance_input = x * torch.mm(relevance_fractions, w)
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=0.01, layer_name=f"Linear(ε={epsilon})"
        )
        
        return relevance_input
    
    def _alphabeta_rule(
        self,
        layer: nn.Linear,
        x: torch.Tensor,
        relevance_output: torch.Tensor,
        alpha: float,
        beta: float,
        epsilon: float
    ) -> torch.Tensor:
        """Apply alpha-beta rule for Linear layer."""
        w = layer.weight
        b = layer.bias if layer.bias is not None else 0.0
        
        w_pos = torch.clamp(w, min=0)
        w_neg = torch.clamp(w, max=0)
        
        z_pos = F.linear(x, w_pos, torch.clamp(b, min=0))
        z_neg = F.linear(x, w_neg, torch.clamp(b, max=0))
        
        z_pos_stabilized = z_pos + epsilon
        z_neg_stabilized = z_neg - epsilon
        
        r_pos_frac = relevance_output / z_pos_stabilized
        r_neg_frac = relevance_output / z_neg_stabilized
        
        relevance_pos = alpha * x * torch.mm(r_pos_frac, w_pos)
        relevance_neg = beta * x * torch.mm(r_neg_frac, w_neg)
        
        relevance_input = relevance_pos + relevance_neg
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=0.05,
            layer_name=f"Linear(α={alpha},β={beta})"
        )
        
        return relevance_input


class ReLULRPHandler(LRPLayerHandler):
    """LRP handler for nn.ReLU activation layers.
    
    For ReLU, relevance is passed through unchanged, since the
    positive activation constraint is already captured in the
    forward activations.
    """
    
    def __init__(self):
        super().__init__(name="ReLUHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.ReLU."""
        return isinstance(layer, nn.ReLU)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store activations (mainly for validation)."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Pass relevance through ReLU unchanged."""
        self.validate_conservation(
            relevance_output, relevance_output,
            tolerance=1e-6, layer_name="ReLU"
        )
        
        return relevance_output


class EmbeddingLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Embedding layers.
    
    Embedding is a lookup operation - relevance flows directly back
    to the embedding vectors that were selected.
    """
    
    def __init__(self):
        super().__init__(name="EmbeddingHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.Embedding."""
        return isinstance(layer, nn.Embedding)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store indices and embeddings."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'indices': input_tensor.detach(),
            'output': output.detach()
        }
        
        logger.debug(
            f"Embedding layer: indices shape {input_tensor.shape}, "
            f"output shape {output.shape}"
        )
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance through embedding layer."""
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError("forward_hook not called for Embedding layer")
        
        # Sum over embedding dimension to get per-token relevance
        if relevance_output.dim() == 3:
            relevance_input = relevance_output.sum(dim=-1)
        else:
            relevance_input = relevance_output
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=1e-6, layer_name="Embedding"
        )
        
        return relevance_input


# ============================================================================
# CNN Layer Handlers
# ============================================================================


class Conv2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Conv2d (convolutional) layers.
    
    Implements epsilon and alpha-beta rules for 2D convolutions.
    Similar to Linear layers but with spatial dimensions.
    
    The key insight: convolution is a linear operation, so we can
    apply the same LRP rules as Linear layers, but need to handle
    the spatial structure properly.
    """
    
    def __init__(self):
        super().__init__(name="Conv2dHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.Conv2d."""
        return isinstance(layer, nn.Conv2d)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store input and output activations."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
        
        logger.debug(
            f"Conv2d layer: input shape {input_tensor.shape}, "
            f"output shape {output.shape}"
        )
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        epsilon: float = 1e-2,
        alpha: float = 2.0,
        beta: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance backward through Conv2d layer.
        
        Args:
            layer: Conv2d module
            relevance_output: Relevance at layer output [batch, out_ch, H, W]
            rule: 'epsilon' or 'alphabeta'
            epsilon: Stabilization parameter
            alpha: Positive contribution weight
            beta: Negative contribution weight
            
        Returns:
            relevance_input: Relevance at layer input [batch, in_ch, H, W]
        """
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError(
                f"forward_hook not called for Conv2d layer. "
                f"Make sure to run forward pass before backward_relevance."
            )
        
        cache = self.activations_cache[module_id]
        x = cache['input']
        
        check_tensor_validity(x, "Conv2d input")
        check_tensor_validity(relevance_output, "Conv2d relevance_output")
        
        if rule == "epsilon":
            return self._epsilon_rule(layer, x, relevance_output, epsilon)
        elif rule == "alphabeta":
            return self._alphabeta_rule(layer, x, relevance_output, alpha, beta, epsilon)
        else:
            raise ValueError(f"Unsupported rule for Conv2dLRPHandler: {rule}")
    
    def _epsilon_rule(
        self,
        layer: nn.Conv2d,
        x: torch.Tensor,
        relevance_output: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        """Apply epsilon rule for Conv2d layer.
        
        Similar to Linear layer but for spatial convolutions.
        """
        # Forward pass
        z = F.conv2d(
            x, layer.weight, layer.bias,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups
        )
        
        # Stabilize denominator
        z_stabilized = stabilize_denominator(z, epsilon, rule="epsilon")
        
        # Relevance fractions
        s = relevance_output / z_stabilized
        
        # Backward pass using transposed convolution
        # This distributes relevance back to inputs
        # Calculate output_padding to match input size exactly
        output_padding = []
        for i in range(2):  # height and width
            out_size = relevance_output.shape[2 + i]
            in_size = x.shape[2 + i]
            # Calculate expected output size from conv_transpose2d formula
            expected_out = (out_size - 1) * layer.stride[i] - 2 * layer.padding[i] + layer.kernel_size[i]
            # Adjust output_padding to match actual input size
            output_padding.append(max(0, in_size - expected_out))
        
        c = F.conv_transpose2d(
            s,
            layer.weight,
            None,
            stride=layer.stride,
            padding=layer.padding,
            output_padding=tuple(output_padding),
            groups=layer.groups,
            dilation=layer.dilation
        )
        
        # Weight by input activations
        relevance_input = x * c
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=0.20, layer_name=f"Conv2d(ε={epsilon})"
        )
        
        return relevance_input
    
    def _alphabeta_rule(
        self,
        layer: nn.Conv2d,
        x: torch.Tensor,
        relevance_output: torch.Tensor,
        alpha: float,
        beta: float,
        epsilon: float
    ) -> torch.Tensor:
        """Apply alpha-beta rule for Conv2d layer."""
        # Separate positive and negative weights
        w_pos = torch.clamp(layer.weight, min=0)
        w_neg = torch.clamp(layer.weight, max=0)
        
        # Positive and negative forward contributions
        z_pos = F.conv2d(
            x, w_pos, 
            torch.clamp(layer.bias, min=0) if layer.bias is not None else None,
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups
        )
        z_neg = F.conv2d(
            x, w_neg,
            torch.clamp(layer.bias, max=0) if layer.bias is not None else None,
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups
        )
        
        # Stabilize
        z_pos_stabilized = z_pos + epsilon
        z_neg_stabilized = z_neg - epsilon
        
        # Relevance fractions
        r_pos_frac = relevance_output / z_pos_stabilized
        r_neg_frac = relevance_output / z_neg_stabilized
        
        # Calculate output_padding to match input size exactly
        output_padding = []
        for i in range(2):  # height and width
            out_size = relevance_output.shape[2 + i]
            in_size = x.shape[2 + i]
            expected_out = (out_size - 1) * layer.stride[i] - 2 * layer.padding[i] + layer.kernel_size[i]
            output_padding.append(max(0, in_size - expected_out))
        
        # Backward passes
        relevance_pos = alpha * F.conv_transpose2d(
            r_pos_frac * z_pos, w_pos, None,
            stride=layer.stride, padding=layer.padding,
            output_padding=tuple(output_padding),
            groups=layer.groups, dilation=layer.dilation
        ) * x / (x + epsilon)
        
        relevance_neg = beta * F.conv_transpose2d(
            r_neg_frac * z_neg, w_neg, None,
            stride=layer.stride, padding=layer.padding,
            output_padding=tuple(output_padding),
            groups=layer.groups, dilation=layer.dilation
        ) * x / (x - epsilon)
        
        relevance_input = relevance_pos + relevance_neg
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=0.05,
            layer_name=f"Conv2d(α={alpha},β={beta})"
        )
        
        return relevance_input


class MaxPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.MaxPool2d pooling layers.
    
    Uses winner-take-all: relevance goes only to the maximum element
    in each pooling window.
    """
    
    def __init__(self):
        super().__init__(name="MaxPool2dHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.MaxPool2d."""
        return isinstance(layer, nn.MaxPool2d)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store input, output, and indices of max elements."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        
        # Get indices of maximum values
        _, indices = F.max_pool2d(
            input_tensor,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=True
        )
        
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach(),
            'indices': indices
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance through MaxPool2d using winner-take-all."""
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError("forward_hook not called for MaxPool2d layer")
        
        cache = self.activations_cache[module_id]
        input_tensor = cache['input']
        input_shape = input_tensor.shape
        indices = cache['indices']
        
        # Unpool: distribute relevance to winning positions
        try:
            relevance_input = F.max_unpool2d(
                relevance_output,
                indices,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                output_size=input_shape
            )
        except RuntimeError:
            # If max_unpool2d fails, fall back to uniform distribution
            relevance_input = F.interpolate(
                relevance_output,
                size=(input_shape[2], input_shape[3]),
                mode='nearest'
            )
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=1e-6, layer_name="MaxPool2d"
        )
        
        return relevance_input


class AvgPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.AvgPool2d pooling layers.
    
    Distributes relevance uniformly across the pooling window.
    """
    
    def __init__(self):
        super().__init__(name="AvgPool2dHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.AvgPool2d."""
        return isinstance(layer, nn.AvgPool2d)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store activations."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance through AvgPool2d uniformly."""
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError("forward_hook not called for AvgPool2d layer")
        
        cache = self.activations_cache[module_id]
        input_shape = cache['input'].shape
        
        # Each output pixel is the average of kernel_size x kernel_size inputs
        # So each output relevance is distributed equally to those inputs
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        stride = layer.stride if layer.stride is not None else kernel_size
        
        # Use transposed average pooling (just upsample and scale)
        relevance_input = F.interpolate(
            relevance_output,
            size=input_shape[2:],
            mode='nearest'
        )
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=0.01, layer_name="AvgPool2d"
        )
        
        return relevance_input


class FlattenLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Flatten layers.
    
    Flatten is just a reshape operation, so relevance flows through unchanged.
    """
    
    def __init__(self):
        super().__init__(name="FlattenHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.Flatten."""
        return isinstance(layer, nn.Flatten)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store input shape for reshape."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input_shape': input_tensor.shape,
            'output': output.detach()
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Reshape relevance back to original shape."""
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError("forward_hook not called for Flatten layer")
        
        cache = self.activations_cache[module_id]
        input_shape = cache['input_shape']
        
        # Simply reshape back
        relevance_input = relevance_output.view(input_shape)
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=1e-6, layer_name="Flatten"
        )
        
        return relevance_input


class BatchNorm2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.BatchNorm2d normalization layers.
    
    BatchNorm is treated as identity for LRP since it doesn't change
    which features are relevant, only their scale.
    """
    
    def __init__(self):
        super().__init__(name="BatchNorm2dHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.BatchNorm2d."""
        return isinstance(layer, nn.BatchNorm2d)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store activations."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Pass relevance through BatchNorm unchanged.
        
        BatchNorm applies: y = γ(x - μ)/σ + β
        For LRP, we treat it as identity since it doesn't change
        which spatial locations/channels are relevant.
        """
        self.validate_conservation(
            relevance_output, relevance_output,
            tolerance=1e-6, layer_name="BatchNorm2d"
        )
        
        return relevance_output


class AdaptiveAvgPool2dLRPHandler(LRPLayerHandler):
    """LRP handler for nn.AdaptiveAvgPool2d pooling layers.
    
    Distributes relevance uniformly, similar to AvgPool2d.
    """
    
    def __init__(self):
        super().__init__(name="AdaptiveAvgPool2dHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.AdaptiveAvgPool2d."""
        return isinstance(layer, nn.AdaptiveAvgPool2d)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store activations."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Propagate relevance through AdaptiveAvgPool2d uniformly."""
        module_id = id(layer)
        if module_id not in self.activations_cache:
            raise RuntimeError("forward_hook not called for AdaptiveAvgPool2d layer")
        
        cache = self.activations_cache[module_id]
        input_tensor = cache['input']
        input_shape = input_tensor.shape
        output_shape = cache['output'].shape
        
        # Handle case where relevance is 2D (flattened) instead of 4D
        # This happens when a Flatten layer follows this pooling layer
        if relevance_output.dim() == 2 and len(output_shape) == 4:
            # Reshape to match the cached output shape
            # E.g., [1, 25088] -> [1, 512, 7, 7] where 25088 = 512 * 7 * 7
            relevance_output = relevance_output.view(output_shape)
        
        # For AdaptiveAvgPool2d, distribute relevance uniformly
        # Direct approach: create a tensor with exact input dimensions
        batch_size, channels, out_h, out_w = relevance_output.shape
        in_h, in_w = input_shape[2], input_shape[3]
        
        # Create output tensor with exact dimensions from cached input
        relevance_input = torch.zeros(
            batch_size, channels, in_h, in_w,
            device=relevance_output.device,
            dtype=relevance_output.dtype
        )
        
        # Distribute each output pixel's relevance uniformly to the corresponding input region
        # For adaptive pooling with output size (1, 1), distribute to entire input
        if out_h == 1 and out_w == 1:
            # Special case: output is 1x1, distribute equally to all input pixels
            relevance_input[:, :, :, :] = relevance_output / (in_h * in_w)
        else:
            # General case: map each output pixel to its input region
            stride_h = in_h / out_h
            stride_w = in_w / out_w
            
            for i in range(out_h):
                for j in range(out_w):
                    h_start = int(i * stride_h)
                    h_end = int((i + 1) * stride_h)
                    w_start = int(j * stride_w)
                    w_end = int((j + 1) * stride_w)
                    
                    # Distribute relevance equally to the region
                    region_size = (h_end - h_start) * (w_end - w_start)
                    relevance_input[:, :, h_start:h_end, w_start:w_end] = \
                        relevance_output[:, :, i:i+1, j:j+1] / region_size
        
        self.validate_conservation(
            relevance_input, relevance_output,
            tolerance=0.5, layer_name="AdaptiveAvgPool2d"
        )
        
        return relevance_input


class DropoutLRPHandler(LRPLayerHandler):
    """LRP handler for nn.Dropout layers.
    
    During evaluation (when we do LRP), dropout is inactive,
    so relevance passes through unchanged.
    """
    
    def __init__(self):
        super().__init__(name="DropoutHandler")
    
    def supports(self, layer: nn.Module) -> bool:
        """Check if layer is nn.Dropout."""
        return isinstance(layer, nn.Dropout)
    
    def forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:
        """Store activations (dropout is inactive in eval mode)."""
        input_tensor = input[0] if isinstance(input, tuple) else input
        module_id = id(module)
        self.activations_cache[module_id] = {
            'input': input_tensor.detach(),
            'output': output.detach()
        }
    
    def backward_relevance(
        self,
        layer: nn.Module,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Pass relevance through Dropout unchanged (eval mode)."""
        self.validate_conservation(
            relevance_output, relevance_output,
            tolerance=1e-6, layer_name="Dropout"
        )
        
        return relevance_output


class AdditionLRPHandler(LRPLayerHandler):
    """LRP handler for addition operations (skip connections).
    
    Handles y = a + b by splitting relevance between the two branches
    proportionally to their contributions.
    """
    
    def __init__(self):
        super().__init__(name="AdditionHandler")
        # Store branch outputs for each addition operation
        self.branch_cache = {}
    
    def supports(self, layer: nn.Module) -> bool:
        """This handler is manually invoked, not via isinstance checks."""
        return False
    
    def forward_hook(self, module: nn.Module, input_tensor: torch.Tensor, output: torch.Tensor):
        """Not used for addition operations."""
        pass
    
    def backward_relevance(
        self,
        module: nn.Module,
        input_relevance: torch.Tensor,
        output_relevance: torch.Tensor,
        rule: str = "epsilon",
        **kwargs
    ) -> torch.Tensor:
        """Not used for addition operations. Use backward_relevance_split instead."""
        return input_relevance
    
    def cache_branches(self, operation_id: int, branch_a: torch.Tensor, branch_b: torch.Tensor):
        """Store the outputs of both branches before addition."""
        self.branch_cache[operation_id] = {
            'branch_a': branch_a.detach(),
            'branch_b': branch_b.detach()
        }
    
    def backward_relevance_split(
        self,
        operation_id: int,
        relevance_output: torch.Tensor,
        rule: str = "epsilon",
        epsilon: float = 1e-9,
        **kwargs
    ) -> tuple:
        """Split relevance between two branches of an addition.
        
        Args:
            operation_id: Unique identifier for this addition operation
            relevance_output: Relevance flowing back through the addition
            rule: LRP rule to use
            epsilon: Stabilization parameter
            
        Returns:
            (relevance_a, relevance_b): Relevance for each branch
        """
        if operation_id not in self.branch_cache:
            raise RuntimeError(f"No cached branches for addition operation {operation_id}")
        
        cache = self.branch_cache[operation_id]
        a = cache['branch_a']
        b = cache['branch_b']
        
        # Split relevance proportionally to contributions
        # R_a = (a / (a + b + eps)) * R_out
        # R_b = (b / (a + b + eps)) * R_out
        
        z = a + b
        z_stabilized = stabilize_denominator(z, epsilon, rule="epsilon")
        
        relevance_a = (a / z_stabilized) * relevance_output
        relevance_b = (b / z_stabilized) * relevance_output
        
        # Validate conservation: R_a + R_b ≈ R_out
        total_relevance = relevance_a + relevance_b
        conservation_error = torch.abs(total_relevance - relevance_output).max().item()
        max_relevance = torch.abs(relevance_output).max().item()
        
        if max_relevance > 1e-8:
            relative_error = conservation_error / max_relevance
            if relative_error > 0.1:  # 10% tolerance
                print(f"Warning: Addition relevance conservation error: {relative_error:.2%}")
        
        return relevance_a, relevance_b


def create_default_registry():
    """Create a registry with default handlers for common layers.
    
    Returns:
        LRPHandlerRegistry with handlers for common layers
    """
    registry = LRPHandlerRegistry()
    
    # Embedding-based model layers
    registry.register(LinearLRPHandler())
    registry.register(ReLULRPHandler())
    registry.register(EmbeddingLRPHandler())
    
    # CNN layers
    registry.register(Conv2dLRPHandler())
    registry.register(MaxPool2dLRPHandler())
    registry.register(AvgPool2dLRPHandler())
    registry.register(AdaptiveAvgPool2dLRPHandler())
    registry.register(FlattenLRPHandler())
    
    # Normalization and regularization
    registry.register(BatchNorm2dLRPHandler())
    registry.register(DropoutLRPHandler())
    
    logger.info("Created default handler registry with 11 handlers")
    return registry
