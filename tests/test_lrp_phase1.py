"""
Unit tests for LRP Phase 1: Core Infrastructure

Tests cover:
- LRPLayerHandler abstract base class
- LRPHandlerRegistry
- ConservationValidator
- LinearLRPHandler
- Basic UnifiedLRP functionality
"""

import unittest
import torch
import torch.nn as nn
import logging

# Import the Phase 1 components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from pyhealth.interpret.methods.lrp_base import (
    LRPLayerHandler,
    LRPHandlerRegistry,
    ConservationValidator,
    stabilize_denominator,
    check_tensor_validity,
    LinearLRPHandler,
    ReLULRPHandler,
    EmbeddingLRPHandler,
    create_default_registry
)
from pyhealth.interpret.methods.lrp import UnifiedLRP


# Set up logging for tests
logging.basicConfig(level=logging.INFO)


class TestLRPLayerHandler(unittest.TestCase):
    """Test the abstract base class and handler interface."""
    
    def test_handler_is_abstract(self):
        """Verify LRPLayerHandler cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            handler = LRPLayerHandler(name="test")
    
    def test_conservation_validation(self):
        """Test conservation property validation."""
        # Create a concrete handler for testing
        handler = LinearLRPHandler()
        
        # Perfect conservation
        r_in = torch.tensor([1.0, 2.0, 3.0])
        r_out = torch.tensor([6.0])
        is_valid, error = handler.validate_conservation(
            r_in, r_out, tolerance=0.01
        )
        self.assertTrue(is_valid)
        self.assertLess(error, 1.0)
        
        # Violated conservation
        r_in = torch.tensor([1.0, 2.0, 3.0])
        r_out = torch.tensor([10.0])
        is_valid, error = handler.validate_conservation(
            r_in, r_out, tolerance=0.01
        )
        self.assertFalse(is_valid)
        self.assertGreater(error, 1.0)


class TestLRPHandlerRegistry(unittest.TestCase):
    """Test the handler registry system."""
    
    def test_registry_creation(self):
        """Test creating an empty registry."""
        registry = LRPHandlerRegistry()
        self.assertEqual(len(registry.list_handlers()), 0)
    
    def test_handler_registration(self):
        """Test registering handlers."""
        registry = LRPHandlerRegistry()
        handler = LinearLRPHandler()
        
        registry.register(handler)
        self.assertEqual(len(registry.list_handlers()), 1)
        self.assertIn("LinearHandler", registry.list_handlers())
    
    def test_handler_lookup(self):
        """Test finding handlers for layers."""
        registry = LRPHandlerRegistry()
        registry.register(LinearLRPHandler())
        registry.register(ReLULRPHandler())
        
        # Linear layer should match LinearHandler
        linear = nn.Linear(10, 5)
        handler = registry.get_handler(linear)
        self.assertIsNotNone(handler)
        self.assertEqual(handler.name, "LinearHandler")
        
        # ReLU should match ReLUHandler
        relu = nn.ReLU()
        handler = registry.get_handler(relu)
        self.assertIsNotNone(handler)
        self.assertEqual(handler.name, "ReLUHandler")
        
        # Conv2d should not match anything (not registered)
        conv = nn.Conv2d(3, 16, kernel_size=3)
        handler = registry.get_handler(conv)
        self.assertIsNone(handler)
    
    def test_invalid_handler_registration(self):
        """Test that only valid handlers can be registered."""
        registry = LRPHandlerRegistry()
        
        with self.assertRaises(TypeError):
            registry.register("not a handler")


class TestConservationValidator(unittest.TestCase):
    """Test the conservation property validator."""
    
    def test_validator_creation(self):
        """Test creating a validator."""
        validator = ConservationValidator(tolerance=0.01)
        self.assertEqual(validator.tolerance, 0.01)
        self.assertEqual(len(validator.violations), 0)
    
    def test_perfect_conservation(self):
        """Test validation with perfect conservation."""
        validator = ConservationValidator(tolerance=0.01)
        
        r_in = torch.tensor([1.0, 2.0, 3.0])
        r_out = torch.tensor([6.0])
        
        is_valid = validator.validate(
            layer_name="test_layer",
            relevance_input=r_in,
            relevance_output=r_out
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(len(validator.violations), 0)
    
    def test_conservation_violation(self):
        """Test validation with conservation violation."""
        validator = ConservationValidator(tolerance=0.01)
        
        r_in = torch.tensor([1.0, 2.0])  # Sum = 3.0
        r_out = torch.tensor([10.0])     # Sum = 10.0, error = 70%
        
        is_valid = validator.validate(
            layer_name="bad_layer",
            relevance_input=r_in,
            relevance_output=r_out
        )
        
        self.assertFalse(is_valid)
        self.assertEqual(len(validator.violations), 1)
        self.assertEqual(validator.violations[0]['layer_name'], 'bad_layer')
    
    def test_validator_summary(self):
        """Test summary statistics."""
        validator = ConservationValidator(tolerance=0.01)
        
        # Add some validations
        validator.validate("layer1", torch.tensor([1.0]), torch.tensor([1.0]))
        validator.validate("layer2", torch.tensor([1.0]), torch.tensor([1.5]))  # 33.33% error
        validator.validate("layer3", torch.tensor([2.0]), torch.tensor([2.0]))
        
        summary = validator.get_summary()
        
        self.assertEqual(summary['total_validations'], 3)
        self.assertEqual(summary['violations_count'], 1)
        self.assertGreater(summary['max_error_pct'], 30.0)


class TestLinearLRPHandler(unittest.TestCase):
    """Test the Linear layer handler."""
    
    def setUp(self):
        """Create test fixtures."""
        self.handler = LinearLRPHandler()
        self.layer = nn.Linear(10, 5, bias=True)
        
        # Initialize with known weights for testing
        with torch.no_grad():
            self.layer.weight.fill_(0.1)
            self.layer.bias.fill_(0.0)
    
    def test_handler_supports_linear(self):
        """Test that handler recognizes Linear layers."""
        self.assertTrue(self.handler.supports(self.layer))
        self.assertFalse(self.handler.supports(nn.ReLU()))
    
    def test_forward_hook_caching(self):
        """Test that forward hook stores activations."""
        x = torch.randn(2, 10)
        
        # Manually call forward hook
        output = self.layer(x)
        self.handler.forward_hook(self.layer, (x,), output)
        
        # Check that activations are cached
        module_id = id(self.layer)
        self.assertIn(module_id, self.handler.activations_cache)
        
        cache = self.handler.activations_cache[module_id]
        self.assertIn('input', cache)
        self.assertIn('output', cache)
    
    def test_epsilon_rule_conservation(self):
        """Test that epsilon rule preserves relevance."""
        batch_size = 2
        x = torch.randn(batch_size, 10)
        
        # Forward pass
        output = self.layer(x)
        self.handler.forward_hook(self.layer, (x,), output)
        
        # Backward relevance
        relevance_output = torch.ones(batch_size, 5)
        relevance_input = self.handler.backward_relevance(
            self.layer,
            relevance_output,
            rule="epsilon",
            epsilon=0.01
        )
        
        # Check conservation (epsilon rule typically ~5% error)
        sum_in = relevance_input.sum().item()
        sum_out = relevance_output.sum().item()
        error_pct = abs(sum_in - sum_out) / abs(sum_out) * 100
        
        self.assertLess(error_pct, 10.0, f"Conservation error: {error_pct:.2f}%")
    
    def test_alphabeta_rule(self):
        """Test alpha-beta rule implementation."""
        batch_size = 2
        x = torch.randn(batch_size, 10)
        
        # Forward pass
        output = self.layer(x)
        self.handler.forward_hook(self.layer, (x,), output)
        
        # Backward relevance
        relevance_output = torch.ones(batch_size, 5)
        relevance_input = self.handler.backward_relevance(
            self.layer,
            relevance_output,
            rule="alphabeta",
            alpha=2.0,
            beta=1.0,
            epsilon=0.01
        )
        
        # Alpha-beta may not conserve exactly (can be 100%+ error by design)
        sum_in = relevance_input.sum().item()
        sum_out = relevance_output.sum().item()
        error_pct = abs(sum_in - sum_out) / abs(sum_out) * 100
        
        self.assertLess(error_pct, 150.0, f"Alpha-beta error: {error_pct:.2f}%")


class TestDefaultRegistry(unittest.TestCase):
    """Test the default handler registry."""
    
    def test_default_registry_creation(self):
        """Test creating default registry with standard handlers."""
        registry = create_default_registry()
        
        # Should have at least Linear, ReLU, Embedding handlers
        handlers = registry.list_handlers()
        self.assertIn("LinearHandler", handlers)
        self.assertIn("ReLUHandler", handlers)
        self.assertIn("EmbeddingHandler", handlers)


class TestUnifiedLRP(unittest.TestCase):
    """Test the unified LRP implementation."""
    
    def setUp(self):
        """Create test model."""
        # Simple 2-layer MLP
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        self.model.eval()
    
    def test_lrp_initialization(self):
        """Test UnifiedLRP initialization."""
        lrp = UnifiedLRP(self.model, rule="epsilon", epsilon=0.01)
        
        self.assertEqual(lrp.rule, "epsilon")
        self.assertEqual(lrp.epsilon, 0.01)
        self.assertIsNotNone(lrp.registry)
    
    def test_lrp_attribute(self):
        """Test computing attributions."""
        lrp = UnifiedLRP(
            self.model,
            rule="epsilon",
            epsilon=0.01,
            validate_conservation=True
        )
        
        # Create dummy input
        x = torch.randn(2, 10)
        inputs = {'input': x}
        
        # Compute attributions
        attributions = lrp.attribute(inputs, target_class=0)
        
        # Check output format
        self.assertIn('input', attributions)
        self.assertEqual(attributions['input'].shape, x.shape)
    
    def test_conservation_validation(self):
        """Test that conservation is validated during attribution."""
        lrp = UnifiedLRP(
            self.model,
            rule="epsilon",
            epsilon=0.01,
            validate_conservation=True,
            conservation_tolerance=0.01
        )
        
        x = torch.randn(2, 10)
        attributions = lrp.attribute({'input': x}, target_class=0)
        
        # Check that validation ran
        summary = lrp.get_conservation_summary()
        self.assertGreater(summary['total_validations'], 0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_stabilize_denominator(self):
        """Test denominator stabilization."""
        z = torch.tensor([1.0, -1.0, 0.0, 2.0])
        
        # Epsilon rule - adds epsilon with sign, so |0 + 0.1*0| = 0
        z_stable = stabilize_denominator(z, epsilon=0.1, rule="epsilon")
        # Check that non-zero values are preserved
        self.assertTrue(torch.abs(z_stable[0]) >= 0.9)  # 1.0 + 0.1*sign(1.0) = 1.1
        self.assertTrue(torch.abs(z_stable[1]) >= 0.9)  # -1.0 + 0.1*sign(-1.0) = -1.1
        
        # Z+ rule
        z_stable = stabilize_denominator(z, epsilon=0.1, rule="z+")
        self.assertTrue(torch.all(z_stable >= 0.1))
    
    def test_check_tensor_validity(self):
        """Test tensor validation."""
        # Valid tensor
        x = torch.randn(3, 3)
        self.assertTrue(check_tensor_validity(x, "test"))
        
        # Tensor with NaN
        x_nan = torch.tensor([1.0, float('nan'), 2.0])
        self.assertFalse(check_tensor_validity(x_nan, "nan_test"))
        
        # Tensor with Inf
        x_inf = torch.tensor([1.0, float('inf'), 2.0])
        self.assertFalse(check_tensor_validity(x_inf, "inf_test"))


if __name__ == '__main__':
    unittest.main()
