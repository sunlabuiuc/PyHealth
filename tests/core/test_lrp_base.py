"""Unit tests for the LRP utility functions and per-layer handlers in lrp_base."""

import logging
import unittest

import torch
from torch import nn

from pyhealth.interpret.methods.lrp_base import (
    AdaptiveAvgPool2dLRPHandler,
    AvgPool2dLRPHandler,
    BatchNorm2dLRPHandler,
    Conv2dLRPHandler,
    DropoutLRPHandler,
    EmbeddingLRPHandler,
    FlattenLRPHandler,
    LinearLRPHandler,
    LRPHandlerRegistry,
    MaxPool2dLRPHandler,
    ReLULRPHandler,
    RNNLRPHandler,
    check_tensor_validity,
    create_default_registry,
    crop_spatial,
    match_relevance_dim,
    pad_to_match,
    stabilize_denominator,
)


def _run_forward(handler, layer, x):
    """Run *layer* on *x* and populate *handler*'s activation cache."""
    out = layer(x)
    handler.forward_hook(layer, (x,), out)
    return out


class TestUtilityFunctions(unittest.TestCase):

    def test_stabilize_denominator_rules(self):
        z = torch.tensor([[2.0, -2.0, 0.0]])
        self.assertTrue(torch.allclose(
            stabilize_denominator(z, 0.1, rule="epsilon"),
            torch.tensor([[2.1, -2.1, 0.1]]),
        ))
        self.assertTrue(torch.allclose(
            stabilize_denominator(z, 0.1, rule="z+"),
            torch.tensor([[2.0, 0.1, 0.1]]),
        ))
        # Unknown rule falls back to adding epsilon
        self.assertTrue(torch.allclose(
            stabilize_denominator(z, 0.1, rule="other"),
            torch.tensor([[2.1, -1.9, 0.1]]),
        ))

    def test_check_tensor_validity(self):
        self.assertTrue(check_tensor_validity(torch.ones(2, 2)))
        with self.assertLogs("pyhealth.interpret.methods.lrp_base", level=logging.ERROR):
            self.assertFalse(check_tensor_validity(torch.tensor([float("nan")]), "r"))
        with self.assertLogs("pyhealth.interpret.methods.lrp_base", level=logging.ERROR):
            self.assertFalse(check_tensor_validity(torch.tensor([float("inf")]), "r"))

    def test_pad_to_match(self):
        x = torch.ones(2, 3)
        same, orig = pad_to_match(x, 3)
        self.assertIs(same, x)
        self.assertEqual(orig, 3)
        padded, orig = pad_to_match(x, 5)
        self.assertEqual(padded.shape, (2, 5))
        self.assertEqual(padded[:, 3:].abs().sum().item(), 0.0)
        truncated, orig = pad_to_match(torch.ones(2, 6), 4)
        self.assertEqual((truncated.shape, orig), ((2, 4), 6))

    def test_match_relevance_dim(self):
        r = torch.ones(1, 3)
        self.assertIs(match_relevance_dim(r, 3), r)
        self.assertEqual(match_relevance_dim(r, 5).shape, (1, 5))
        self.assertEqual(match_relevance_dim(r, 2).shape, (1, 2))

    def test_crop_spatial(self):
        t = torch.ones(1, 1, 4, 4)
        self.assertIs(crop_spatial(t, (1, 1, 4, 4)), t)
        self.assertEqual(crop_spatial(t, (1, 1, 2, 3)).shape, (1, 1, 2, 3))
        padded = crop_spatial(t, (1, 1, 6, 5))
        self.assertEqual(padded.shape, (1, 1, 6, 5))
        # Padding is zero-filled and the sum is preserved
        self.assertEqual(padded.sum().item(), 16.0)


class TestRegistry(unittest.TestCase):

    def test_register_rejects_non_handlers(self):
        with self.assertRaises(TypeError):
            LRPHandlerRegistry().register(object())

    def test_lookup_cache_and_clear(self):
        registry = LRPHandlerRegistry()
        handler = LinearLRPHandler()
        registry.register(handler)
        self.assertIs(registry.get_handler(nn.Linear(2, 2)), handler)
        # Second lookup is served from the per-type cache
        self.assertIs(registry.get_handler(nn.Linear(3, 3)), handler)
        self.assertIsNone(registry.get_handler(nn.Tanh()))
        registry.clear()
        self.assertIsNone(registry.get_handler(nn.Linear(2, 2)))

    def test_default_registry_covers_all_layer_types(self):
        registry = create_default_registry()
        layers = [
            nn.Linear(2, 2), nn.ReLU(), nn.Embedding(5, 2), nn.Conv2d(1, 1, 3),
            nn.MaxPool2d(2), nn.AvgPool2d(2), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.BatchNorm2d(1), nn.Dropout(), nn.LSTM(2, 2), nn.GRU(2, 2),
        ]
        for layer in layers:
            self.assertIsNotNone(registry.get_handler(layer), type(layer).__name__)

    def test_backward_without_forward_raises(self):
        with self.assertRaises(RuntimeError):
            LinearLRPHandler().backward_relevance(nn.Linear(2, 2), torch.ones(1, 2))

    def test_clear_cache(self):
        handler, layer = LinearLRPHandler(), nn.Linear(2, 2)
        _run_forward(handler, layer, torch.randn(1, 2))
        handler.clear_cache()
        self.assertEqual(handler.activations_cache, {})


class TestLinearHandler(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.handler = LinearLRPHandler()

    def test_epsilon_conserves_relevance_without_bias(self):
        layer = nn.Linear(6, 3, bias=False)
        x = torch.randn(2, 6)
        out = _run_forward(self.handler, layer, x)
        r_in = self.handler.backward_relevance(layer, out, rule="epsilon", epsilon=1e-6)
        self.assertTrue(torch.allclose(r_in.sum(dim=1), out.sum(dim=1), atol=1e-3))

    def test_flattens_higher_dim_input(self):
        layer = nn.Linear(12, 2)
        x = torch.randn(2, 3, 4)
        handler = self.handler
        handler.forward_hook(layer, (x,), layer(x.view(2, -1)))
        for rule in ("epsilon", "alphabeta"):
            r = handler.backward_relevance(layer, torch.ones(2, 2), rule=rule)
            self.assertEqual(r.shape, (2, 12))

    def test_input_width_mismatch_is_padded_and_trimmed(self):
        # The cached input is narrower than the weight matrix (e.g. a model
        # that concatenates extra columns); relevance comes back at input width.
        layer = nn.Linear(5, 2)
        self.handler.forward_hook(layer, (torch.randn(1, 3),), torch.randn(1, 2))
        for rule in ("epsilon", "alphabeta"):
            r = self.handler.backward_relevance(layer, torch.ones(1, 2), rule=rule)
            self.assertEqual(r.shape, (1, 3))

    def test_alphabeta_without_bias(self):
        layer = nn.Linear(4, 2, bias=False)
        out = _run_forward(self.handler, layer, torch.randn(1, 4))
        r = self.handler.backward_relevance(layer, out, rule="alphabeta", alpha=2.0, beta=1.0)
        self.assertTrue(torch.isfinite(r).all())

    def test_unsupported_rule_raises(self):
        layer = nn.Linear(4, 2)
        _run_forward(self.handler, layer, torch.randn(1, 4))
        with self.assertRaises(ValueError):
            self.handler.backward_relevance(layer, torch.ones(1, 2), rule="gamma")


class TestConv2dHandler(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.handler = Conv2dLRPHandler()

    def test_rules_preserve_input_shape_with_stride(self):
        layer = nn.Conv2d(2, 3, kernel_size=3, stride=2, padding=1)
        x = torch.randn(1, 2, 7, 7)
        out = _run_forward(self.handler, layer, x)
        for rule in ("epsilon", "alphabeta"):
            r = self.handler.backward_relevance(layer, out, rule=rule)
            self.assertEqual(r.shape, x.shape)
            self.assertTrue(torch.isfinite(r).all())

    def test_alphabeta_without_bias(self):
        layer = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
        x = torch.randn(1, 1, 5, 5)
        out = _run_forward(self.handler, layer, x)
        r = self.handler.backward_relevance(layer, out, rule="alphabeta", alpha=2.0, beta=1.0)
        self.assertEqual(r.shape, x.shape)

    def test_unsupported_rule_raises(self):
        layer = nn.Conv2d(1, 1, 3)
        _run_forward(self.handler, layer, torch.randn(1, 1, 5, 5))
        with self.assertRaises(ValueError):
            self.handler.backward_relevance(layer, torch.ones(1, 1, 3, 3), rule="gamma")


class TestPoolingHandlers(unittest.TestCase):

    def test_maxpool_routes_to_winners(self):
        handler, layer = MaxPool2dLRPHandler(), nn.MaxPool2d(2, 2)
        x = torch.randn(1, 2, 4, 4)
        out = _run_forward(handler, layer, x)
        r = handler.backward_relevance(layer, torch.ones_like(out))
        self.assertEqual(r.shape, x.shape)
        # Exactly one winner per 2x2 window
        self.assertEqual(r.sum().item(), out.numel())
        self.assertEqual(int((r != 0).sum()), out.numel())

    def test_maxpool_falls_back_to_interpolation(self):
        handler, layer = MaxPool2dLRPHandler(), nn.MaxPool2d(2, 2)
        x = torch.randn(1, 1, 4, 4)
        _run_forward(handler, layer, x)
        # Channel count inconsistent with the cached indices -> unpool raises
        # RuntimeError and relevance is resized to the input's spatial dims.
        r = handler.backward_relevance(layer, torch.ones(1, 2, 2, 2))
        self.assertEqual(r.shape[2:], x.shape[2:])

    def test_avgpool_distributes_uniformly(self):
        for layer in (nn.AvgPool2d(2, 2), nn.AvgPool2d((2, 2), (2, 2), (0, 0))):
            handler = AvgPool2dLRPHandler()
            x = torch.randn(1, 2, 4, 4)
            out = _run_forward(handler, layer, x)
            r = handler.backward_relevance(layer, torch.ones_like(out))
            self.assertEqual(r.shape, x.shape)
            self.assertAlmostEqual(r.sum().item(), out.numel(), places=5)

    def test_adaptive_avgpool_global(self):
        handler, layer = AdaptiveAvgPool2dLRPHandler(), nn.AdaptiveAvgPool2d(1)
        x = torch.randn(2, 3, 4, 4)
        _run_forward(handler, layer, x)
        # Flattened relevance (as after nn.Flatten) is reshaped back first
        r = handler.backward_relevance(layer, torch.ones(2, 3))
        self.assertEqual(r.shape, x.shape)
        self.assertAlmostEqual(r.sum().item(), 6.0, places=5)

    def test_adaptive_avgpool_regions(self):
        handler, layer = AdaptiveAvgPool2dLRPHandler(), nn.AdaptiveAvgPool2d((2, 2))
        x = torch.randn(1, 1, 4, 4)
        out = _run_forward(handler, layer, x)
        r = handler.backward_relevance(layer, torch.ones_like(out))
        self.assertEqual(r.shape, x.shape)
        self.assertTrue(torch.allclose(r, torch.full_like(x, 0.25)))


class TestPassThroughHandlers(unittest.TestCase):

    def test_identity_handlers(self):
        relevance = torch.randn(1, 3, 4, 4)
        for handler, layer in (
            (ReLULRPHandler(), nn.ReLU()),
            (BatchNorm2dLRPHandler(), nn.BatchNorm2d(3)),
            (DropoutLRPHandler(), nn.Dropout()),
        ):
            self.assertTrue(handler.supports(layer))
            self.assertIs(handler.backward_relevance(layer, relevance), relevance)

    def test_embedding_handler(self):
        handler, layer = EmbeddingLRPHandler(), nn.Embedding(10, 4)
        idx = torch.tensor([[1, 2, 3]])
        _run_forward(handler, layer, idx)
        self.assertTrue(torch.equal(handler.activations_cache[id(layer)]["indices"], idx))
        self.assertEqual(handler.backward_relevance(layer, torch.ones(1, 3, 4)).shape, (1, 3))
        r2 = torch.ones(1, 3)
        self.assertIs(handler.backward_relevance(layer, r2), r2)

    def test_flatten_handler(self):
        handler, layer = FlattenLRPHandler(), nn.Flatten()
        x = torch.randn(2, 3, 2, 2)
        out = _run_forward(handler, layer, x)
        self.assertEqual(handler.backward_relevance(layer, out).shape, x.shape)

    def test_rnn_handler(self):
        handler = RNNLRPHandler()
        for layer in (nn.LSTM(4, 3, batch_first=True), nn.GRU(4, 3, batch_first=True)):
            x = torch.randn(2, 5, 4)
            out, _ = layer(x)
            handler.forward_hook(layer, (x,), out)
            r = handler.backward_relevance(layer, torch.ones(2, 3))
            self.assertEqual(r.shape, (2, 5, 3))
            # Spreading over timesteps conserves the total
            self.assertAlmostEqual(r.sum().item(), 6.0, places=5)

    def test_rnn_handler_non_sequence_input(self):
        handler, layer = RNNLRPHandler(), nn.GRU(4, 3)
        x = torch.randn(5, 4)  # unbatched input
        out, _ = layer(x)
        handler.forward_hook(layer, (x,), out)
        r = torch.ones(5, 3)
        self.assertIs(handler.backward_relevance(layer, r), r)


if __name__ == "__main__":
    unittest.main()
