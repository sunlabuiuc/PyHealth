"""Regression tests for separate attention-map and gradient capture."""

import unittest
from unittest.mock import patch

import torch

from pyhealth.datasets import get_dataloader
from pyhealth.interpret.api import (
    AttentionInterpretable,
    CheferInterpretable,
    GradientInterpretable,
)
from pyhealth.interpret.methods import AttentionRollout, CheferRelevance
from pyhealth.models.stagenet_mha import StageNetAttentionLayer
from pyhealth.models.transformer import (
    MultiHeadedAttention,
    TransformerBlock,
    TransformerLayer,
)
from tests.core import test_attention_rollout, test_stagenet_mha


class AttentionOnly(torch.nn.Module, AttentionInterpretable):
    """Minimal custom model whose forward needs no autograd."""

    feature_keys = ["codes"]

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.layers = {}

    def set_attention_hooks(self, enabled, *, capture_gradients=False):
        self.enabled = enabled

    def forward(self, **data):
        self.layers = {"codes": [(torch.eye(2).reshape(1, 1, 2, 2), None)]}
        return {"logit": torch.zeros(1, 1)}

    def get_attention_layers(self):
        return self.layers

    def get_relevance_tensor(self, R, **data):
        return {key: value[:, 0] for key, value in R.items()}


class TestAttentionCapture(unittest.TestCase):
    def models_and_batches(self):
        for case_class in (
            test_attention_rollout.TestAttentionRollout,
            test_stagenet_mha.TestStageNetMHA,
        ):
            torch.manual_seed(42)
            case = case_class()
            case.setUp()
            case.model.eval()
            batch = next(iter(get_dataloader(case.dataset, batch_size=2)))
            if case_class is test_stagenet_mha.TestStageNetMHA:
                # Supply the supported explicit padding mask for raw token IDs.
                for key in case.model.feature_keys:
                    time, value = batch[key]
                    batch[key] = (time, value, value.ne(0))
            yield case.model, batch

    def assert_captured(self, model, gradients):
        for layers in model.get_attention_layers().values():
            self.assertTrue(layers)
            for attention, gradient in layers:
                self.assertIsNotNone(attention)
                self.assertFalse(attention.requires_grad)
                self.assertIsNone(attention.grad_fn)
                if gradients:
                    self.assertIsNotNone(gradient)
                    self.assertEqual(attention.shape, gradient.shape)
                else:
                    self.assertIsNone(gradient)

    def test_alias_and_interfaces(self):
        self.assertIs(CheferInterpretable, GradientInterpretable)
        for model, _ in self.models_and_batches():
            self.assertIsInstance(model, AttentionInterpretable)
            self.assertIsInstance(model, GradientInterpretable)

    def test_rollout_no_grad_and_existing_parameter_gradients(self):
        for model, batch in self.models_and_batches():
            with self.subTest(model=type(model).__name__):
                for parameter in model.parameters():
                    parameter.grad = torch.ones_like(parameter)
                before = [p.grad.clone() for p in model.parameters()]
                interpreter = AttentionRollout(model)
                baseline = interpreter.attribute(**batch)
                with torch.no_grad():
                    result = interpreter.attribute(**batch)
                for key in baseline:
                    torch.testing.assert_close(result[key], baseline[key])
                self.assert_captured(model, gradients=False)
                for previous, parameter in zip(before, model.parameters()):
                    self.assertTrue(torch.equal(previous, parameter.grad))

    def test_disable_preserves_results_until_next_forward(self):
        for model, batch in self.models_and_batches():
            with self.subTest(model=type(model).__name__):
                model.set_attention_hooks(True)
                output = model(**batch)
                model.set_attention_hooks(False, capture_gradients=True)
                self.assert_captured(model, gradients=False)
                output["logit"].sum().backward()
                self.assert_captured(model, gradients=True)
                with torch.no_grad():
                    model(**batch)
                for layers in model.get_attention_layers().values():
                    for attention, gradient in layers:
                        self.assertIsNone(attention)
                        self.assertIsNone(gradient)

    def test_forward_only_does_not_capture_gradients_after_backward(self):
        for model, batch in self.models_and_batches():
            model.set_attention_hooks(True, capture_gradients=False)
            model(**batch)["logit"].sum().backward()
            self.assert_captured(model, gradients=False)

    def test_chefer_rollout_chefer_sequence(self):
        for model, batch in self.models_and_batches():
            with self.subTest(model=type(model).__name__):
                chefer = CheferRelevance(model)
                first = chefer.attribute(**batch)
                self.assert_captured(model, gradients=True)
                AttentionRollout(model).attribute(**batch)
                self.assert_captured(model, gradients=False)
                second = chefer.attribute(**batch)
                self.assert_captured(model, gradients=True)
                for key in first:
                    torch.testing.assert_close(first[key], second[key])

    def test_chefer_explicitly_requests_gradients(self):
        for model, batch in self.models_and_batches():
            original = model.set_attention_hooks

            def false_default(enabled, *, capture_gradients=False):
                original(enabled, capture_gradients=capture_gradients)

            with patch.object(model, "set_attention_hooks", false_default):
                CheferRelevance(model).attribute(**batch)
                self.assert_captured(model, gradients=True)

    def test_cleanup_after_forward_exception(self):
        for model, batch in self.models_and_batches():
            for interpreter_class in (AttentionRollout, CheferRelevance):
                with patch.object(model, "forward", side_effect=RuntimeError("failed")):
                    with self.assertRaisesRegex(RuntimeError, "failed"):
                        interpreter_class(model).attribute(**batch)
                with torch.no_grad():
                    model(**batch)
                for layers in model.get_attention_layers().values():
                    self.assertTrue(all(pair == (None, None) for pair in layers))

    def test_attention_only_and_method_only_models(self):
        model = AttentionOnly()
        with torch.no_grad():
            result = AttentionRollout(model).attribute(codes=torch.ones(1, 2))
        torch.testing.assert_close(result["codes"], torch.tensor([[1.0, 0.0]]))
        with self.assertRaises(ValueError):
            CheferRelevance(model)

        class MethodOnly(torch.nn.Module):
            set_attention_hooks = AttentionOnly.set_attention_hooks
            get_attention_layers = AttentionOnly.get_attention_layers
            get_relevance_tensor = AttentionOnly.get_relevance_tensor

        with self.assertRaises(TypeError):
            AttentionRollout(MethodOnly())

    def test_obsolete_signature_requires_migration(self):
        class Obsolete(AttentionOnly, GradientInterpretable):
            def set_attention_hooks(self, enabled):
                self.enabled = enabled

        for interpreter_class in (AttentionRollout, CheferRelevance):
            with self.assertRaisesRegex(TypeError, "capture_gradients"):
                interpreter_class(Obsolete()).attribute(codes=torch.ones(1, 2))

    def test_chefer_missing_or_mismatched_capture(self):
        model, batch = next(self.models_and_batches())
        attention = torch.ones(2, 2, 3, 3)
        cases = (
            [],
            [(None, attention)],
            [(attention, None)],
            [(attention, attention[:, :, :1])],
        )
        for layers in cases:
            with self.subTest(layers=layers):
                with patch.object(
                    model, "get_attention_layers", return_value={"codes": layers}
                ):
                    with self.assertRaisesRegex(RuntimeError, "codes"):
                        CheferRelevance(model).attribute(**batch)

    def test_low_level_positional_gradient_argument(self):
        x = torch.randn(2, 3, 6, requires_grad=True)
        mha = MultiHeadedAttention(2, 6)
        block = TransformerBlock(6, 2, 0.0)
        layer = TransformerLayer(6, heads=2)
        stage = StageNetAttentionLayer(6, chunk_size=2, levels=3, num_heads=2)
        calls = (
            (mha, lambda: mha(x, x, x, None, True)),
            (block.attention, lambda: block(x, None, True)),
            (layer.transformer[0].attention, lambda: layer(x, None, True)[0]),
            (stage.mha, lambda: stage(x, None, None, True)[0]),
        )
        for attention, call in calls:
            call().sum().backward()
            self.assertIsNotNone(attention.get_attn_map())
            self.assertIsNotNone(attention.get_attn_grad())


if __name__ == "__main__":
    unittest.main()
