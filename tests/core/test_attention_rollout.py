# Author: Felipe Amaral Bonchristiano
# NetID: felipea5
# Description: Unit tests for the AttentionRollout interpretability method
#              (Abnar & Zuidema, 2020, https://arxiv.org/abs/2005.00928).

import unittest

import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import AttentionRollout
from pyhealth.models import Transformer


def _make_dataset(samples, input_schema):
    """Build a tiny sample dataset."""
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema={"label": "binary"},
        dataset_name="rollout_test",
    )


class TestAttentionRollout(unittest.TestCase):
    """Tests for :class:`AttentionRollout`."""

    def setUp(self):
        torch.manual_seed(42)

        self.samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "conditions": ["A05B", "A05C", "A06A"],
                "procedures": ["P01", "P02"],
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v0",
                "conditions": ["A05B"],
                "procedures": ["P01"],
                "label": 0,
            },
        ]
        self.input_schema = {
            "conditions": "sequence",
            "procedures": "sequence",
        }
        self.dataset = _make_dataset(self.samples, self.input_schema)
        self.model = Transformer(
            dataset=self.dataset,
            embedding_dim=8,
            heads=2,
            num_layers=2,
        )
        self.interpreter = AttentionRollout(self.model)
        self.loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        self.batch = next(iter(self.loader))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _capture_relevance(self, model):
        """Wrap ``get_relevance_tensor`` to capture pre/post-expansion tensors.

        Returns ``(captured_R, captured_pre)`` dicts that are populated as a
        side effect of the next ``attribute`` call:

        * ``captured_R`` — the composed rollout matrices ``[batch, seq, seq]``.
        * ``captured_pre`` — the per-token relevance *before* the
          input-shape expansion done by ``_map_to_input_shapes``.
        """
        original = model.get_relevance_tensor
        captured_R = {}
        captured_pre = {}

        def spy(R, **data):
            for key, value in R.items():
                captured_R[key] = value.detach().clone()
            out = original(R, **data)
            for key, value in out.items():
                captured_pre[key] = value.detach().clone()
            return out

        model.get_relevance_tensor = spy
        return captured_R, captured_pre

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_returns_dict_keyed_by_feature_keys(self):
        """attribute() returns a dict keyed by exactly the model feature keys."""

        attributions = self.interpreter.attribute(**self.batch)

        self.assertIsInstance(attributions, dict)
        self.assertEqual(
            set(attributions.keys()),
            set(self.model.feature_keys),
        )

    def test_output_shape_matches_input_seq_length(self):
        """Each attribution matches its input feature's shape (seq length)."""

        attributions = self.interpreter.attribute(**self.batch)

        for key in self.model.feature_keys:
            self.assertIsInstance(attributions[key], torch.Tensor)
            # Flat sequence inputs are [batch, seq_len] - the attribution
            # must line up token-for-token with the raw input.
            self.assertEqual(attributions[key].shape, self.batch[key].shape)
            self.assertEqual(attributions[key].shape[0], 2) # batch size

    def test_multi_feature_key_model(self):
        """A model with several feature streams yields one entry per stream."""

        attributions = self.interpreter.attribute(**self.batch)

        self.assertIn("conditions", attributions)
        self.assertIn("procedures", attributions)
        self.assertEqual(len(attributions), 2)

    def test_row_stochastic_invariant(self):
        """Pre-expansion relevance is a distribution over tokens (sums to 1)."""

        _, captured_pre = self._capture_relevance(self.model)
        self.interpreter.attribute(**self.batch)

        self.assertTrue(captured_pre) # something was captured
        for key, relevance in captured_pre.items():
            token_sums = relevance.sum(dim=-1)
            self.assertTrue(
                torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5),
                msg=f"feature '{key}' relevance does not sum to 1: {token_sums}",
            )
            # Rollout produces non-negative relevance.
            self.assertTrue(torch.all(relevance >= 0))

    def test_rollout_matrices_are_row_stochastic(self):
        """Every composed rollout matrix has rows summing to 1."""

        captured_R, _ = self._capture_relevance(self.model)
        self.interpreter.attribute(**self.batch)

        for key, rollout in captured_R.items():
            row_sums = rollout.sum(dim=-1)
            self.assertTrue(
                torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5),
                msg=f"feature '{key}' rollout not row-stochastic: {row_sums}",
            )

    def test_identity_attention_gives_identity_rollout(self):
        """If attention is the identity at every layer, rollout is identity."""

        original_layers = self.model.get_attention_layers

        def identity_layers():
            # Reuse real shapes captured during the forward pass, but
            # overwrite each attention map with an identity per head.
            real = original_layers()
            patched = {}
            for key, layers in real.items():
                new_layers = []
                for attn_map, grad in layers:
                    batch, heads, seq, _ = attn_map.shape
                    eye = (
                        torch.eye(seq, dtype=attn_map.dtype, device=attn_map.device)
                        .reshape(1, 1, seq, seq)
                        .expand(batch, heads, seq, seq)
                        .contiguous()
                    )
                    new_layers.append((eye, grad))
                patched[key] = new_layers
            return patched

        self.model.get_attention_layers = identity_layers
        captured_R, _ = self._capture_relevance(self.model)
        self.interpreter.attribute(**self.batch)

        self.assertTrue(captured_R)
        for key, rollout in captured_R.items():
            batch, seq, _ = rollout.shape
            expected = (
                torch.eye(seq, dtype=rollout.dtype)
                .unsqueeze(0)
                .expand(batch, seq, seq)
            )
            self.assertTrue(
                torch.allclose(rollout, expected, atol=1e-6),
                msg=f"feature '{key}' rollout is not identity",
            )

    def test_single_layer(self):
        """num_layers=1 still produces valid, row-stochastic attributions."""

        torch.manual_seed(42)
        model = Transformer(
            dataset=self.dataset, embedding_dim=8, heads=2, num_layers=1
        )
        interpreter = AttentionRollout(model)
        _, captured_pre = self._capture_relevance(model)

        attributions = interpreter.attribute(**self.batch)

        self.assertEqual(set(attributions.keys()), set(model.feature_keys))
        for relevance in captured_pre.values():
            token_sums = relevance.sum(dim=-1)
            self.assertTrue(
                torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)
            )

    def test_single_head(self):
        """heads=1 (no head fusion needed) produces valid attributions."""

        torch.manual_seed(42)
        model = Transformer(
            dataset=self.dataset, embedding_dim=8, heads=1, num_layers=2
        )
        interpreter = AttentionRollout(model)
        _, captured_pre = self._capture_relevance(model)

        attributions = interpreter.attribute(**self.batch)

        self.assertEqual(set(attributions.keys()), set(model.feature_keys))
        for relevance in captured_pre.values():
            token_sums = relevance.sum(dim=-1)
            self.assertTrue(
                torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5)
            )

    def test_masked_padded_sequence(self):
        """Padded batches (uneven sequence lengths) stay row-stochastic."""
        # Sanity check that padding actually occurred.
        self.assertEqual(self.batch["conditions"].shape[1], 3)

        _, captured_pre = self._capture_relevance(self.model)
        self.interpreter.attribute(**self.batch)

        for key, relevance in captured_pre.items():
            token_sums = relevance.sum(dim=-1)
            self.assertTrue(
                torch.allclose(token_sums, torch.ones_like(token_sums), atol=1e-5),
                msg=f"padded feature '{key}' relevance does not sum to 1",
            )

    def test_target_class_idx_is_a_noop(self):
        """Rollout is class-agnostic: target_class_idx must not change output."""

        baseline = self.interpreter.attribute(**self.batch)
        with_target = self.interpreter.attribute(target_class_idx=1, **self.batch)

        for key in baseline:
            self.assertTrue(
                torch.allclose(baseline[key], with_target[key], atol=1e-6),
                msg=f"target_class_idx changed attributions for '{key}'",
            )

    def test_incompatible_model_raises_type_error(self):
        """Model lacking the attention-readout methods raises TypeError."""

        class PlainModel(nn.Module):
            def forward(self, **data):
                return {"logit": torch.zeros(1, 1)}

        with self.assertRaises(TypeError):
            AttentionRollout(PlainModel())

    def test_unsupported_head_fusion_raises_value_error(self):
        """An unsupported head_fusion value raises ValueError (not AttributeError)."""
        with self.assertRaises(ValueError):
            AttentionRollout(self.model, head_fusion="max")

    def test_model_is_in_eval_mode(self):
        """Constructing the interpreter puts the model in eval mode, disabling dropout
        and making attributions deterministic for a given input.
        """

        self.assertFalse(self.model.training)

    def test_attribute_is_deterministic(self):
        """Repeated calls on the same batch produce identical attributions."""

        first = self.interpreter.attribute(**self.batch)
        second = self.interpreter.attribute(**self.batch)

        for key in first:
            self.assertTrue(
                torch.allclose(first[key], second[key], atol=1e-6),
                msg=f"attributions for '{key}' are not deterministic",
            )

    def test_callable_interface_matches_attribute(self):
        """Calling the interpreter directly is equivalent to attribute()."""
        
        via_attribute = self.interpreter.attribute(**self.batch)
        via_call = self.interpreter(**self.batch)

        self.assertEqual(set(via_attribute.keys()), set(via_call.keys()))
        for key in via_attribute:
            self.assertTrue(
                torch.allclose(via_attribute[key], via_call[key], atol=1e-6)
            )


if __name__ == "__main__":
    unittest.main()
