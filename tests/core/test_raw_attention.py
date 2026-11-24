"""Unit tests for the RawAttention interpretability method
(Chefer et al., 2021, https://arxiv.org/abs/2012.09838)."""

import unittest

import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import RawAttention
from pyhealth.models import Transformer
from pyhealth.models.stagenet_mha import StageAttentionNet


class TestRawAttention(unittest.TestCase):
    """Tests for :class:`RawAttention` on the Transformer."""

    def setUp(self):
        torch.manual_seed(42)

        samples = [
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
        self.dataset = create_sample_dataset(
            samples=samples,
            input_schema={"conditions": "sequence", "procedures": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="raw_attention_test",
        )
        self.model = Transformer(
            dataset=self.dataset, embedding_dim=8, heads=2, num_layers=2
        )
        # the interpreter does this too; expected values need it before then
        self.model.eval()
        self.loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        self.batch = next(iter(self.loader))

    def _capture_pre_expansion(self, model):
        """Wrap ``get_relevance_tensor`` to capture its input and output."""
        original = model.get_relevance_tensor
        captured_R, captured_pre = {}, {}

        def spy(R, **data):
            captured_R.update({k: v.detach().clone() for k, v in R.items()})
            out = original(R, **data)
            captured_pre.update({k: v.detach().clone() for k, v in out.items()})
            return out

        model.get_relevance_tensor = spy
        return captured_R, captured_pre

    def _cls_rows(self, layer, fuse):
        """Expected relevance: CLS row of one layer's fused attention."""
        self.model.set_attention_hooks(True)
        try:
            self.model(**self.batch)
        finally:
            self.model.set_attention_hooks(False)
        return {
            key: fuse(layers[layer][0].detach())[:, 0]
            for key, layers in self.model.get_attention_layers().items()
        }

    def test_output_keys_and_shapes(self):
        """One attribution per feature key, shaped like the raw input."""
        attributions = RawAttention(self.model).attribute(**self.batch)

        self.assertEqual(set(attributions), set(self.model.feature_keys))
        for key in self.model.feature_keys:
            self.assertEqual(attributions[key].shape, self.batch[key].shape)

    def test_default_is_last_layer_mean_cls_row(self):
        """Default reads the CLS row of the last layer, heads averaged."""
        expected = self._cls_rows(-1, lambda a: a.mean(dim=1))
        _, captured_pre = self._capture_pre_expansion(self.model)
        RawAttention(self.model).attribute(**self.batch)

        for key, relevance in captured_pre.items():
            self.assertTrue(torch.allclose(relevance, expected[key], atol=1e-6))

    def test_layer_selects_first_layer(self):
        """layer=0 reads the first layer, which differs from the last."""
        expected = self._cls_rows(0, lambda a: a.mean(dim=1))
        _, captured_pre = self._capture_pre_expansion(self.model)
        RawAttention(self.model, layer=0).attribute(**self.batch)

        for key, relevance in captured_pre.items():
            self.assertTrue(torch.allclose(relevance, expected[key], atol=1e-6))

    def test_head_fusion_max(self):
        """head_fusion='max' takes the elementwise max over heads."""
        expected = self._cls_rows(-1, lambda a: a.max(dim=1).values)
        _, captured_pre = self._capture_pre_expansion(self.model)
        RawAttention(self.model, head_fusion="max").attribute(**self.batch)

        for key, relevance in captured_pre.items():
            self.assertTrue(torch.allclose(relevance, expected[key], atol=1e-6))

    def test_mean_fusion_is_a_distribution(self):
        """With mean fusion each relevance row is non-negative and sums to 1."""
        _, captured_pre = self._capture_pre_expansion(self.model)
        RawAttention(self.model).attribute(**self.batch)

        self.assertTrue(captured_pre)
        for relevance in captured_pre.values():
            sums = relevance.sum(dim=-1)
            self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))
            self.assertTrue(torch.all(relevance >= 0))

    def test_identity_attention_gives_one_hot_cls(self):
        """Identity attention puts all relevance on the CLS token."""
        original_layers = self.model.get_attention_layers

        def identity_layers():
            patched = {}
            for key, layers in original_layers().items():
                new_layers = []
                for attn_map, grad in layers:
                    b, h, s, _ = attn_map.shape
                    eye = torch.eye(s).expand(b, h, s, s).contiguous()
                    new_layers.append((eye, grad))
                patched[key] = new_layers
            return patched

        self.model.get_attention_layers = identity_layers
        _, captured_pre = self._capture_pre_expansion(self.model)
        RawAttention(self.model).attribute(**self.batch)

        for relevance in captured_pre.values():
            expected = torch.zeros_like(relevance)
            expected[:, 0] = 1.0
            self.assertTrue(torch.equal(relevance, expected))

    def test_target_class_idx_is_a_noop(self):
        """Raw attention is class-agnostic."""
        interpreter = RawAttention(self.model)
        baseline = interpreter.attribute(**self.batch)
        with_target = interpreter.attribute(target_class_idx=1, **self.batch)

        for key in baseline:
            self.assertTrue(torch.equal(baseline[key], with_target[key]))

    def test_callable_matches_attribute(self):
        """Calling the interpreter is equivalent to attribute()."""
        interpreter = RawAttention(self.model)
        via_attribute = interpreter.attribute(**self.batch)
        via_call = interpreter(**self.batch)

        for key in via_attribute:
            self.assertTrue(torch.equal(via_attribute[key], via_call[key]))

    def test_out_of_range_layer_raises_value_error(self):
        """A layer index the model doesn't have raises ValueError."""
        with self.assertRaises(ValueError):
            RawAttention(self.model, layer=2).attribute(**self.batch)
        with self.assertRaises(ValueError):
            RawAttention(self.model, layer=-3).attribute(**self.batch)

    def test_invalid_arguments_raise(self):
        """Bad head_fusion, non-int layer and incompatible models are rejected."""

        class PlainModel(nn.Module):
            def forward(self, **data):
                return {"logit": torch.zeros(1, 1)}

        with self.assertRaises(ValueError):
            RawAttention(self.model, head_fusion="min")
        with self.assertRaises(TypeError):
            RawAttention(self.model, layer="last")
        with self.assertRaises(TypeError):
            RawAttention(PlainModel())

    def test_hooks_disabled_after_attribute(self):
        """attribute() turns hooks off, so inference under no_grad still works."""
        RawAttention(self.model).attribute(**self.batch)
        with torch.no_grad():
            self.model(**self.batch)


class TestRawAttentionStageAttentionNet(unittest.TestCase):
    """RawAttention also works on StageAttentionNet (single attention layer)."""

    def test_stageattn_attributions(self):
        """Nested visits: per-visit scores are expanded over each visit's codes."""
        torch.manual_seed(42)
        samples = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "codes": ([0.0, 1.5], [["P1", "P2", "P3"], ["P4", "P5"]]),
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "codes": ([0.0], [["P6"]]),
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "stagenet"},
            output_schema={"label": "binary"},
            dataset_name="raw_attention_stageattn_test",
        )
        model = StageAttentionNet(dataset=dataset, chunk_size=2, levels=3, num_heads=3)
        batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))

        attributions = RawAttention(model).attribute(**batch)

        values = batch["codes"][
            dataset.input_processors["codes"].schema().index("value")
        ]
        self.assertEqual(set(attributions), {"codes"})
        self.assertEqual(attributions["codes"].shape, values.shape)
        self.assertTrue(torch.all(attributions["codes"] >= 0))
        # every code in a visit gets that visit's score
        first_visit = attributions["codes"][0, 0]
        self.assertTrue(torch.all(first_visit == first_visit[0]))
        with self.assertRaises(ValueError):
            RawAttention(model, layer=1).attribute(**batch)


if __name__ == "__main__":
    unittest.main()
