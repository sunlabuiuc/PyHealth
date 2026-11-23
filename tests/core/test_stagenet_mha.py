import unittest
import warnings

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import (
    AttentionRollout,
    CheferRelevance,
    RawAttention,
)
from pyhealth.models.stagenet_mha import StageAttentionNet as StageNetMHA


class TestStageNetMHA(unittest.TestCase):
    """Tests for the StageNet variant with MHA inserted after SA-LSTM."""

    def setUp(self):
        # Mixed input types to exercise masking and time handling
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ([0.0, 1.0, 2.0], ["A1", "A2", "A3"]),
                "procedures": (
                    [0.0, 1.5],
                    [["P1", "P2", "P3"], ["P4", "P5"]],
                ),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": ([0.0, 2.0], ["B1", "B2"]),
                "procedures": ([0.0], [["P6"]]),
                "label": 0,
            },
        ]

        self.input_schema = {"codes": "stagenet", "procedures": "stagenet"}
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test_stagenet_mha",
        )

        # hidden_dim = chunk_size * levels = 2 * 3 = 6 so pick num_heads=3
        self.model = StageNetMHA(
            dataset=self.dataset, chunk_size=2, levels=3, num_heads=3
        )

    def test_forward_pass(self):
        """Forward pass returns expected keys and shapes."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)
        self.assertEqual(ret["logit"].shape[0], 2)
        self.assertEqual(ret["y_prob"].shape[0], 2)
        self.assertEqual(ret["loss"].dim(), 0)

    def test_backward_pass(self):
        """Backward pass produces gradients through the MHA variant."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        ret = self.model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No gradients found after backward pass")

    def test_time_none_support(self):
        """MHA variant works when time intervals are absent (mask-driven)."""
        samples_no_time = [
            {
                "patient_id": "p0",
                "visit_id": "v0",
                "codes": (None, ["X1", "X2"]),
                "label": 1,
            },
            {
                "patient_id": "p1",
                "visit_id": "v1",
                "codes": (None, ["Y1"]),
                "label": 0,
            },
        ]
        dataset_no_time = create_sample_dataset(
            samples=samples_no_time,
            input_schema={"codes": "stagenet"},
            output_schema=self.output_schema,
            dataset_name="test_stagenet_mha_no_time",
        )
        model_no_time = StageNetMHA(
            dataset=dataset_no_time, chunk_size=2, levels=2, num_heads=2
        )

        loader = get_dataloader(dataset_no_time, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        time, _ = batch["codes"]
        self.assertIsNone(time)

        with torch.no_grad():
            ret = model_no_time(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape[0], 2)

    def test_attention_hook_records_map_and_grad(self):
        """Attention hook exposes maps and gradients for each feature."""
        loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        self.model.set_attention_hooks(True)
        ret = self.model(**batch)
        ret["loss"].backward()
        self.model.set_attention_hooks(False)

        for feature_key, layer in self.model.stagenet.items():
            attn_map = layer.get_attn_map()
            self.assertIsNotNone(attn_map, f"{feature_key} attn_map missing")
            self.assertEqual(attn_map.dim(), 4)
            self.assertEqual(attn_map.shape[1], layer.mha.h)
            self.assertEqual(attn_map.shape[-2], attn_map.shape[-1])

            attn_grad = layer.get_attn_grad()
            self.assertIsNotNone(attn_grad, f"{feature_key} attn_grad missing")
            self.assertEqual(attn_grad.shape, attn_map.shape)
            self.assertFalse(torch.isnan(attn_grad).any())


def _marked_relevance(batch_size, seq_len):
    """Relevance matrix whose row ``t`` is filled with the value ``t``."""
    rows = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    return rows.expand(batch_size, seq_len, seq_len).contiguous().clone()


class TestStageAttentionNetFlatRelevance(unittest.TestCase):
    """Issue #1498: flat StageNet code sequences must keep a per-timestep mask."""

    def setUp(self):
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "codes": ([0.0, 1.0, 2.0], ["A1", "A2", "A3"]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "codes": ([0.0, 2.0], ["B1", "B2"]),
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "stagenet"},
            output_schema={"label": "binary"},
            dataset_name="test_stagenet_mha_flat",
        )
        self.model = StageNetMHA(
            dataset=dataset, chunk_size=2, levels=2, num_heads=2
        )
        loader = get_dataloader(dataset, batch_size=2, shuffle=False)
        self.batch = next(iter(loader))

    def test_get_relevance_tensor_flat_stagenet_codes(self):
        """Flat padded code sequences return one relevance score per timestep."""
        _, values = self.batch["codes"]
        batch_size, seq_len = values.shape

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rollout = AttentionRollout(self.model).attribute(**self.batch)
            raw = RawAttention(self.model).attribute(**self.batch)
            chefer = CheferRelevance(self.model).attribute(**self.batch)
            direct = self.model.get_relevance_tensor(
                {"codes": _marked_relevance(batch_size, seq_len)},
                **self.batch,
            )

        for name, attribution in (
            ("AttentionRollout", rollout),
            ("RawAttention", raw),
            ("CheferRelevance", chefer),
        ):
            self.assertEqual(
                tuple(attribution["codes"].shape),
                (batch_size, seq_len),
                name,
            )
            self.assertFalse(torch.isnan(attribution["codes"]).any(), name)
        self.assertEqual(tuple(direct["codes"].shape), (batch_size, seq_len))

    def test_get_relevance_tensor_picks_last_unpadded_timestep(self):
        """Padded flat codes select the last real timestep, not the pad slot."""
        _, values = self.batch["codes"]
        batch_size, seq_len = values.shape
        self.assertEqual((batch_size, seq_len), (2, 3))
        self.assertNotEqual(values[1, 1].item(), 0)
        self.assertEqual(values[1, 2].item(), 0)

        picked = self.model.get_relevance_tensor(
            {"codes": _marked_relevance(batch_size, seq_len)},
            **self.batch,
        )["codes"]

        self.assertTrue(torch.equal(picked[0], torch.full((seq_len,), 2.0)))
        self.assertTrue(torch.equal(picked[1], torch.full((seq_len,), 1.0)))

    def test_get_relevance_tensor_explicit_mask(self):
        """An explicit per-timestep mask overrides the values."""
        embedding = self.model.get_embedding_model()
        self.assertIsInstance(embedding, torch.nn.Module)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outputs = self.model(**self.batch, embed=True)
        self.assertIn("embed", outputs)
        self.assertEqual(outputs["embed"].shape[0], 2)

        time, values = self.batch["codes"]
        batch_size, seq_len = values.shape
        mask = (values != 0).int()
        mask[0].zero_()
        mask[0, 0] = 1
        overridden = dict(self.batch)
        overridden["codes"] = (time, values, mask)

        picked = self.model.get_relevance_tensor(
            {"codes": _marked_relevance(batch_size, seq_len)},
            **overridden,
        )["codes"]
        self.assertTrue(torch.equal(picked[0], torch.zeros(seq_len)))
        self.assertTrue(torch.equal(picked[1], torch.full((seq_len,), 1.0)))

    def test_get_relevance_tensor_falls_back_to_last_position_without_value(self):
        """With no value and no mask, every row uses the final position."""
        _, values = self.batch["codes"]
        batch_size, seq_len = values.shape
        processor = self.model.dataset.input_processors["codes"]
        original_schema = processor.schema
        processor.schema = lambda: ("time",)
        try:
            picked = self.model.get_relevance_tensor(
                {"codes": _marked_relevance(batch_size, seq_len)},
                codes=(self.batch["codes"][0],),
            )["codes"]
        finally:
            processor.schema = original_schema

        expected = torch.full((seq_len,), float(seq_len - 1))
        self.assertTrue(torch.equal(picked[0], expected))
        self.assertTrue(torch.equal(picked[1], expected))


class TestStageAttentionNetFlatContinuousRelevance(unittest.TestCase):
    """Flat stagenet_tensor values are [batch, seq] and have no feature axis."""

    def setUp(self):
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "vitals": ([0.0, 1.0, 2.0], [1.5, 2.0, 2.5]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "vitals": ([0.0, 1.0], [3.0, 4.0]),
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"vitals": "stagenet_tensor"},
            output_schema={"label": "binary"},
            dataset_name="test_stagenet_mha_flat_continuous",
        )
        self.model = StageNetMHA(
            dataset=dataset, chunk_size=2, levels=2, num_heads=2
        )
        self.batch = next(
            iter(get_dataloader(dataset, batch_size=2, shuffle=False))
        )

    def test_flat_continuous_keeps_a_per_timestep_mask(self):
        """A scalar series must not collapse the sequence axis."""
        processor = self.model.dataset.input_processors["vitals"]
        _, values = self.batch["vitals"]
        self.assertFalse(processor.is_token())
        self.assertEqual(processor.dim(), (1, 1))
        self.assertEqual(tuple(values.shape), (2, 3))
        self.assertEqual(values[1, 2].item(), 0.0)
        self.assertNotEqual(values[1, 1].item(), 0.0)

        batch_size, seq_len = values.shape
        picked = self.model.get_relevance_tensor(
            {"vitals": _marked_relevance(batch_size, seq_len)},
            **self.batch,
        )["vitals"]
        self.assertTrue(torch.equal(picked[0], torch.full((seq_len,), 2.0)))
        self.assertTrue(torch.equal(picked[1], torch.full((seq_len,), 1.0)))

    def test_explicit_sequence_mask_is_not_collapsed(self):
        """A [batch, seq] mask is already time-aligned and must be kept."""
        time, values = self.batch["vitals"]
        batch_size, seq_len = values.shape
        mask = (values != 0).int()
        mask[0].zero_()
        mask[0, 0] = 1
        overridden = dict(self.batch)
        overridden["vitals"] = (time, values, mask)

        picked = self.model.get_relevance_tensor(
            {"vitals": _marked_relevance(batch_size, seq_len)},
            **overridden,
        )["vitals"]
        self.assertTrue(torch.equal(picked[0], torch.zeros(seq_len)))
        self.assertTrue(torch.equal(picked[1], torch.full((seq_len,), 1.0)))


class TestStageAttentionNetRelevanceOtherFeatures(unittest.TestCase):
    """Continuous measurements and nested visits must keep their own masks."""

    def test_continuous_features_and_nested_codes_keep_last_valid_step(self):
        samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "labs": (
                    [0.0, 1.0, 2.0],
                    [[1.0, 0.0], [2.0, 3.0], [4.0, 5.0]],
                ),
                "visits": ([0.0, 1.0], [["A", "B"], ["C"]]),
                "label": 1,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "labs": ([0.0, 1.0], [[1.0, 2.0], [0.0, 3.0]]),
                "visits": ([0.0], [["D"]]),
                "label": 0,
            },
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"labs": "stagenet_tensor", "visits": "stagenet"},
            output_schema={"label": "binary"},
            dataset_name="test_stagenet_mha_continuous",
        )
        model = StageNetMHA(
            dataset=dataset, chunk_size=2, levels=2, num_heads=2
        )
        batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))

        _, labs = batch["labs"]
        _, visits = batch["visits"]
        self.assertEqual(labs.shape[-1], 2)
        self.assertEqual(labs[0, 0, 1].item(), 0.0)
        self.assertNotEqual(labs[0, 0, 0].item(), 0.0)
        self.assertEqual(tuple(labs.shape), (2, 3, 2))
        self.assertTrue(torch.all(labs[1, 2] == 0))
        self.assertEqual(visits.shape[0], 2)
        self.assertTrue(torch.all(visits[1, -1] == 0))

        lab_rows = _marked_relevance(labs.shape[0], labs.shape[1])
        visit_rows = _marked_relevance(visits.shape[0], visits.shape[1])
        picked = model.get_relevance_tensor(
            {"labs": lab_rows, "visits": visit_rows},
            **batch,
        )
        self.assertTrue(torch.equal(picked["labs"][0], torch.full((3,), 2.0)))
        self.assertTrue(torch.equal(picked["labs"][1], torch.full((3,), 1.0)))
        self.assertTrue(
            torch.equal(
                picked["visits"][0],
                torch.full((visits.shape[1],), float(visits.shape[1] - 1)),
            )
        )
        self.assertTrue(torch.equal(picked["visits"][1], torch.zeros(visits.shape[1])))

        lab_time, lab_values = batch["labs"]
        feature_mask = (lab_values.abs().sum(dim=-1, keepdim=True) != 0).int()
        feature_mask = feature_mask.expand_as(lab_values).clone()
        feature_mask[0, 2] = 0
        masked = dict(batch)
        masked["labs"] = (lab_time, lab_values, feature_mask)
        masked_labs = model.get_relevance_tensor(
            {"labs": lab_rows, "visits": visit_rows},
            **masked,
        )["labs"]
        self.assertTrue(torch.equal(masked_labs[0], torch.full((3,), 1.0)))
        self.assertTrue(torch.equal(masked_labs[1], torch.full((3,), 1.0)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            attributions = AttentionRollout(model).attribute(**batch)
        self.assertEqual(tuple(attributions["labs"].shape), tuple(labs.shape))
        self.assertEqual(tuple(attributions["visits"].shape), tuple(visits.shape))


if __name__ == "__main__":
    unittest.main()
