import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
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


if __name__ == "__main__":
    unittest.main()
