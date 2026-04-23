import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TPC


def _make_ts_payload(
    n_features: int = 2,
    pred_hours: int = 5,
    seed: int = 0,
) -> dict:
    """Build a minimal raw ts payload as produced by RemainingLengthOfStayTPC_MIMIC4."""
    rng = np.random.RandomState(seed)
    prefill_start = datetime(2020, 1, 1, 0, 0, 0)
    pred_start = prefill_start + timedelta(hours=5)
    pred_end = pred_start + timedelta(hours=pred_hours)

    feature_itemids = [str(i) for i in range(n_features)]

    # One observation per feature at hour 0 (prefill window)
    timestamps = [prefill_start + timedelta(hours=0)] * n_features
    itemids = feature_itemids[:]
    values = rng.uniform(1.0, 10.0, size=n_features).tolist()

    return {
        "prefill_start": prefill_start,
        "icu_start": prefill_start,
        "pred_start": pred_start,
        "pred_end": pred_end,
        "feature_itemids": feature_itemids,
        "long_df": {
            "timestamp": timestamps,
            "itemid": itemids,
            "value": values,
            "source": ["chartevents"] * n_features,
        },
    }


def _make_static_dict() -> dict:
    """Build a minimal static feature dict as produced by RemainingLengthOfStayTPC_MIMIC4."""
    return {
        "gender": "M",
        "race": "WHITE",
        "admission_location": "EMERGENCY ROOM",
        "insurance": "Medicare",
        "first_careunit": "Medical Intensive Care Unit (MICU)",
        "hour_of_admission": 8,
        "admission_height": 170.0,
        "admission_weight": 80.0,
        "gcs_eye": 4.0,
        "gcs_motor": 6.0,
        "gcs_verbal": 5.0,
        "anchor_age": 65,
    }


def _make_dataset(
    n_samples: int = 4,
    n_features: int = 2,
    pred_hours: int = 5,
) -> "SampleDataset":
    """Create a minimal in-memory SampleDataset for TPC testing."""
    samples = []
    for i in range(n_samples):
        ts = _make_ts_payload(n_features=n_features, pred_hours=pred_hours, seed=i)
        static = _make_static_dict()
        # Alternate gender/careunit for categorical vocab diversity
        static["gender"] = "M" if i % 2 == 0 else "F"
        static["first_careunit"] = (
            "Medical Intensive Care Unit (MICU)"
            if i % 2 == 0
            else "Surgical Intensive Care Unit (SICU)"
        )
        y = [max(float(pred_hours - h) / 24.0, 1.0 / 48.0) for h in range(pred_hours)]
        samples.append(
            {
                "patient_id": f"p{i}",
                "stay_id": f"s{i}",
                "hadm_id": f"h{i}",
                "ts": ts,
                "static": static,
                "y": y,
            }
        )

    return create_sample_dataset(
        samples=samples,
        input_schema={"ts": ("tpc_timeseries", {}), "static": ("tpc_static", {})},
        output_schema={"y": ("regression_sequence", {})},
        dataset_name="test_tpc",
        in_memory=True,
    )


class TestTPC(unittest.TestCase):
    """Unit tests for the TPC model."""

    def setUp(self):
        """Set up shared dataset and model for all tests."""
        torch.manual_seed(0)
        np.random.seed(0)

        self.n_features = 2
        self.pred_hours = 5
        self.batch_size = 2

        self.dataset = _make_dataset(
            n_samples=4,
            n_features=self.n_features,
            pred_hours=self.pred_hours,
        )

        self.model = TPC(
            dataset=self.dataset,
            temporal_channels=4,
            pointwise_channels=3,
            num_layers=2,
            kernel_size=2,
            main_dropout=0.0,
            temporal_dropout=0.0,
            use_batchnorm=False,  # off for small batch sizes in tests
            final_hidden=8,
        )

        self.loader = get_dataloader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

    def test_model_initialisation(self):
        """Model initialises correctly and infers F and S from processors."""
        self.assertIsInstance(self.model, TPC)
        self.assertEqual(self.model.F, self.n_features)
        self.assertGreater(self.model.S, 0)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(self.model.temporal_channels, 4)
        self.assertEqual(self.model.pointwise_channels, 3)
        self.assertIn("ts", self.model.feature_keys)
        self.assertIn("static", self.model.feature_keys)
        self.assertEqual(len(self.model.label_keys), 1)
        self.assertEqual(self.model.label_keys[0], "y")
        self.assertEqual(self.model.mode, "regression")

    def test_blocks_count(self):
        """Number of TPCBlocks equals num_layers."""
        self.assertEqual(len(self.model.blocks), self.model.num_layers)

    def test_feature_dimension_growth(self):
        """Feature dimension grows by pointwise_channels each layer."""
        expected_in_features = self.n_features
        for i, block in enumerate(self.model.blocks):
            self.assertEqual(block.in_features, expected_in_features)
            expected_in_features += self.model.pointwise_channels


    def test_forward_output_keys(self):
        """Forward pass returns required output keys."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

    def test_forward_loss_is_scalar(self):
        """Loss is a scalar tensor."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = self.model(**batch)
        self.assertEqual(ret["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_forward_y_prob_y_true_are_1d_masked(self):
        """y_prob and y_true are 1D and contain only real (unpadded) timesteps."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = self.model(**batch)

        self.assertEqual(ret["y_prob"].dim(), 1)
        self.assertEqual(ret["y_true"].dim(), 1)
        # Both should have the same length (all real timesteps in the batch)
        self.assertEqual(ret["y_prob"].shape[0], ret["y_true"].shape[0])
        # No padded zeros in y_true
        self.assertTrue((ret["y_true"] > 0).all())

    def test_forward_logit_shape(self):
        """Logit has shape (B, T_max) — full padded output."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = self.model(**batch)

        self.assertEqual(ret["logit"].dim(), 2)
        self.assertEqual(ret["logit"].shape[0], self.batch_size)

    def test_forward_y_prob_bounds(self):
        """Predictions are within HardTanh bounds [1/48, 100] days."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = self.model(**batch)

        self.assertTrue((ret["y_prob"] >= 1.0 / 48.0).all())
        self.assertTrue((ret["y_prob"] <= 100.0).all())

    def test_forward_without_labels(self):
        """Forward pass without labels returns y_prob and logit but no loss or y_true."""
        batch = next(iter(self.loader))
        batch_no_labels = {k: v for k, v in batch.items() if k != "y"}

        with torch.no_grad():
            ret = self.model(**batch_no_labels)

        self.assertIn("y_prob", ret)
        self.assertIn("logit", ret)
        self.assertNotIn("loss", ret)
        self.assertNotIn("y_true", ret)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def test_backward_pass(self):
        """Loss backward populates gradients on model parameters."""
        batch = next(iter(self.loader))
        ret = self.model(**batch)
        ret["loss"].backward()

        has_grad = any(
            p.requires_grad and p.grad is not None
            for p in self.model.parameters()
        )
        self.assertTrue(has_grad, "No parameters have gradients after backward pass.")

    def test_loss_decreases_after_one_step(self):
        """A single gradient step reduces the training loss."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        batch = next(iter(self.loader))

        ret_before = self.model(**batch)
        loss_before = ret_before["loss"].item()

        optimizer.zero_grad()
        ret_before["loss"].backward()
        optimizer.step()

        with torch.no_grad():
            ret_after = self.model(**batch)
        loss_after = ret_after["loss"].item()

        self.assertLess(loss_after, loss_before)

    def test_custom_hyperparameters(self):
        """Model initialises and runs forward with non-default hyperparameters."""
        model = TPC(
            dataset=self.dataset,
            temporal_channels=8,
            pointwise_channels=4,
            num_layers=3,
            kernel_size=3,
            main_dropout=0.1,
            temporal_dropout=0.1,
            use_batchnorm=False,
            final_hidden=16,
        )

        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertTrue(torch.isfinite(ret["loss"]))

    def test_tpc_block_output_shape(self):
        """TPCBlock output has correct shape (B, T, R+Z, Y+1)."""
        from pyhealth.models.tpc import TPCBlock

        B, T, R, C = 2, 10, self.n_features, 2
        Y, Z, S = 4, 3, self.model.S

        block = TPCBlock(
            in_features=R,
            in_channels=C,
            temporal_channels=Y,
            pointwise_channels=Z,
            kernel_size=2,
            dilation=1,
            main_dropout=0.0,
            temporal_dropout=0.0,
            use_batchnorm=False,
            static_dim=S,
        )

        x = torch.randn(B, T, R, C)
        static = torch.randn(B, S)

        with torch.no_grad():
            out = block(x, static=static)

        self.assertEqual(out.shape, (B, T, R + Z, Y + 1))

    def test_tpc_block_wrong_input_raises(self):
        """TPCBlock raises ValueError when input shape does not match."""
        from pyhealth.models.tpc import TPCBlock

        block = TPCBlock(
            in_features=4,
            in_channels=2,
            temporal_channels=4,
            pointwise_channels=3,
            kernel_size=2,
            dilation=1,
            main_dropout=0.0,
            temporal_dropout=0.0,
            use_batchnorm=False,
            static_dim=0,
        )

        x_wrong = torch.randn(2, 10, 3, 2)  # R=3 but block expects R=4
        with self.assertRaises(ValueError):
            block(x_wrong)

    def test_msle_loss_is_non_negative(self):
        """MSLE loss is always >= 0."""
        batch = next(iter(self.loader))
        with torch.no_grad():
            ret = self.model(**batch)
        self.assertGreaterEqual(ret["loss"].item(), 0.0)

if __name__ == "__main__":
    unittest.main()