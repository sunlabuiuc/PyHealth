"""Tests for the GA2M model (synthetic data only, fast)."""

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.ga2m import (
    GA2M,
    MISSING_VALUE,
    all_pairs,
    rank_pairs_by_contribution_variance,
)


def _samples(n: int = 6, n_features: int = 4):
    samples = []
    for i in range(n):
        features = [float((i + j) % 5) for j in range(n_features)]
        samples.append(
            {
                "patient_id": f"p{i}",
                "features": features,
                "mortality": i % 2,
            }
        )
    return samples


def _dataset(n: int = 6, n_features: int = 4, name: str = "test_ga2m"):
    return create_sample_dataset(
        samples=_samples(n, n_features),
        input_schema={"features": "tensor"},
        output_schema={"mortality": "binary"},
        dataset_name=name,
    )


class TestGA2M(unittest.TestCase):
    """Test cases for the GA2M model."""

    def setUp(self):
        self.dataset = _dataset()
        self.model = GA2M(dataset=self.dataset, n_bins=4, pairs=[(0, 1), (2, 3)])

    def test_model_initialization(self):
        self.assertEqual(self.model.n_features, 4)
        self.assertEqual(self.model.label_key, "mortality")
        # 1D tables: one row per feature, n_bins + 1 (reserved missing bin).
        self.assertEqual(tuple(self.model.terms_1d.shape), (4, 5))
        # 2D tables: one per requested pair.
        self.assertEqual(len(self.model.terms_2d), 2)
        for table in self.model.terms_2d.values():
            self.assertEqual(tuple(table.shape), (5, 5))
        self.assertEqual(tuple(self.model.bin_edges.shape), (4, 3))

    def test_invalid_pair_raises(self):
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, pairs=[(0, 0)])
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, pairs=[(0, 99)])

    def test_duplicate_pair_raises(self):
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, pairs=[(0, 1), (1, 0)])
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, pairs=[(0, 1), (0, 1)])

    def test_non_binary_mode_raises(self):
        dataset = create_sample_dataset(
            samples=[
                {"patient_id": f"p{i}", "features": [float(i), 1.0], "label": i % 3}
                for i in range(6)
            ],
            input_schema={"features": "tensor"},
            output_schema={"label": "multiclass"},
            dataset_name="test_ga2m_multiclass",
        )
        with self.assertRaises(ValueError):
            GA2M(dataset=dataset)

    def test_fit_bins_rejects_drop_last_loader(self):
        loader = get_dataloader(self.dataset, batch_size=3, drop_last=True)
        with self.assertRaises(ValueError):
            self.model.fit_bins(loader)

    def test_feature_names_length_checked(self):
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, feature_names=["a", "b"])

    def test_forward_shapes_and_keys(self):
        train_loader = get_dataloader(self.dataset, batch_size=3, shuffle=False)
        batch = next(iter(train_loader))
        ret = self.model(**batch)
        self.assertEqual(set(ret.keys()), {"logit", "y_prob", "y_true", "loss"})
        self.assertEqual(ret["logit"].shape, (3, 1))
        self.assertEqual(ret["y_prob"].shape, (3, 1))
        self.assertEqual(ret["y_true"].shape, (3, 1))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_forward_without_label_key(self):
        train_loader = get_dataloader(self.dataset, batch_size=3, shuffle=False)
        batch = next(iter(train_loader))
        batch.pop("mortality")
        ret = self.model(**batch)
        self.assertNotIn("loss", ret)
        self.assertNotIn("y_true", ret)

    def test_gradients_flow_to_1d_and_2d_tables(self):
        batch = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))
        ret = self.model(**batch)
        ret["loss"].backward()
        self.assertIsNotNone(self.model.terms_1d.grad)
        self.assertTrue(torch.any(self.model.terms_1d.grad != 0))
        for table in self.model.terms_2d.values():
            self.assertIsNotNone(table.grad)
        self.assertIsNotNone(self.model.bias.grad)

    def test_state_dict_round_trip(self):
        """A trained model's predictions survive a state_dict save/load
        round trip into a freshly constructed model (docs/how_to_contribute
        .rst's model-test checklist: "Model can save and load state
        correctly")."""
        batch = next(iter(get_dataloader(self.dataset, batch_size=6, shuffle=False)))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        for _ in range(5):
            optimizer.zero_grad()
            self.model(**batch)["loss"].backward()
            optimizer.step()
        # Confirms the round trip below is meaningful, not just two
        # zero-initialized models trivially agreeing.
        self.assertTrue(torch.any(self.model.terms_1d != 0))

        self.model.eval()
        with torch.no_grad():
            original_logit = self.model(**batch)["logit"]

        new_model = GA2M(dataset=self.dataset, n_bins=4, pairs=[(0, 1), (2, 3)])
        new_model.load_state_dict(self.model.state_dict())
        new_model.eval()
        with torch.no_grad():
            loaded_logit = new_model(**batch)["logit"]

        torch.testing.assert_close(original_logit, loaded_logit)

    def test_determinism_in_eval_mode(self):
        self.model.eval()
        batch = next(iter(get_dataloader(self.dataset, batch_size=4, shuffle=False)))
        with torch.no_grad():
            first = self.model(**batch)["logit"]
            second = self.model(**batch)["logit"]
        torch.testing.assert_close(first, second)

    def test_missing_value_uses_reserved_bin(self):
        """Two different 'true' underlying values that are both reported as
        the -1.0 sentinel land in the same (reserved) bin and therefore
        produce the same 1D contribution, regardless of magnitude."""
        self.model.eval()
        with torch.no_grad():
            a = torch.tensor([[MISSING_VALUE, 1.0, 1.0, 1.0]])
            b = torch.tensor([[MISSING_VALUE, 1.0, 1.0, 1.0]])
            bins_a = self.model._bin_indices(a, self.model.bin_edges)
            bins_b = self.model._bin_indices(b, self.model.bin_edges)
            self.assertEqual(bins_a[0, 0].item(), 0)
            torch.testing.assert_close(bins_a, bins_b)
            # Untrained tables are all zero, so give the missing bin and a
            # real bin of feature 0 different weights to make the contrast
            # observable.
            self.model.terms_1d[0, 0] = 5.0
            self.model.terms_1d[0, 1:] = -5.0
            out_missing = self.model(features=a)["logit"]
            present = a.clone()
            present[0, 0] = 2.0
            out_present = self.model(features=present)["logit"]
        self.assertFalse(torch.allclose(out_missing, out_present))

    def test_fit_bins_changes_edges_and_uses_train_split_only(self):
        before = self.model.bin_edges.clone()
        bigger_dataset = _dataset(n=20, name="test_ga2m_refit")
        self.model.fit_bins(bigger_dataset)
        self.assertFalse(torch.equal(before, self.model.bin_edges))
        self.assertEqual(tuple(self.model.bin_edges.shape), tuple(before.shape))

    def test_all_pairs_count(self):
        self.assertEqual(len(all_pairs(34)), 561)
        self.assertEqual(len(all_pairs(4)), 6)

    def test_rank_pairs_by_contribution_variance(self):
        loader = get_dataloader(self.dataset, batch_size=6, shuffle=False)
        with torch.no_grad():
            self.model.terms_2d[self.model._pair_key(0, 1)].add_(
                torch.randn(5, 5) * 10
            )
        ranked = rank_pairs_by_contribution_variance(self.model, loader)
        self.assertEqual(len(ranked), 2)
        self.assertEqual(ranked[0][0], (0, 1))
        self.assertGreater(ranked[0][1], ranked[1][1])

    def test_rank_pairs_with_no_pairs_returns_empty(self):
        model = GA2M(dataset=self.dataset, n_bins=4)
        self.assertEqual(
            rank_pairs_by_contribution_variance(
                model, get_dataloader(self.dataset, batch_size=4)
            ),
            [],
        )

    def test_get_shape_functions(self):
        shapes = self.model.get_shape_functions()
        self.assertIn("x0", shapes)
        self.assertIn(("x0", "x1"), shapes)
        edges, values = shapes["x0"]
        self.assertEqual(values.shape, (5,))
        self.assertEqual(edges.shape, (3,))
        # Centered over the real bins only: the mean of the real bins
        # (indices 1:) is ~0, not necessarily the mean of all 5 entries.
        self.assertAlmostEqual(values[1:].mean().item(), 0.0, places=5)

    def test_get_shape_functions_centers_on_real_bins_only(self):
        """A large missing-bin weight must not shift the real-bin curve."""
        with torch.no_grad():
            self.model.terms_1d[0, 0] = 1000.0  # missing bin: huge weight
            self.model.terms_1d[0, 1:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        _, values = self.model.get_shape_functions()["x0"]
        real_mean = torch.tensor([1.0, 2.0, 3.0, 4.0]).mean()
        # Real bins are centered on their own mean, unaffected by the
        # missing bin's weight.
        torch.testing.assert_close(
            values[1:], torch.tensor([1.0, 2.0, 3.0, 4.0]) - real_mean
        )
        # The missing bin's displayed value is relative to that same
        # baseline (not its own, now-meaningless, "self-mean").
        torch.testing.assert_close(values[0], torch.tensor(1000.0) - real_mean)

    def test_excluded_feature_has_no_effect_or_gradient(self):
        model = GA2M(dataset=self.dataset, n_bins=4, excluded_features=[1])
        with torch.no_grad():
            model.terms_1d[1] = 5.0  # must be ignored by forward
        batch = next(iter(get_dataloader(self.dataset, batch_size=4)))
        ret = model(**batch)
        ret["loss"].backward()
        self.assertTrue(torch.all(model.terms_1d.grad[1] == 0))
        self.assertTrue(torch.any(model.terms_1d.grad[0] != 0))
        with torch.no_grad():
            model.terms_1d[1] = 0.0
        torch.testing.assert_close(model(**batch)["logit"], ret["logit"])
        self.assertNotIn("x1", model.get_shape_functions())

    def test_smoothness_penalty_zero_for_linear_curves_only(self):
        model = GA2M(dataset=self.dataset, n_bins=4, pairs=[(0, 1)])
        with torch.no_grad():
            model.terms_1d[:, 1:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
            model.terms_1d[:, 0] = 100.0  # missing bin: not smoothed
        self.assertAlmostEqual(model.smoothness_penalty().item(), 0.0, places=6)
        with torch.no_grad():
            model.terms_1d[0, 2] = 5.0
        penalty = model.smoothness_penalty()
        self.assertGreater(penalty.item(), 0.0)
        penalty.backward()
        self.assertTrue(torch.all(model.terms_1d.grad[:, 0] == 0))

    def test_excluded_feature_rejected_in_pairs_and_range(self):
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, pairs=[(0, 1)], excluded_features=[1])
        with self.assertRaises(ValueError):
            GA2M(dataset=self.dataset, excluded_features=[99])

    def test_missing_only_feature_falls_back_to_single_bin(self):
        """A feature with fewer than 2 observed values still produces a
        valid (degenerate) set of edges rather than raising."""
        samples = [
            {
                "patient_id": f"p{i}",
                "features": [MISSING_VALUE, 1.0],
                "mortality": i % 2,
            }
            for i in range(4)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"features": "tensor"},
            output_schema={"mortality": "binary"},
            dataset_name="test_ga2m_all_missing",
        )
        model = GA2M(dataset=dataset, n_bins=4)
        batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
        ret = model(**batch)
        self.assertTrue(torch.isfinite(ret["logit"]).all())


if __name__ == "__main__":
    unittest.main()
