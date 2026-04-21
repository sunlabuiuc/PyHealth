"""Tests for the GA2M PyHealth model.

All tests use small synthetic data (2-5 patients, minimal features) and
complete in milliseconds. No real datasets (MIMIC etc.) are required.

Run with:
    python test_ga2m.py
or via pytest:
    pytest test_ga2m.py -v
"""

import torch
import numpy as np
import pytest
from pyhealth.models.ga2m import GA2M, UNKNOWN_SENTINEL
import pandas as pd

# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_dataset(
    n_patients: int = 4,
    n_features: int = 6,
    include_unknowns: bool = False,
):
    """Build a minimal InMemorySampleDataset for testing.

    Uses PyHealth's create_sample_dataset with in_memory=True so no disk
    I/O is needed and tests stay millisecond-fast.

    Args:
        n_patients: Number of synthetic ICU stays.
        n_features: Dimensionality of the feature vector.
        include_unknowns: If True, inserts UNKNOWN_SENTINEL values into
            some features to test the unknown bin routing.

    Returns:
        SampleDataset instance.
    """
    from pyhealth.datasets import create_sample_dataset

    rng = np.random.default_rng(42)
    samples = []
    for i in range(n_patients):
        features = rng.uniform(0.0, 5.0, size=n_features).tolist()
        if include_unknowns and i % 2 == 0:
            features[0] = UNKNOWN_SENTINEL  # mark first feature unknown
        samples.append({
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "features": features,
            "label": int(rng.integers(0, 2)),
        })

    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_test",
        in_memory=True,
    )


def _make_loader(dataset, batch_size: int = 4):
    """Wrap dataset in a DataLoader."""
    from pyhealth.datasets import get_dataloader
    return get_dataloader(dataset, batch_size=batch_size)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_dataset():
    return _make_dataset(n_patients=4, n_features=6)


@pytest.fixture(scope="module")
def unknown_dataset():
    return _make_dataset(n_patients=4, n_features=6, include_unknowns=True)


@pytest.fixture(scope="module")
def fitted_model(base_dataset):
    """A GA2M that has completed both Stage 1 and Stage 2 setup."""
    loader = _make_loader(base_dataset)
    model = GA2M(
        dataset=base_dataset,
        n_bins=8,                  # small bins for speed
        top_k_interactions=3,      # few interactions for speed
        use_interactions=True,
    )
    model.fit_bins(loader)
    model.fit_main_effects(loader, epochs=2, lr=1e-2)
    model.select_top_interactions()
    return model


# ---------------------------------------------------------------------------
# 1. Instantiation
# ---------------------------------------------------------------------------

class TestInstantiation:

    def test_instantiation_defaults(self, base_dataset):
        """Model instantiates with paper-default hyperparameters."""
        model = GA2M(dataset=base_dataset, n_bins=256, top_k_interactions=34)
        assert model.n_bins == 256
        assert model.top_k_interactions == 34
        assert model.use_interactions is True

    def test_instantiation_no_interactions(self, base_dataset):
        """use_interactions=False creates no interaction modules."""
        model = GA2M(dataset=base_dataset, n_bins=8, use_interactions=False)
        assert len(model.interactions) == 0

    def test_main_effects_count(self, base_dataset):
        """One embedding per input feature is created."""
        model = GA2M(dataset=base_dataset, n_bins=8)
        assert len(model.main_effects) == model.input_dim

    def test_embedding_size(self, base_dataset):
        """Each main effect embedding has n_bins + 1 rows (unknown bin)."""
        model = GA2M(dataset=base_dataset, n_bins=8)
        for emb in model.main_effects:
            assert emb.weight.shape == (9, 1)  # 8 bins + 1 unknown

    def test_wrong_feature_keys_raises(self):
        """Model raises if dataset has multiple feature keys."""
        from pyhealth.datasets import create_sample_dataset
        samples = [
            {"patient_id": "p0", "visit_id": "v0",
            "feat_a": [1.0, 2.0], "feat_b": [3.0], "label": 0},
            {"patient_id": "p1", "visit_id": "v1",
            "feat_a": [2.0, 3.0], "feat_b": [4.0], "label": 1},
        ]
        ds = create_sample_dataset(
            samples=samples,
            input_schema={"feat_a": "tensor", "feat_b": "tensor"},
            output_schema={"label": "binary"},
            in_memory=True,
        )
        with pytest.raises(AssertionError):
            GA2M(dataset=ds, n_bins=8)


# ---------------------------------------------------------------------------
# 2. Bin fitting
# ---------------------------------------------------------------------------

class TestBinFitting:

    def test_fit_bins_sets_flag(self, base_dataset):
        model = GA2M(dataset=base_dataset, n_bins=8)
        loader = _make_loader(base_dataset)
        assert not model._bins_fitted
        model.fit_bins(loader)
        assert model._bins_fitted

    def test_bin_edges_shape(self, base_dataset):
        """bin_edges has shape (input_dim, n_bins - 1)."""
        model = GA2M(dataset=base_dataset, n_bins=8)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)
        assert model.bin_edges.shape == (model.input_dim, 7)

    def test_bin_edges_monotone(self, base_dataset):
        """Quantile edges must be non-decreasing per feature."""
        model = GA2M(dataset=base_dataset, n_bins=8)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)
        edges = model.bin_edges.numpy()
        for d in range(model.input_dim):
            diffs = np.diff(edges[d])
            assert np.all(diffs >= 0), (
                f"Feature {d} bin edges are not monotone: {edges[d]}"
            )

    def test_forward_before_fit_bins_raises(self, base_dataset):
        """Calling forward() without fit_bins should raise RuntimeError."""
        model = GA2M(
            dataset=base_dataset,
            n_bins=8,
            use_interactions=False,
        )
        loader = _make_loader(base_dataset)
        batch = next(iter(loader))
        with pytest.raises(RuntimeError, match="fit_bins"):
            model(**batch)


# ---------------------------------------------------------------------------
# 3. Bin assignment
# ---------------------------------------------------------------------------

class TestBinAssignment:

    def test_known_values_in_range(self, base_dataset):
        """Known values map to bins 0..n_bins-1."""
        model = GA2M(dataset=base_dataset, n_bins=8)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)

        x = torch.tensor([[1.0, 2.0, 3.0, 1.5, 2.5, 0.5]])
        idx = model._assign_bins(x)
        assert idx.shape == (1, model.input_dim)
        assert (idx < model.n_bins).all(), (
            "Known values should not be routed to unknown bin."
        )

    def test_unknown_sentinel_routes_to_unknown_bin(self, base_dataset):
        """UNKNOWN_SENTINEL values must map to bin index n_bins."""
        model = GA2M(dataset=base_dataset, n_bins=8)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)

        x = torch.full((1, model.input_dim), 1.0)
        x[0, 0] = UNKNOWN_SENTINEL
        idx = model._assign_bins(x)
        assert idx[0, 0].item() == model.n_bins, (
            "UNKNOWN_SENTINEL should route to the dedicated unknown bin."
        )
        # Other features should stay in normal range.
        assert (idx[0, 1:] < model.n_bins).all()


# ---------------------------------------------------------------------------
# 4. Stage 1: main effects training
# ---------------------------------------------------------------------------

class TestMainEffectsTraining:

    def test_weights_change_after_training(self, base_dataset):
        """Main effect weights should differ from zeros after Stage 1."""
        model = GA2M(dataset=base_dataset, n_bins=8, use_interactions=False)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)

        # Record initial weights (all zero).
        initial = model.main_effects[0].weight.data.clone()
        model.fit_main_effects(loader, epochs=3, lr=1e-1)
        after = model.main_effects[0].weight.data

        assert not torch.allclose(initial, after), (
            "Main effect weights should update during Stage 1 training."
        )

    def test_forward_main_effects_only_output_shapes(self, base_dataset):
        """Internal main-effects-only forward returns correct shapes."""
        model = GA2M(dataset=base_dataset, n_bins=8, use_interactions=False)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)

        batch = next(iter(loader))
        out = model._forward_main_effects_only(batch)
        B = batch["features"].shape[0]
        assert out["logits"].shape == (B, 1)
        assert out["loss"].shape == ()  # scalar


# ---------------------------------------------------------------------------
# 5. Interaction selection
# ---------------------------------------------------------------------------

class TestInteractionSelection:

    def test_select_top_interactions_count(self, fitted_model):
        """Exactly top_k_interactions pairs are selected."""
        assert len(fitted_model.interaction_pairs) == fitted_model.top_k_interactions

    def test_interaction_module_dict_keys(self, fitted_model):
        """ModuleDict keys match selected pair tuples."""
        for i, j in fitted_model.interaction_pairs:
            assert f"{i}_{j}" in fitted_model.interactions

    def test_interaction_embedding_size(self, fitted_model):
        """Each interaction embedding has (n_bins+1)^2 rows."""
        expected = (fitted_model.n_bins + 1) ** 2
        for emb in fitted_model.interactions.values():
            assert emb.weight.shape[0] == expected

    def test_no_self_interactions(self, fitted_model):
        """No feature is paired with itself."""
        for i, j in fitted_model.interaction_pairs:
            assert i != j

    def test_pairs_are_upper_triangular(self, fitted_model):
        """Pairs are stored as (i, j) with i < j (no duplicates)."""
        for i, j in fitted_model.interaction_pairs:
            assert i < j

    def test_forward_raises_before_select(self, base_dataset):
        """Full forward should raise if select_top_interactions not called."""
        model = GA2M(dataset=base_dataset, n_bins=8, use_interactions=True)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)
        model.fit_main_effects(loader, epochs=1, lr=1e-2)
        # Intentionally skip select_top_interactions.
        batch = next(iter(loader))
        with pytest.raises(RuntimeError, match="select_top_interactions"):
            model(**batch)


# ---------------------------------------------------------------------------
# 6. Full forward pass
# ---------------------------------------------------------------------------

class TestFullForward:

    def test_output_keys(self, fitted_model, base_dataset):
        """Forward returns expected dict keys."""
        loader = _make_loader(base_dataset)
        batch = next(iter(loader))
        out = fitted_model(**batch)
        for key in ("loss", "y_prob", "y_true", "logits"):
            assert key in out, f"Missing key '{key}' in forward output."

    def test_output_shapes(self, fitted_model, base_dataset):
        """Logits and y_prob have shape (B, 1); loss is scalar."""
        loader = _make_loader(base_dataset)
        batch = next(iter(loader))
        B = batch["features"].shape[0]
        out = fitted_model(**batch)

        assert out["logits"].shape == (B, 1), (
            f"Expected logits shape ({B}, 1), got {out['logits'].shape}"
        )
        assert out["y_prob"].shape == (B, 1), (
            f"Expected y_prob shape ({B}, 1), got {out['y_prob'].shape}"
        )
        assert out["loss"].shape == (), "Loss should be a scalar tensor."

    def test_y_prob_in_unit_interval(self, fitted_model, base_dataset):
        """Predicted probabilities must lie in [0, 1]."""
        loader = _make_loader(base_dataset)
        batch = next(iter(loader))
        out = fitted_model(**batch)
        assert (out["y_prob"] >= 0.0).all() and (out["y_prob"] <= 1.0).all()

    def test_loss_is_positive(self, fitted_model, base_dataset):
        """BCE loss must be strictly positive."""
        loader = _make_loader(base_dataset)
        batch = next(iter(loader))
        out = fitted_model(**batch)
        assert out["loss"].item() > 0.0

    def test_backward_pass(self, fitted_model, base_dataset):
        """Gradients must flow through the full model."""
        loader = _make_loader(base_dataset)
        batch = next(iter(loader))
        out = fitted_model(**batch)
        out["loss"].backward()

        # Check that at least one main effect got a gradient.
        has_grad = any(
            emb.weight.grad is not None
            for emb in fitted_model.main_effects
        )
        assert has_grad, "No gradients flowed to main effect embeddings."

    def test_no_interactions_flag(self, base_dataset):
        """use_interactions=False runs without select_top_interactions."""
        model = GA2M(dataset=base_dataset, n_bins=8, use_interactions=False)
        loader = _make_loader(base_dataset)
        model.fit_bins(loader)
        model.fit_main_effects(loader, epochs=1)

        batch = next(iter(loader))
        out = model(**batch)
        B = batch["features"].shape[0]
        assert out["logits"].shape == (B, 1)
        assert out["loss"].item() > 0.0


# ---------------------------------------------------------------------------
# 7. Unknown bin handling
# ---------------------------------------------------------------------------

class TestUnknownBin:

    def test_unknown_values_produce_finite_output(self, unknown_dataset):
        """Samples with UNKNOWN_SENTINEL features should not produce NaN/Inf."""
        loader = _make_loader(unknown_dataset)
        model = GA2M(
            dataset=unknown_dataset,
            n_bins=8,
            top_k_interactions=2,
            use_interactions=True,
        )
        model.fit_bins(loader)
        model.fit_main_effects(loader, epochs=2)
        model.select_top_interactions()

        batch = next(iter(loader))
        out = model(**batch)
        assert torch.isfinite(out["logits"]).all(), (
            "Logits contain NaN or Inf for samples with unknown features."
        )

    def test_unknown_bin_weight_is_independent(self, unknown_dataset):
        """The unknown bin embedding is a separate learnable parameter."""
        model = GA2M(dataset=unknown_dataset, n_bins=8, use_interactions=False)
        # n_bins + 1 rows: last row is the unknown bin.
        assert model.main_effects[0].weight.shape[0] == model.n_bins + 1


# ---------------------------------------------------------------------------
# 8. Interpretability helpers
# ---------------------------------------------------------------------------

class TestInterpretabilityHelpers:

    def test_get_shape_function_lengths(self, fitted_model):
        """Shape function returns arrays of length n_bins."""
        midpoints, risks = fitted_model.get_shape_function(0)
        assert len(midpoints) == fitted_model.n_bins
        assert len(risks) == fitted_model.n_bins

    def test_get_shape_function_finite(self, fitted_model):
        """Shape function values must all be finite."""
        for d in range(fitted_model.input_dim):
            midpoints, risks = fitted_model.get_shape_function(d)
            assert np.all(np.isfinite(risks)), (
                f"Feature {d} shape function contains non-finite risk values."
            )

    def test_get_interaction_shape_selected(self, fitted_model):
        """get_interaction_shape returns a 2D array for selected pairs."""
        i, j = fitted_model.interaction_pairs[0]
        grid = fitted_model.get_interaction_shape(i, j)
        assert grid is not None
        assert grid.shape == (fitted_model.n_bins, fitted_model.n_bins)

    def test_get_interaction_shape_unselected_returns_none(self, fitted_model):
        """get_interaction_shape returns None for non-selected pairs."""
        # Find a pair definitely not selected.
        selected = set(fitted_model.interaction_pairs)
        for i in range(fitted_model.input_dim):
            for j in range(i + 1, fitted_model.input_dim):
                if (i, j) not in selected:
                    result = fitted_model.get_interaction_shape(i, j)
                    assert result is None
                    return  # one check is enough


# ---------------------------------------------------------------------------
# Entry point for running without pytest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running GA2M tests...")

    ds = _make_dataset(n_patients=4, n_features=6)
    loader = _make_loader(ds)

    model = GA2M(dataset=ds, n_bins=8, top_k_interactions=3)
    model.fit_bins(loader)
    model.fit_main_effects(loader, epochs=3, lr=1e-1)
    model.select_top_interactions()

    batch = next(iter(loader))
    out = model(**batch)
    out["loss"].backward()

    print(f"  loss    : {out['loss'].item():.4f}")
    print(f"  y_prob  : {out['y_prob'].squeeze().tolist()}")
    print(f"  y_true  : {out['y_true'].squeeze().tolist()}")
    print(f"  logits  : {out['logits'].squeeze().tolist()}")
    print(f"  selected pairs: {model.interaction_pairs}")

    midpoints, risks = model.get_shape_function(0)
    print(f"  feature 0 risk range: [{risks.min():.4f}, {risks.max():.4f}]")

    print("\nAll checks passed!")