"""Tests for the GRU-D model implementation.

All tests use synthetic/pseudo data — no real datasets required.
Each test completes in milliseconds using minimal tensor sizes
(4 patients, 3 timesteps, 2 variables).

References:
    Che, Z., Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018).
    Recurrent neural networks for multivariate time series with missing
    values. Scientific Reports, 8(1), 6085.
    https://doi.org/10.1038/s41598-018-24271-9

    Nestor, B., McDermott, M. B. A., Boag, W., Berner, G., Naumann, T.,
    Hughes, M. C., Goldenberg, A., & Ghassemi, M. (2019). Feature
    robustness in non-stationary health records: Caveats to deployable
    model performance in common clinical machine learning tasks.
    arXiv:1908.00690. https://arxiv.org/abs/1908.00690

Run with:
    python -m unittest tests/core/test_grud.py -v
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import GRUD
from pyhealth.models.grud import FilterLinear, GRUDLayer

np.random.seed(42)
torch.manual_seed(42)

# ── Constants — keep tiny for millisecond test times ─────────────────────────
N_VARS     = 2   # number of clinical variables
SEQ_LEN    = 3   # number of hourly timesteps
N_PATIENTS = 4   # 2-5 patients


# ── Synthetic data helpers ────────────────────────────────────────────────────

def make_interleaved(
    seq_len: int = SEQ_LEN,
    n_vars: int = N_VARS,
    seed: int = 0,
) -> list:
    """Creates a synthetic interleaved feature list.

    Channels are ordered as (mask, mean, time_since_measured) per
    variable, matching the format produced by the simple imputer
    pipeline in processing.py.

    Args:
        seq_len: Number of timesteps.
        n_vars: Number of clinical variables.
        seed: Random seed for reproducibility.

    Returns:
        Nested list of shape ``(seq_len, n_vars * 3)``.
    """
    rng = np.random.RandomState(seed)
    mask  = rng.randint(0, 2, (seq_len, n_vars)).astype(np.float32)
    mean  = rng.randn(seq_len, n_vars).astype(np.float32)
    delta = (rng.rand(seq_len, n_vars) * 3).astype(np.float32)
    interleaved = np.empty((seq_len, n_vars * 3), dtype=np.float32)
    interleaved[:, 0::3] = mask
    interleaved[:, 1::3] = mean
    interleaved[:, 2::3] = delta
    return interleaved.tolist()


def make_synthetic_dataset(
    n_patients: int = N_PATIENTS,
    n_vars: int = N_VARS,
    seq_len: int = SEQ_LEN,
) -> SampleDataset:
    """Creates a minimal SampleDataset with synthetic binary labels.

    Generates interleaved (mask, mean, time_since_measured) channel
    tensors for each patient, matching the format produced by the
    simple imputer pipeline in processing.py.

    Args:
        n_patients: Number of patient samples (2-5 for fast tests).
        n_vars: Number of clinical variables per timestep.
        seq_len: Number of hourly timesteps per stay.

    Returns:
        A :class:`~pyhealth.datasets.SampleDataset` containing
        synthetic ICU stay samples with binary mortality labels.
    """
    samples = [
        {
            "patient_id":  f"synth_patient_{i}",
            "visit_id":    f"synth_visit_{i}",
            "time_series": make_interleaved(seq_len, n_vars, seed=i),
            "label":       i % 2,
        }
        for i in range(n_patients)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"time_series": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="synthetic_grud_test",
    )


def make_model(
    dataset: SampleDataset,
    hidden_size: int = 8,
    dropout: float = 0.0,
) -> GRUD:
    """Instantiates a minimal GRUD model for testing.

    Uses small hidden size and zero dropout to keep tests fast
    and deterministic.

    Args:
        dataset: A :class:`~pyhealth.datasets.SampleDataset` used
            to infer feature keys, label keys, and x_mean.
        hidden_size: GRU-D hidden state size. Default is ``8``.
        dropout: Dropout probability. Default is ``0.0``.

    Returns:
        An initialised :class:`~pyhealth.models.grud.GRUD` model.
    """
    return GRUD(dataset=dataset, hidden_size=hidden_size, dropout=dropout)


# ── FilterLinear tests ────────────────────────────────────────────────────────

class TestFilterLinear(unittest.TestCase):
    """Tests for the diagonal FilterLinear layer.

    FilterLinear implements the input decay weight matrix (Wgamma_x)
    from Che et al. (2018) as a diagonal structure, ensuring each
    feature's decay rate is learned independently.
    """

    def test_output_shape(self):
        """Output shape matches (batch_size, out_features)."""
        layer = FilterLinear(4, 4, torch.eye(4))
        self.assertEqual(layer(torch.randn(3, 4)).shape, (3, 4))

    def test_diagonal_filter_zeros_off_diagonal(self):
        """Off-diagonal weights are zeroed by the identity filter."""
        layer = FilterLinear(3, 3, torch.eye(3), bias=False)
        with torch.no_grad():
            layer.weight.fill_(1.0)
        out = layer(torch.ones(1, 3))
        self.assertTrue(torch.allclose(out, torch.ones(1, 3)))

    def test_no_bias_option(self):
        """Bias parameter is None when bias=False."""
        self.assertIsNone(FilterLinear(3, 3, torch.eye(3), bias=False).bias)

    def test_filter_matrix_not_learnable(self):
        """Filter matrix is a registered buffer, not a learnable parameter.

        Verifies the register_buffer() change — the matrix must not appear
        in model.parameters() and must appear in model.named_buffers().
        requires_grad=False alone would pass for the old nn.Parameter form.
        """
        layer = FilterLinear(3, 3, torch.eye(3))
        self.assertFalse(layer.filter_square_matrix.requires_grad)
        # Must be a buffer, not a parameter
        buffer_names = dict(layer.named_buffers()).keys()
        param_names  = dict(layer.named_parameters()).keys()
        self.assertIn("filter_square_matrix", buffer_names)
        self.assertNotIn("filter_square_matrix", param_names)

    def test_weight_gradient_flows(self):
        """Gradients reach the weight matrix via backward."""
        layer = FilterLinear(3, 3, torch.eye(3))
        layer(torch.randn(2, 3)).sum().backward()
        self.assertIsNotNone(layer.weight.grad)

    def test_repr_contains_class_name(self):
        """__repr__ includes FilterLinear and dimension info."""
        r = repr(FilterLinear(4, 4, torch.eye(4)))
        self.assertIn("FilterLinear", r)
        self.assertIn("in_features=4", r)


# ── GRUDLayer tests ───────────────────────────────────────────────────────────

class TestGRUDLayer(unittest.TestCase):
    """Tests for the GRUDLayer recurrent cell.

    GRUDLayer implements the core GRU-D recurrent step including
    input decay (gamma_x) and hidden state decay (gamma_h) as
    described in Che et al. (2018).
    """

    def setUp(self):
        """Creates a minimal GRUDLayer and synthetic batch tensors."""
        self.layer = GRUDLayer(
            input_size=N_VARS,
            hidden_size=8,
            x_mean=torch.zeros(1, 1, N_VARS),
        )
        B = 2
        self.x      = torch.randn(B, SEQ_LEN, N_VARS)
        self.x_last = torch.randn(B, SEQ_LEN, N_VARS)
        self.mask   = torch.randint(0, 2, (B, SEQ_LEN, N_VARS)).float()
        self.delta  = torch.rand(B, SEQ_LEN, N_VARS) * 3

    def test_output_shape(self):
        """Final hidden state shape is (batch_size, hidden_size)."""
        out = self.layer(self.x, self.x_last, self.mask, self.delta)
        self.assertEqual(out.shape, (2, 8))

    def test_deterministic_output(self):
        """Same input always produces the same output."""
        out1 = self.layer(self.x, self.x_last, self.mask, self.delta)
        out2 = self.layer(self.x, self.x_last, self.mask, self.delta)
        self.assertTrue(torch.allclose(out1, out2))

    def test_observed_vs_missing_differ(self):
        """Fully observed and fully missing masks produce different outputs."""
        out_obs = self.layer(
            self.x, self.x_last,
            torch.ones(2, SEQ_LEN, N_VARS), self.delta,
        )
        out_mis = self.layer(
            self.x, self.x_last,
            torch.zeros(2, SEQ_LEN, N_VARS), self.delta,
        )
        self.assertFalse(torch.allclose(out_obs, out_mis))

    def test_gradient_flows(self):
        """Gradients propagate back to the input tensor."""
        x = self.x.requires_grad_(True)
        self.layer(x, self.x_last, self.mask, self.delta).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_x_mean_is_buffer(self):
        """x_mean is a registered buffer (not a learnable parameter)."""
        self.assertIn("x_mean", dict(self.layer.named_buffers()))

    def test_zero_delta_produces_finite_output(self):
        """Zero elapsed time produces finite output (no numerical issues)."""
        out = self.layer(
            self.x, self.x_last, self.mask,
            torch.zeros_like(self.x),
        )
        self.assertTrue(torch.isfinite(out).all())


# ── prepare_input / _split_channels tests ────────────────────────────────────

class TestPrepareInput(unittest.TestCase):
    """Tests for GRUD.prepare_input and its inverse _split_channels.

    Both methods are static utilities on GRUD that handle the interleaved
    channel format. prepare_input converts separate tensors into interleaved
    format; _split_channels is its exact inverse. Tests here cover both
    methods together since they form a round-trip pair.
    """

    def setUp(self):
        self.values = torch.randn(4, SEQ_LEN, N_VARS)
        self.mask   = torch.randint(0, 2, (4, SEQ_LEN, N_VARS)).float()
        self.delta  = torch.rand(4, SEQ_LEN, N_VARS) * 5

    # ── _split_channels ───────────────────────────────────────────────────────

    def test_split_channels_shape(self):
        """_split_channels returns correct shapes for each channel."""
        x = torch.randn(3, SEQ_LEN, N_VARS * 3)
        mask, mean, delta = GRUD._split_channels(x)
        for t in (mask, mean, delta):
            self.assertEqual(t.shape, (3, SEQ_LEN, N_VARS))

    def test_split_channels_index_correctness(self):
        """_split_channels extracts mask=0, mean=1, delta=2 per triplet."""
        x2 = torch.tensor([[[1., 2., 3., 4., 5., 6.]]])
        mask2, mean2, delta2 = GRUD._split_channels(x2)
        self.assertAlmostEqual(mask2[0, 0, 0].item(),  1., places=5)
        self.assertAlmostEqual(mean2[0, 0, 0].item(),  2., places=5)
        self.assertAlmostEqual(delta2[0, 0, 0].item(), 3., places=5)
        self.assertAlmostEqual(mask2[0, 0, 1].item(),  4., places=5)
        self.assertAlmostEqual(mean2[0, 0, 1].item(),  5., places=5)
        self.assertAlmostEqual(delta2[0, 0, 1].item(), 6., places=5)

    # ── prepare_input ─────────────────────────────────────────────────────────

    def test_output_shape(self):
        """prepare_input produces shape (..., n_vars * 3)."""
        x = GRUD.prepare_input(self.values, self.mask, self.delta)
        self.assertEqual(x.shape, (4, SEQ_LEN, N_VARS * 3))

    def test_channel_order(self):
        """Interleaved order is mask=0, mean=1, delta=2 per triplet."""
        x = GRUD.prepare_input(self.values, self.mask, self.delta)
        self.assertTrue(torch.equal(x[..., 0::3], self.mask))
        self.assertTrue(torch.equal(x[..., 1::3], self.values))
        self.assertTrue(torch.equal(x[..., 2::3], self.delta))

    def test_round_trip_with_split_channels(self):
        """prepare_input and _split_channels are exact inverses."""
        x = GRUD.prepare_input(self.values, self.mask, self.delta)
        mask2, values2, delta2 = GRUD._split_channels(x)
        self.assertTrue(torch.equal(mask2,   self.mask))
        self.assertTrue(torch.equal(values2, self.values))
        self.assertTrue(torch.equal(delta2,  self.delta))

    def test_single_sample_no_batch_dim(self):
        """prepare_input works for a single sample without batch dimension."""
        values = torch.randn(SEQ_LEN, N_VARS)
        mask   = torch.ones(SEQ_LEN, N_VARS)
        delta  = torch.zeros(SEQ_LEN, N_VARS)
        x = GRUD.prepare_input(values, mask, delta)
        self.assertEqual(x.shape, (SEQ_LEN, N_VARS * 3))

    def test_device_preserved(self):
        """Output tensor is on the same device as the inputs."""
        x = GRUD.prepare_input(self.values, self.mask, self.delta)
        self.assertEqual(x.device, self.values.device)

    def test_dtype_preserved(self):
        """Output tensor has the same dtype as the inputs."""
        x = GRUD.prepare_input(self.values, self.mask, self.delta)
        self.assertEqual(x.dtype, self.values.dtype)


# ── GRUD model integration tests ──────────────────────────────────────────────

class TestGRUD(unittest.TestCase):
    """Integration tests for the GRUD PyHealth model.

    Uses 4 synthetic patients with 3 timesteps and 2 variables.
    All tests complete in milliseconds. Covers instantiation, forward
    pass output shapes, gradient computation, and model persistence.
    """

    def setUp(self):
        """Creates a synthetic dataset, model, and batch for testing."""
        self.tmp_dir = tempfile.mkdtemp()
        self.dataset = make_synthetic_dataset()
        self.model   = make_model(self.dataset)
        loader       = get_dataloader(
            self.dataset, batch_size=N_PATIENTS, shuffle=False
        )
        self.batch = next(iter(loader))

    def tearDown(self):
        """Removes the temporary directory after each test."""
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ── Initialisation ────────────────────────────────────────────────────────

    def test_model_initialization(self):
        """Model initialises correctly with expected attributes."""
        from pyhealth.models import BaseModel
        self.assertIsInstance(self.model, BaseModel)
        self.assertIsInstance(self.model, torch.nn.Module)
        self.assertEqual(self.model.input_size, N_VARS)
        self.assertEqual(self.model.hidden_size, 8)
        self.assertIn("time_series", self.model.grud_layers)
        self.assertIsInstance(
            self.model.grud_layers["time_series"], GRUDLayer
        )
        self.assertEqual(self.model.fc.out_features, 1)

    def test_hidden_size_stored(self):
        """hidden_size attribute is correctly stored."""
        model = make_model(self.dataset, hidden_size=16)
        self.assertEqual(model.hidden_size, 16)

    def test_invalid_channels_raises_value_error(self):
        """Non-divisible-by-3 channel count raises ValueError."""
        bad_ds = create_sample_dataset(
            samples=[
                {
                    "patient_id":  "p0",
                    "visit_id":    "v0",
                    "time_series": [[1.0, 2.0]],
                    "label":       0,
                },
                {
                    "patient_id":  "p1",
                    "visit_id":    "v1",
                    "time_series": [[1.0, 2.0]],
                    "label":       1,
                },
            ],
            input_schema={"time_series": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="bad_test",
        )
        with self.assertRaises(ValueError):
            GRUD(dataset=bad_ds)

    def test_x_mean_shape(self):
        """x_mean buffer shape is (1, 1, input_size) — global mean over samples and time."""
        x_mean = self.model.grud_layers["time_series"].x_mean
        self.assertEqual(x_mean.shape, (1, 1, N_VARS))

    # ── Forward pass ──────────────────────────────────────────────────────────

    def test_model_forward(self):
        """Forward pass returns correct output keys and shapes."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(**self.batch)
        self.assertIn("loss",   out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit",  out)
        self.assertEqual(out["loss"].ndim, 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertEqual(out["y_prob"].shape, (N_PATIENTS, 1))
        self.assertEqual(out["y_true"].shape[0], N_PATIENTS)
        self.assertEqual(out["logit"].shape, out["y_prob"].shape)
        self.assertTrue((out["y_prob"] >= 0).all())
        self.assertTrue((out["y_prob"] <= 1).all())

    # ── Gradient computation ──────────────────────────────────────────────────

    def test_model_backward(self):
        """Backward pass updates parameters with no NaN gradients."""
        opt    = torch.optim.Adam(self.model.parameters(), lr=0.01)
        before = {n: p.clone() for n, p in self.model.named_parameters()}
        self.model(**self.batch)["loss"].backward()
        opt.step()
        changed = any(
            not torch.equal(p, before[n])
            for n, p in self.model.named_parameters()
            if p.requires_grad
        )
        self.assertTrue(changed, "No parameters updated after backward pass")
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                self.assertFalse(
                    torch.isnan(param.grad).any(),
                    msg=f"NaN gradient in {name}",
                )

    # ── Model persistence ─────────────────────────────────────────────────────

    def test_model_save_and_load(self):
        """Model state can be saved and reloaded from a temp directory."""
        save_path = os.path.join(self.tmp_dir, "grud.pt")
        torch.save(self.model.state_dict(), save_path)
        self.assertTrue(os.path.exists(save_path))

        model2 = make_model(self.dataset, hidden_size=8)
        model2.load_state_dict(
            torch.load(save_path, weights_only=True)
        )
        self.model.eval()
        model2.eval()
        with torch.no_grad():
            out1 = self.model(**self.batch)
            out2 = model2(**self.batch)
        self.assertTrue(
            torch.allclose(out1["y_prob"], out2["y_prob"])
        )

    # ── Multiple feature keys ─────────────────────────────────────────────────

    def test_multiple_feature_keys(self):
        """GRUD concatenates embeddings from multiple feature keys."""
        samples = [
            {
                "patient_id":    f"p{i}",
                "visit_id":      f"v{i}",
                "time_series":   make_interleaved(seed=i),
                "time_series_2": make_interleaved(seed=i + 10),
                "label":         i % 2,
            }
            for i in range(N_PATIENTS)
        ]
        ds = create_sample_dataset(
            samples=samples,
            input_schema={
                "time_series":   "tensor",
                "time_series_2": "tensor",
            },
            output_schema={"label": "binary"},
            dataset_name="multi_key",
        )
        model = GRUD(dataset=ds, hidden_size=8, dropout=0.0)
        self.assertEqual(len(model.grud_layers), 2)
        self.assertEqual(model.fc.in_features, 16)

        loader = get_dataloader(ds, batch_size=N_PATIENTS, shuffle=False)
        model.eval()
        with torch.no_grad():
            out = model(**next(iter(loader)))
        self.assertTrue(torch.isfinite(out["loss"]))

    # ── Hidden size variants ──────────────────────────────────────────────────

    def test_hidden_size_4(self):
        """Model runs correctly with hidden_size=4."""
        self._check_hidden_size(4)

    def test_hidden_size_8(self):
        """Model runs correctly with hidden_size=8."""
        self._check_hidden_size(8)

    def test_hidden_size_16(self):
        """Model runs correctly with hidden_size=16."""
        self._check_hidden_size(16)

    def _check_hidden_size(self, hidden_size: int) -> None:
        """Helper that verifies a given hidden_size produces valid output."""
        ds     = make_synthetic_dataset()
        model  = make_model(ds, hidden_size=hidden_size)
        loader = get_dataloader(ds, batch_size=N_PATIENTS, shuffle=False)
        model.eval()
        with torch.no_grad():
            out = model(**next(iter(loader)))
        self.assertEqual(out["y_prob"].shape, (N_PATIENTS, 1))
        self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()