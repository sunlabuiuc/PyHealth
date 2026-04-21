"""Unit tests for :class:`pyhealth.models.BulkRNABert`."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.models import (
    BulkRNABert,
    BulkRNABertConfig,
    bin_expression_values,
    compute_normalization_factor,
    load_expression_csv,
)


def _import_mlm_example():
    """Import the MLM example module so we can test its helpers directly."""
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "tcga_rnaseq_mlm_bulk_rna_bert.py"
    )
    spec = importlib.util.spec_from_file_location(
        "tcga_rnaseq_mlm_bulk_rna_bert", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tcga_rnaseq_mlm_bulk_rna_bert"] = module
    spec.loader.exec_module(module)
    return module


def _small_config(expression_mode: str = "discrete") -> BulkRNABertConfig:
    return BulkRNABertConfig(
        n_genes=32,
        n_bins=8,
        embed_dim=16,
        num_layers=2,
        num_heads=4,
        ffn_embed_dim=32,
        init_gene_embed_dim=16,
        expression_mode=expression_mode,
        continuous_hidden_dim=16 if expression_mode == "continuous" else None,
    )


class TestBulkRNABertDiscrete(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_config("discrete")
        self.model = BulkRNABert(
            dataset=None, config=self.cfg, feature_key="expression"
        )

    def test_mode(self):
        self.assertEqual(self.model.mode, "multiclass")

    def test_forward_train_shapes(self):
        self.model.train()
        tokens = torch.randint(0, self.cfg.n_bins, (4, self.cfg.n_genes))
        out = self.model(expression=tokens)
        self.assertIn("loss", out)
        self.assertEqual(out["loss"].dim(), 0)
        self.assertEqual(
            out["logits"].shape, (4, self.cfg.n_genes, self.cfg.n_bins)
        )
        # y_prob / y_true gathered at masked positions
        self.assertEqual(out["y_prob"].dim(), 2)
        self.assertEqual(out["y_prob"].shape[1], self.cfg.n_bins)
        self.assertEqual(out["y_true"].dim(), 1)
        self.assertEqual(out["y_prob"].shape[0], out["y_true"].shape[0])

    def test_backward(self):
        self.model.train()
        tokens = torch.randint(0, self.cfg.n_bins, (4, self.cfg.n_genes))
        out = self.model(expression=tokens)
        out["loss"].backward()
        grads = [
            p.grad for p in self.model.parameters() if p.requires_grad
        ]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_eval_no_mask(self):
        self.model.eval()
        tokens = torch.randint(0, self.cfg.n_bins, (2, self.cfg.n_genes))
        out = self.model(expression=tokens)
        self.assertEqual(float(out["loss"]), 0.0)
        self.assertEqual(
            out["y_prob"].shape, (2 * self.cfg.n_genes, self.cfg.n_bins)
        )

    def test_masking_ratio(self):
        """Empirical mask ratio should be close to mlm_probability."""
        torch.manual_seed(0)
        self.model.train()
        big_tokens = torch.zeros(
            (8, self.cfg.n_genes), dtype=torch.long
        )
        _, mask_positions = self.model._apply_mask_discrete(big_tokens)
        ratio = mask_positions.float().mean().item()
        self.assertAlmostEqual(ratio, self.cfg.mlm_probability, delta=0.05)

    def test_sequence_length_mismatch(self):
        self.model.train()
        tokens = torch.randint(0, self.cfg.n_bins, (2, self.cfg.n_genes - 1))
        with self.assertRaises(ValueError):
            self.model(expression=tokens)


class TestBulkRNABertContinuous(unittest.TestCase):
    def setUp(self):
        self.cfg = _small_config("continuous")
        self.model = BulkRNABert(
            dataset=None, config=self.cfg, feature_key="expression"
        )

    def test_mode(self):
        self.assertEqual(self.model.mode, "regression")

    def test_forward_train_shapes(self):
        self.model.train()
        values = torch.rand(4, self.cfg.n_genes) * 5.0
        out = self.model(expression=values)
        self.assertIn("loss", out)
        self.assertEqual(out["loss"].dim(), 0)
        self.assertEqual(out["predictions"].shape, (4, self.cfg.n_genes))
        self.assertEqual(out["y_prob"].dim(), 1)
        self.assertEqual(out["y_prob"].shape, out["y_true"].shape)

    def test_backward(self):
        self.model.train()
        values = torch.rand(4, self.cfg.n_genes) * 5.0
        out = self.model(expression=values)
        out["loss"].backward()
        grads = [
            p.grad for p in self.model.parameters() if p.requires_grad
        ]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_eval_no_mask(self):
        self.model.eval()
        values = torch.rand(2, self.cfg.n_genes)
        out = self.model(expression=values)
        self.assertEqual(float(out["loss"]), 0.0)
        self.assertEqual(out["predictions"].shape, (2, self.cfg.n_genes))


class TestTrainingLoop(unittest.TestCase):
    """Integration test: a few manual optimizer steps should decrease loss."""

    def _run_loop(self, expression_mode: str):
        torch.manual_seed(0)
        cfg = _small_config(expression_mode)
        model = BulkRNABert(
            dataset=None, config=cfg, feature_key="expression"
        )
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        model.train()

        if expression_mode == "discrete":
            data = torch.randint(0, cfg.n_bins, (16, cfg.n_genes))
        else:
            data = torch.rand(16, cfg.n_genes) * 5.0

        first_losses, last_losses = [], []
        for step in range(20):
            out = model(expression=data)
            optim.zero_grad()
            out["loss"].backward()
            optim.step()
            if step < 3:
                first_losses.append(float(out["loss"]))
            if step >= 17:
                last_losses.append(float(out["loss"]))
        # loss should decrease on this tiny fixed batch
        self.assertLess(
            sum(last_losses) / len(last_losses),
            sum(first_losses) / len(first_losses),
        )

    def test_discrete_loop(self):
        self._run_loop("discrete")

    def test_continuous_loop(self):
        self._run_loop("continuous")


class TestBinExpressionValues(unittest.TestCase):
    def test_shape_and_range(self):
        vals = np.random.rand(4, 20) * 5.0
        bins = bin_expression_values(vals, n_bins=16)
        self.assertEqual(bins.shape, (4, 20))
        self.assertTrue(bins.min() >= 0)
        self.assertTrue(bins.max() < 16)

    def test_zero_values_go_to_bin_zero(self):
        vals = np.zeros((2, 5))
        bins = bin_expression_values(vals, n_bins=8)
        self.assertTrue(torch.equal(bins, torch.zeros(2, 5, dtype=torch.long)))

    def test_raw_tpm_path(self):
        tpm = np.array([[0.0, 1.0, 10.0, 100.0]])
        bins_log = bin_expression_values(
            np.log10(tpm + 1.0), n_bins=8, already_log_normalized=True
        )
        bins_raw = bin_expression_values(
            tpm, n_bins=8, already_log_normalized=False
        )
        self.assertTrue(torch.equal(bins_log, bins_raw))

    def test_monotonic(self):
        vals = np.linspace(0.0, 5.5, 50).reshape(1, -1)
        bins = bin_expression_values(vals, n_bins=16).squeeze(0)
        # non-decreasing
        self.assertTrue(torch.all(bins[1:] >= bins[:-1]))


class TestAutocastDtypeConfig(unittest.TestCase):
    def test_invalid_value_rejected(self):
        with self.assertRaises(ValueError):
            BulkRNABertConfig(
                n_genes=8, n_bins=4, embed_dim=8, num_layers=1,
                num_heads=2, ffn_embed_dim=16, init_gene_embed_dim=8,
                autocast_dtype="float64",
            )

    def test_cpu_autocast_is_noop(self):
        """autocast_dtype is ignored on CPU — forward still works normally."""
        cfg = BulkRNABertConfig(
            n_genes=8, n_bins=4, embed_dim=8, num_layers=1,
            num_heads=2, ffn_embed_dim=16, init_gene_embed_dim=8,
            autocast_dtype="bfloat16",
        )
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        self.assertFalse(model._autocast_enabled())  # CPU -> disabled
        tokens = torch.randint(0, cfg.n_bins, (2, cfg.n_genes))
        model.train()
        out = model(expression=tokens)
        self.assertIn("loss", out)


class TestLoadExpressionCSV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "expr.csv"
        self.n_genes = 10
        self.n_samples = 3
        rng = np.random.default_rng(0)
        tpm = rng.uniform(0.0, 100.0, size=(self.n_samples, self.n_genes))
        cols = [f"ENSG{i:011d}" for i in range(self.n_genes)]
        df = pd.DataFrame(tpm, columns=cols)
        df["identifier"] = [f"sample-{i}" for i in range(self.n_samples)]
        df.to_csv(self.path, index=False)
        self.cols = cols
        self.tpm = tpm

    def tearDown(self):
        self.tmp.cleanup()

    def test_continuous_shape_and_log10(self):
        tensor, genes = load_expression_csv(self.path, mode="continuous")
        self.assertEqual(genes, self.cols)
        self.assertEqual(tensor.shape, (self.n_samples, self.n_genes))
        self.assertEqual(tensor.dtype, torch.float32)
        # Continuous mode applies the same normalization_factor as discrete
        # tokenization so the model sees inputs on a consistent scale; see
        # reference dataloader.py :: load_continuous.
        norm = 5.547176906585117
        expected = (np.log10(self.tpm + 1.0) / norm).astype(np.float32)
        np.testing.assert_allclose(tensor.numpy(), expected, rtol=1e-5)

    def test_discrete_returns_long_tokens(self):
        tokens, genes = load_expression_csv(
            self.path, mode="discrete", n_bins=64
        )
        self.assertEqual(genes, self.cols)
        self.assertEqual(tokens.shape, (self.n_samples, self.n_genes))
        self.assertEqual(tokens.dtype, torch.long)
        self.assertGreaterEqual(tokens.min().item(), 0)
        self.assertLess(tokens.max().item(), 64)

    def test_identifier_column_dropped(self):
        _, genes = load_expression_csv(self.path, mode="continuous")
        self.assertNotIn("identifier", genes)

    def test_non_numeric_residual_raises(self):
        bad = Path(self.tmp.name) / "bad.csv"
        df = pd.DataFrame(
            {"g0": [1.0, 2.0], "g1": [3.0, 4.0], "cohort": ["A", "B"]}
        )
        df.to_csv(bad, index=False)
        with self.assertRaises(ValueError):
            load_expression_csv(bad, mode="continuous", drop_columns=())

    def test_already_log_normalized_skips_log(self):
        logged = np.log10(self.tpm + 1.0)
        df = pd.DataFrame(logged, columns=self.cols)
        logged_path = Path(self.tmp.name) / "logged.csv"
        df.to_csv(logged_path, index=False)
        tensor, _ = load_expression_csv(
            logged_path, mode="continuous", already_log_normalized=True
        )
        norm = 5.547176906585117
        np.testing.assert_allclose(
            tensor.numpy(), (logged / norm).astype(np.float32), rtol=1e-5
        )

    def test_feeds_into_model(self):
        cfg = BulkRNABertConfig(
            n_genes=self.n_genes,
            n_bins=8,
            embed_dim=16,
            num_layers=1,
            num_heads=4,
            ffn_embed_dim=32,
            init_gene_embed_dim=16,
            expression_mode="continuous",
            continuous_hidden_dim=16,
        )
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        tensor, _ = load_expression_csv(self.path, mode="continuous")
        model.train()
        out = model(expression=tensor)
        self.assertEqual(out["predictions"].shape, tensor.shape)


class TestComputeNormalizationFactor(unittest.TestCase):
    """Pins the contract of the helper that re-derives ``normalization_factor``.

    The helper is not used by the pipeline; it exists so users can compute
    the value for their own corpus and independently reproduce the hardcoded
    default against the reference TCGA pre-training CSV.
    """

    def test_matches_max_log10_formula_float64(self):
        """Helper returns exactly ``max(log10(TPM + 1))`` in float64, both
        for raw-TPM CSVs (the default path) and for already-log-normalized
        CSVs (``already_log_normalized=True`` skips the log step)."""
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            tpm = rng.uniform(0.0, 1e5, size=(50, 20))
            tpm[7, 13] = 3.52513e5  # inject a known maximum
            cols = [f"ENSG{i:011d}" for i in range(tpm.shape[1])]

            # Raw-TPM CSV with an identifier metadata column.
            raw_path = Path(tmp) / "raw.csv"
            df = pd.DataFrame(tpm, columns=cols)
            df["identifier"] = [f"sample-{i}" for i in range(len(df))]
            df.to_csv(raw_path, index=False)
            self.assertEqual(
                compute_normalization_factor(raw_path),
                float(np.log10(tpm + 1.0).max()),
            )

            # Same data but pre-logged; flag must skip the log step.
            logged = np.log10(tpm + 1.0)
            log_path = Path(tmp) / "logged.csv"
            pd.DataFrame(logged, columns=cols).to_csv(log_path, index=False)
            self.assertEqual(
                compute_normalization_factor(
                    log_path, already_log_normalized=True
                ),
                float(logged.max()),
            )


class TestLoadGeneEmbeddingFromPt(unittest.TestCase):
    """Tests for the MLM example's PyTorch-native gene-embedding loader.

    Includes a JAX→PT parity check that mirrors what
    ``scripts/convert_jax_pretrain_to_pyhealth.py`` does for the
    ``gene_embedding`` tensors: the JAX linear ``w`` is stored in (in, out)
    layout while ``torch.nn.Linear.weight`` is (out, in), so the converter
    transposes before saving. This test pins that contract end-to-end —
    starting JAX arrays must equal the model's loaded weights after a
    round-trip through a real ``.pt`` file on disk.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pt_path = Path(self.tmp.name) / "gene_embedding.pt"
        self.n_genes = 32
        self.init_dim = 20
        self.embed_dim = 16
        self.cfg = BulkRNABertConfig(
            n_genes=self.n_genes,
            n_bins=8,
            embed_dim=self.embed_dim,
            num_layers=2,
            num_heads=4,
            ffn_embed_dim=32,
            init_gene_embed_dim=self.init_dim,
        )
        self.mlm = _import_mlm_example()

    def tearDown(self):
        self.tmp.cleanup()

    def _fresh_model(self) -> BulkRNABert:
        return BulkRNABert(dataset=None, config=self.cfg, feature_key="expression")

    def test_load_from_pt_matches_values(self):
        """Tensors on disk should be copied byte-for-byte into the model."""
        rng = np.random.default_rng(0)
        ge = torch.from_numpy(
            rng.standard_normal((self.n_genes, self.init_dim)).astype(np.float32)
        )
        pw = torch.from_numpy(
            rng.standard_normal((self.embed_dim, self.init_dim)).astype(np.float32)
        )
        pb = torch.from_numpy(
            rng.standard_normal((self.embed_dim,)).astype(np.float32)
        )
        torch.save(
            {
                "gene_embedding.embed.weight": ge,
                "gene_embedding.proj.weight": pw,
                "gene_embedding.proj.bias": pb,
            },
            self.pt_path,
        )
        model = self._fresh_model()
        self.mlm._load_gene_embedding_from_pt(model, self.pt_path)
        self.assertTrue(torch.equal(model.gene_embedding.embed.weight.data, ge))
        self.assertTrue(torch.equal(model.gene_embedding.proj.weight.data, pw))
        self.assertTrue(torch.equal(model.gene_embedding.proj.bias.data, pb))

    def test_extra_keys_ignored(self):
        """A full model state_dict with attention keys must load cleanly."""
        rng = np.random.default_rng(1)
        ge = torch.from_numpy(
            rng.standard_normal((self.n_genes, self.init_dim)).astype(np.float32)
        )
        pw = torch.from_numpy(
            rng.standard_normal((self.embed_dim, self.init_dim)).astype(np.float32)
        )
        pb = torch.from_numpy(
            rng.standard_normal((self.embed_dim,)).astype(np.float32)
        )
        torch.save(
            {
                "gene_embedding.embed.weight": ge,
                "gene_embedding.proj.weight": pw,
                "gene_embedding.proj.bias": pb,
                "layers.0.attention.q_proj.weight": torch.zeros(4, 4),
                "lm_head.weight": torch.zeros(8, 16),
            },
            self.pt_path,
        )
        model = self._fresh_model()
        self.mlm._load_gene_embedding_from_pt(model, self.pt_path)
        self.assertTrue(torch.equal(model.gene_embedding.proj.weight.data, pw))

    def test_jax_to_pt_parity(self):
        """Starting JAX arrays must equal model weights after a round-trip file save.

        Mimics ``scripts/convert_jax_pretrain_to_pyhealth.py``: takes JAX-shaped
        numpy arrays, applies the (in, out) → (out, in) transpose for the
        linear layer, saves to a ``.pt`` file, and loads via the MLM example's
        loader. The final model weights must match the JAX inputs exactly, and
        the model's ``gene_embedding()`` forward must reproduce the expected
        ``JAX_ge @ JAX_w + JAX_b`` computation.
        """
        rng = np.random.default_rng(42)
        # JAX / Haiku naming: flat dict keyed by module path.
        jax_ge = rng.standard_normal(
            (self.n_genes, self.init_dim)
        ).astype(np.float32)
        jax_lw = rng.standard_normal(
            (self.init_dim, self.embed_dim)
        ).astype(np.float32)  # JAX stores (in, out)
        jax_lb = rng.standard_normal((self.embed_dim,)).astype(np.float32)

        # The conversion the script performs for the gene_embedding subset:
        pt_state = {
            "gene_embedding.embed.weight": torch.from_numpy(jax_ge.copy()),
            "gene_embedding.proj.weight": torch.from_numpy(jax_lw.T.copy()),
            "gene_embedding.proj.bias": torch.from_numpy(jax_lb.copy()),
        }
        torch.save(pt_state, self.pt_path)

        model = self._fresh_model()
        self.mlm._load_gene_embedding_from_pt(model, self.pt_path)

        # (1) Tensor-level parity vs the starting JAX arrays.
        np.testing.assert_array_equal(
            model.gene_embedding.embed.weight.detach().numpy(), jax_ge
        )
        np.testing.assert_array_equal(
            model.gene_embedding.proj.weight.detach().numpy(), jax_lw.T
        )
        np.testing.assert_array_equal(
            model.gene_embedding.proj.bias.detach().numpy(), jax_lb
        )

        # (2) Forward-level parity: model's gene_embedding(gene_ids) must
        # reproduce the JAX-side computation ge @ w + b.
        with torch.no_grad():
            got = model.gene_embedding().detach().numpy()
        expected = jax_ge @ jax_lw + jax_lb
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)

    def test_missing_keys_raises(self):
        torch.save(
            {"gene_embedding.embed.weight": torch.zeros(self.n_genes, self.init_dim)},
            self.pt_path,
        )
        model = self._fresh_model()
        with self.assertRaises(KeyError) as ctx:
            self.mlm._load_gene_embedding_from_pt(model, self.pt_path)
        self.assertIn("gene_embedding.proj.weight", str(ctx.exception))

    def test_shape_mismatch_raises(self):
        torch.save(
            {
                "gene_embedding.embed.weight": torch.zeros(self.n_genes + 1, self.init_dim),
                "gene_embedding.proj.weight": torch.zeros(self.embed_dim, self.init_dim),
                "gene_embedding.proj.bias": torch.zeros(self.embed_dim),
            },
            self.pt_path,
        )
        model = self._fresh_model()
        with self.assertRaises(ValueError) as ctx:
            self.mlm._load_gene_embedding_from_pt(model, self.pt_path)
        self.assertIn("embed.weight shape mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
