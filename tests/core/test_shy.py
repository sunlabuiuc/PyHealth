"""Tests for the SHy (Self-Explaining Hypergraph Neural Networks) model.

Uses small synthetic data only. No real datasets (MIMIC, etc.) are used.
"""

import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SHy
from pyhealth.models.shy import (
    FinalClassifier,
    HSLPart1,
    HierarchicalEmbedding,
    UniGINConv,
    shy_loss,
)


SYNTHETIC_CODE_LEVELS = np.array(
    [[1, 1, 1], [1, 1, 2], [1, 2, 3], [2, 3, 4], [2, 3, 5]]
)

SMALL_HPARAMS = dict(
    code_levels=SYNTHETIC_CODE_LEVELS,
    single_dim=8,
    hgnn_dim=16,
    after_hgnn_dim=16,
    hgnn_layer_num=1,
    nhead=2,
    num_tp=2,
    temperatures=[0.5, 0.5],
    add_ratios=[0.1, 0.1],
    n_c=3,
    hid_state_dim=16,
    key_dim=16,
    sa_head=2,
    dropout=0.0,
)

CODES = ["C001", "C002", "C003", "C004", "C005"]


def _make_dataset():
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": [[CODES[0], CODES[1]], [CODES[2]]],
            "label": [CODES[0], CODES[2]],
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": [[CODES[2], CODES[3]], [CODES[4]]],
            "label": [CODES[3], CODES[4]],
        },
        {
            "patient_id": "p2",
            "visit_id": "v2",
            "conditions": [[CODES[0], CODES[4]], [CODES[1], CODES[3]]],
            "label": [CODES[0], CODES[1], CODES[4]],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence"},
        output_schema={"label": "multilabel"},
        dataset_name="test_shy",
    )


def _make_single_visit_dataset():
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": [[CODES[0], CODES[1]]],
            "label": [CODES[0]],
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": [[CODES[2]]],
            "label": [CODES[2], CODES[3]],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence"},
        output_schema={"label": "multilabel"},
        dataset_name="test_shy_single_visit",
    )


def _make_varying_visit_dataset():
    samples = [
        {
            "patient_id": "p0",
            "visit_id": "v0",
            "conditions": [[CODES[0]]],  # 1 visit
            "label": [CODES[0]],
        },
        {
            "patient_id": "p1",
            "visit_id": "v1",
            "conditions": [[CODES[1]], [CODES[2]]],  # 2 visits
            "label": [CODES[1]],
        },
        {
            "patient_id": "p2",
            "visit_id": "v2",
            "conditions": [[CODES[2]], [CODES[3]], [CODES[4]]],  # 3 visits
            "label": [CODES[2], CODES[4]],
        },
        {
            "patient_id": "p3",
            "visit_id": "v3",
            "conditions": [[CODES[0]], [CODES[1]], [CODES[2]], [CODES[3]]],  # 4 visits
            "label": [CODES[3]],
        },
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"conditions": "nested_sequence"},
        output_schema={"label": "multilabel"},
        dataset_name="test_shy_varying_visits",
    )


def _get_batch(dataset, batch_size):
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _assert_valid_output(testcase, ret):
    testcase.assertIn("loss", ret)
    testcase.assertIn("y_prob", ret)
    testcase.assertIn("y_true", ret)
    testcase.assertIn("logit", ret)
    testcase.assertEqual(ret["y_prob"].shape, ret["y_true"].shape)
    testcase.assertTrue(np.isfinite(ret["loss"].item()))
    testcase.assertTrue((ret["y_prob"] >= 0).all().item())
    testcase.assertTrue((ret["y_prob"] <= 1).all().item())


class TestSHyLearns(unittest.TestCase):
    """Verify the model trains end-to-end and loss decreases."""

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_loss_decreases_after_training(self):
        """Train for 20 steps and verify loss drops."""
        dataset = _make_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        batch = _get_batch(dataset, batch_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        ret = model(**batch)
        initial_loss = ret["loss"].item()

        for _ in range(20):
            ret = model(**batch)
            optimizer.zero_grad()
            ret["loss"].backward()
            optimizer.step()

        final_loss = ret["loss"].item()

        self.assertLess(final_loss, initial_loss)
        self.assertTrue(np.isfinite(final_loss))
        self.assertEqual(ret["y_prob"].shape, ret["y_true"].shape)
        self.assertTrue((ret["y_prob"] >= 0).all())
        self.assertTrue((ret["y_prob"] <= 1).all())

        max_grad = max(
            p.grad.abs().max().item()
            for p in model.parameters()
            if p.grad is not None
        )
        self.assertGreater(max_grad, 0.0)

    def test_single_visit_patients(self):
        """Forward pass should work with single-visit patients."""
        dataset = _make_single_visit_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        batch = _get_batch(dataset, batch_size=2)

        ret = model(**batch)
        _assert_valid_output(self, ret)

    def test_batch_size_one(self):
        """Forward pass should work with batch size 1."""
        dataset = _make_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        batch = _get_batch(dataset, batch_size=1)

        ret = model(**batch)
        self.assertEqual(ret["y_prob"].shape[0], 1)
        _assert_valid_output(self, ret)

    def test_k_equals_one_forward(self):
        """Forward pass should work with a single phenotype."""
        dataset = _make_dataset()
        params = dict(SMALL_HPARAMS)
        params["num_tp"] = 1
        params["temperatures"] = [0.5]
        params["add_ratios"] = [0.1]

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=3)

        ret = model(**batch)
        _assert_valid_output(self, ret)

    def test_k_equals_seven_forward(self):
        """Forward pass should work with many phenotypes."""
        dataset = _make_dataset()
        params = dict(SMALL_HPARAMS)
        params["num_tp"] = 7
        params["temperatures"] = [0.5] * 7
        params["add_ratios"] = [0.1] * 7

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=3)

        ret = model(**batch)
        _assert_valid_output(self, ret)

    def test_no_hgnn_layers_linear_fallback(self):
        """Forward pass should work when hgnn_layer_num=-1."""
        dataset = _make_dataset()
        params = dict(SMALL_HPARAMS)
        params["hgnn_layer_num"] = -1

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=3)

        ret = model(**batch)
        _assert_valid_output(self, ret)

    def test_varying_visit_lengths_same_batch(self):
        """Forward pass should work when visit counts differ per patient."""
        dataset = _make_varying_visit_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        batch = _get_batch(dataset, batch_size=4)

        ret = model(**batch)
        self.assertEqual(ret["y_prob"].shape[0], 4)
        _assert_valid_output(self, ret)

    def test_unigat_forward(self):
        """Forward pass should work with UniGATConv."""
        dataset = _make_dataset()
        params = dict(SMALL_HPARAMS)
        params["conv_type"] = "UniGATConv"

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=3)

        ret = model(**batch)
        _assert_valid_output(self, ret)

    def test_eval_mode_outputs_valid(self):
        """model.eval() should still produce valid outputs."""
        dataset = _make_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        batch = _get_batch(dataset, batch_size=3)

        model.eval()
        with torch.no_grad():
            ret = model(**batch)

        _assert_valid_output(self, ret)

    def test_save_load_identical_predictions(self):
        """Saving/loading state_dict should preserve predictions."""
        dataset = _make_dataset()
        model = SHy(dataset=dataset, **SMALL_HPARAMS)
        batch = _get_batch(dataset, batch_size=3)

        model.eval()
        with torch.no_grad():
            pred_before = model(**batch)["y_prob"].detach().cpu()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/shy.pt"
            torch.save(model.state_dict(), path)

            restored = SHy(dataset=dataset, **SMALL_HPARAMS)
            restored.load_state_dict(torch.load(path, map_location="cpu"))
            restored.eval()

            with torch.no_grad():
                pred_after = restored(**batch)["y_prob"].detach().cpu()

        self.assertTrue(torch.allclose(pred_before, pred_after, atol=1e-6, rtol=1e-6))

    def test_positional_embeddings_forward(self):
        """Forward pass should work with positional embeddings enabled."""
        dataset = _make_dataset()
        params = dict(SMALL_HPARAMS)
        params["use_positional"] = True
        params["max_visits"] = 20

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=3)

        ret = model(**batch)
        _assert_valid_output(self, ret)

        # Verify positional embeddings were created
        self.assertTrue(
            hasattr(model.shy_layer.encoder.hgnn, "position_embeddings")
        )
        pos_emb = model.shy_layer.encoder.hgnn.position_embeddings
        self.assertEqual(pos_emb.num_embeddings, 20)  # max_visits

    def test_positional_embeddings_affect_predictions(self):
        """Predictions should differ when positional embeddings are enabled."""
        dataset = _make_dataset()
        batch = _get_batch(dataset, batch_size=3)

        # Model without positional embeddings
        model_no_pos = SHy(dataset=dataset, **SMALL_HPARAMS)
        model_no_pos.eval()
        with torch.no_grad():
            pred_no_pos = model_no_pos(**batch)["y_prob"].detach().cpu()

        # Model with positional embeddings
        params_with_pos = dict(SMALL_HPARAMS)
        params_with_pos["use_positional"] = True
        params_with_pos["max_visits"] = 20
        model_with_pos = SHy(dataset=dataset, **params_with_pos)
        model_with_pos.eval()
        with torch.no_grad():
            pred_with_pos = model_with_pos(**batch)["y_prob"].detach().cpu()

        # Predictions should differ (different random initializations)
        self.assertFalse(
            torch.allclose(pred_no_pos, pred_with_pos, atol=1e-4)
        )

    def test_positional_embeddings_gradients_flow(self):
        """Gradients should flow to positional embeddings during training."""
        dataset = _make_dataset()
        params = dict(SMALL_HPARAMS)
        params["use_positional"] = True
        params["max_visits"] = 20

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=3)

        model.train()
        ret = model(**batch)
        ret["loss"].backward()

        # Check that positional embeddings received gradients
        pos_emb = model.shy_layer.encoder.hgnn.position_embeddings
        self.assertIsNotNone(pos_emb.weight.grad)
        self.assertGreater(pos_emb.weight.grad.abs().sum().item(), 0.0)

    def test_positional_embeddings_varying_visit_lengths(self):
        """Positional embeddings should work with varying visit counts."""
        dataset = _make_varying_visit_dataset()
        params = dict(SMALL_HPARAMS)
        params["use_positional"] = True
        params["max_visits"] = 10

        model = SHy(dataset=dataset, **params)
        batch = _get_batch(dataset, batch_size=4)

        ret = model(**batch)
        _assert_valid_output(self, ret)
        self.assertEqual(ret["y_prob"].shape[0], 4)


class TestSHyComponents(unittest.TestCase):
    """Component-level tests for SHy internals."""

    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)

    def test_uniginconv_output_shape_and_self_loop_effect(self):
        """UniGINConv should preserve expected shape and include self term."""
        conv = UniGINConv(in_channels=2, out_channels=2, heads=1)

        with torch.no_grad():
            conv.W.weight.copy_(torch.eye(2))
            conv.eps.fill_(0.1)

        x = torch.tensor([[1.0, 2.0]])
        vertex = torch.tensor([0])
        edges = torch.tensor([0])

        out = conv(x, vertex, edges)

        self.assertEqual(out.shape, (1, 2))
        expected = 2.1 * x  # (1 + eps) * X + Xv, with Xv = X here
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_hierarchical_embedding_shared_ancestors(self):
        """Codes sharing a level-0 ancestor should share that sub-embedding."""
        code_levels = torch.tensor(
            [
                [1, 1, 1],
                [1, 1, 2],
                [2, 3, 4],
            ],
            dtype=torch.long,
        )
        emb = HierarchicalEmbedding(
            code_levels=code_levels,
            code_num_in_levels=[3, 4, 5],
            code_dims=[2, 2, 2],
        )

        out = emb()
        self.assertEqual(out.shape, (3, 6))

        # First 2 dims correspond to level-0 embedding
        self.assertTrue(torch.allclose(out[0, :2], out[1, :2], atol=1e-6))
        self.assertFalse(torch.allclose(out[0, :2], out[2, :2], atol=1e-6))

    def test_hslpart1_probabilities_in_range(self):
        """HSLPart1 probabilities should lie in [0, 1]."""
        layer = HSLPart1(emb_dim=4)
        x = torch.randn(5, 4)
        vertex = torch.tensor([0, 1, 2, 3, 4])
        edges = torch.tensor([0, 0, 1, 1, 1])

        prob = layer(x, vertex, edges)

        self.assertTrue((prob >= 0).all().item())
        self.assertTrue((prob <= 1).all().item())

    def test_final_classifier_predictions_sum_to_one(self):
        """FinalClassifier softmax predictions should sum to 1."""
        clf = FinalClassifier(
            in_channel=8,
            code_num=5,
            key_dim=8,
            sa_head=2,
            num_tp=2,
        )

        latent_tp = torch.randn(3, 2, 8)
        pred, alpha = clf(latent_tp)

        self.assertEqual(pred.shape, (3, 5))
        self.assertEqual(alpha.shape, (3, 2))
        self.assertTrue(
            torch.allclose(pred.sum(dim=-1), torch.ones(3), atol=1e-5)
        )

    def test_shy_loss_returns_all_four_components_when_k_gt_1(self):
        """shy_loss should return 4 named components when K > 1."""
        device = torch.device("cpu")
        pred = torch.full((2, 5), 0.5, device=device)
        label = torch.tensor(
            [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]],
            dtype=torch.float32,
            device=device,
        )

        original_h = torch.tensor(
            [
                [[1, 0], [0, 1], [0, 0], [1, 0], [0, 0]],
                [[0, 1], [1, 0], [0, 0], [0, 1], [1, 0]],
            ],
            dtype=torch.float32,
            device=device,
        )

        reconstruction = [
            torch.full((5, 2), 0.5, device=device),
            torch.full((5, 2), 0.5, device=device),
        ]

        # K > 1 path => tps[j] shape should have > 2 dims
        tps = [
            torch.rand(2, 5, 2, device=device),
            torch.rand(2, 5, 2, device=device),
        ]

        alphas = torch.tensor(
            [[0.6, 0.4], [0.5, 0.5]],
            dtype=torch.float32,
            device=device,
        )
        visit_lens = [2, 2]
        obj_r = [1.0, 0.1, 0.01, 0.01]

        total, components, names = shy_loss(
            pred=pred,
            label=label,
            original_h=original_h,
            reconstruction=reconstruction,
            tps=tps,
            alphas=alphas,
            visit_lens=visit_lens,
            obj_r=obj_r,
            device=device,
        )

        self.assertTrue(torch.isfinite(total).item())
        self.assertEqual(len(components), 4)
        self.assertEqual(
            names,
            ["Prediction", "Fidelity", "Distinctness", "Alpha"],
        )


if __name__ == "__main__":
    unittest.main()
