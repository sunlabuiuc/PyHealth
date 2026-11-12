"""Tests for the SHy model.

Contributor: Alon Levy (NetID: alonl2, alonl2@illinois.edu)
Paper: "Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction"
    (Yu et al., CHIL 2025), https://arxiv.org/abs/2502.10689

Uses a handful of synthetic samples, a fake 3-level code hierarchy (so no
ICD-9 table is downloaded) and tiny layer sizes, so every test runs in
milliseconds.
"""

import os
import tempfile
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import SHy
from pyhealth.models.shy import (
    FidelityDecoder,
    HierarchicalCodeEmbedding,
    Incidences,
    PhenotypeEncoder,
    UniGINConv,
    build_code_levels,
)

ANCESTORS = {
    "4280": ["428", "390-459"],
    "4019": ["401", "390-459"],
    "25000": ["250", "240-279"],
    "2724": ["272", "240-279"],
    "5849": ["584", "580-629"],
}

SAMPLES = [
    {
        "patient_id": "p0",
        "conditions": [["4280", "25000"], ["4019", "4280", "2724"]],
        "visit_gaps": [0.0, 12.0],
        "next_conditions": ["4280", "5849"],
    },
    {
        "patient_id": "p1",
        "conditions": [["2724"], ["25000"], ["4019", "5849"]],
        "visit_gaps": [0.0, 30.0, 400.0],
        "next_conditions": ["4019"],
    },
    {
        "patient_id": "p2",
        "conditions": [["5849", "4019"]],
        "visit_gaps": [0.0],
        "next_conditions": ["2724", "25000"],
    },
]

SMALL = {
    "code_ancestors": ANCESTORS,
    "dim_per_level": 4,
    "num_levels": 3,
    "hgnn_dim": 8,
    "num_heads": 2,
    "emb_dim": 8,
    "num_phenotypes": 3,
    "num_cos_heads": 2,
    "hidden_dim": 8,
    "key_dim": 8,
    "num_attention_heads": 2,
    "add_ratio": 0.5,
}


def make_dataset(with_gaps: bool = False):
    input_schema = {"conditions": "nested_sequence"}
    samples = SAMPLES
    if with_gaps:
        input_schema["visit_gaps"] = "tensor"
    else:
        samples = [{k: v for k, v in s.items() if k != "visit_gaps"} for s in SAMPLES]
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema={"next_conditions": "multilabel"},
        dataset_name="shy_test",
    )


class TestSHy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=3)))
        cls.num_labels = cls.dataset.output_processors["next_conditions"].size()

    def make_model(self, **overrides):
        kwargs = dict(SMALL)
        kwargs.update(overrides)
        return SHy(self.dataset, **kwargs)

    def test_instantiation(self):
        model = self.make_model()
        self.assertIsInstance(model, SHy)
        self.assertEqual(model.mode, "multilabel")
        self.assertEqual(len(model.extractors), 3)
        self.assertEqual(model.code_embedding.output_dim, 12)

    def test_forward_outputs_and_shapes(self):
        model = self.make_model()
        out = model(**self.batch)
        for key in ("loss", "y_prob", "y_true", "logit", "phenotype_weights"):
            self.assertIn(key, out)
        self.assertEqual(out["y_prob"].shape, (3, self.num_labels))
        self.assertEqual(out["y_true"].shape, (3, self.num_labels))
        self.assertEqual(out["phenotype_weights"].shape, (3, 3))
        self.assertEqual(out["loss"].dim(), 0)
        self.assertTrue(torch.isfinite(out["loss"]))
        # Mixture of softmax predictions: each row sums to one.
        torch.testing.assert_close(
            out["y_prob"].sum(-1), torch.ones(3), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            out["phenotype_weights"].sum(-1), torch.ones(3), atol=1e-5, rtol=0
        )

    def test_backward_gradients(self):
        model = self.make_model()
        model.train()
        out = model(**self.batch)
        out["loss"].backward()
        grads = {
            name: p.grad
            for name, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
        self.assertTrue(any(n.startswith("code_embedding") for n in grads))
        self.assertTrue(any(n.startswith("convs") for n in grads))
        self.assertTrue(any("mask_mlp" in n for n in grads))
        self.assertTrue(any(n.startswith("decoder") for n in grads))
        self.assertTrue(any(n.startswith("classifier") for n in grads))

    def test_inference_without_labels(self):
        model = self.make_model()
        model.eval()
        batch = {k: v for k, v in self.batch.items() if k != "next_conditions"}
        with torch.no_grad():
            out = model(**batch)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (3, self.num_labels))

    def test_eval_is_deterministic_without_sampling(self):
        model = self.make_model(stochastic_eval=False)
        model.eval()
        with torch.no_grad():
            a = model(**self.batch)["y_prob"]
            b = model(**self.batch)["y_prob"]
        torch.testing.assert_close(a, b)

    def test_eval_sampling_is_seeded(self):
        model = self.make_model()
        model.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            a = model(**self.batch)["y_prob"]
            torch.manual_seed(0)
            b = model(**self.batch)["y_prob"]
        torch.testing.assert_close(a, b)

    def test_single_phenotype(self):
        model = self.make_model(num_phenotypes=1)
        out = model(**self.batch)
        self.assertEqual(out["phenotype_weights"].shape, (3, 1))
        self.assertEqual(float(out["loss_distinct"]), 0.0)
        out["loss"].backward()

    def test_no_recovery_and_random_links(self):
        for overrides in ({"add_ratio": 0.0}, {"random_links": True}):
            model = self.make_model(**overrides)
            out = model(**self.batch)
            self.assertTrue(torch.isfinite(out["loss"]))

    def test_recovery_adds_expected_number_of_links(self):
        model = self.make_model(add_ratio=0.5)
        explanations = model.explain(**self.batch)
        self.assertEqual(len(explanations), 3)
        self.assertEqual(len(explanations[0]["phenotypes"]), 3)
        inc = Incidences(self.batch["conditions"], model.num_codes)
        x0 = model.code_embedding()
        x_local, x_shared = model._encode_codes(inc, x0)
        x_edge = inc.edge_mean(x_local[inc.node])
        edge, code = model.extractors[0].recover(x_local, x_shared, x_edge, inc)
        # p0 has 5 links -> 2 added, p1 has 4 -> 2, p2 has 2 -> 1.
        self.assertEqual(edge.numel(), 5)
        original = set(zip(inc.edge.tolist(), inc.code.tolist()))
        self.assertTrue(original.isdisjoint(zip(edge.tolist(), code.tolist())))
        self.assertTrue(bool((code > 1).all()))

    def test_flat_embedding_and_no_message_passing(self):
        model = self.make_model(use_hierarchy=False, num_layers=0)
        self.assertFalse(model.code_embedding.hierarchical)
        out = model(**self.batch)
        out["loss"].backward()

    def test_time_gaps(self):
        dataset = make_dataset(with_gaps=True)
        batch = next(iter(get_dataloader(dataset, batch_size=3)))
        model = SHy(dataset, use_time_gaps=True, **SMALL)
        out = model(**batch)
        self.assertTrue(torch.isfinite(out["loss"]))
        with self.assertRaises(ValueError):
            SHy(self.dataset, use_time_gaps=True, **SMALL)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.make_model(loss_weights=(0.003, 0.00025))
        model = self.make_model()
        batch = dict(self.batch, conditions=self.batch["conditions"][0])
        with self.assertRaises(ValueError):
            model(**batch)

    def test_explain_restores_training_mode(self):
        model = self.make_model().train()
        model.explain(**self.batch)
        self.assertTrue(model.training)

    def test_loss_weights_zero_gives_prediction_loss(self):
        model = self.make_model(loss_weights=(0.0, 0.0, 0.0))
        model.eval()
        with torch.no_grad():
            out = model(**self.batch)
        torch.testing.assert_close(out["loss"], out["loss_pred"])

    def test_checkpoint_round_trip(self):
        model = self.make_model(stochastic_eval=False)
        model.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shy.pt")
            torch.save(model.state_dict(), path)
            restored = self.make_model(stochastic_eval=False)
            restored.load_state_dict(torch.load(path))
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(
                model(**self.batch)["y_prob"], restored(**self.batch)["y_prob"]
            )


class TestSHyLayers(unittest.TestCase):
    def test_build_code_levels(self):
        codes = ["<pad>", "<unk>", "4280", "4019", "9999"]
        levels, sizes = build_code_levels(
            codes, lambda c: ANCESTORS.get(c, []), num_levels=3
        )
        self.assertEqual(levels.shape, (5, 3))
        self.assertEqual(levels[0].tolist(), [0, 0, 0])
        # 4280 and 4019 share the chapter, not the category.
        self.assertEqual(levels[2, 0], levels[3, 0])
        self.assertNotEqual(levels[2, 1], levels[3, 1])
        self.assertEqual(len(sizes), 3)

    def test_long_paths_keep_top_levels_and_code(self):
        # Real ICD-9-CM shape: 4-digit codes have 3 ancestors, 5-digit ones 4.
        ancestors = {
            "4280": ["428", "420-429", "390-459"],
            "4290": ["429", "420-429", "390-459"],
            "25000": ["250.0", "250", "249-259", "240-279"],
            "25010": ["250.1", "250", "249-259", "240-279"],
        }
        levels, _ = build_code_levels(
            ["<pad>", "4280", "4290", "25000", "25010"],
            lambda c: ancestors.get(c, []),
            num_levels=4,
        )
        # 428x and 429x share chapter and block but not the category.
        self.assertEqual(levels[1, :2].tolist(), levels[2, :2].tolist())
        self.assertNotEqual(levels[1, 2].item(), levels[2, 2].item())
        # 5-digit codes keep chapter, block, category and the code itself.
        self.assertEqual(levels[3, :3].tolist(), levels[4, :3].tolist())
        self.assertNotEqual(levels[3, 3].item(), levels[4, 3].item())
        self.assertEqual(len(set(levels[:, 3].tolist())), 5)

    def test_hierarchical_embedding_shares_ancestors(self):
        levels, sizes = build_code_levels(
            ["<pad>", "4280", "4019"], lambda c: ANCESTORS.get(c, []), 3
        )
        emb = HierarchicalCodeEmbedding(3, 4, levels, sizes, 3)
        x = emb()
        self.assertEqual(x.shape, (3, 12))
        torch.testing.assert_close(x[1, :4], x[2, :4])

    def test_incidences(self):
        codes = torch.tensor([[[2, 3], [3, 0]], [[4, 0], [0, 0]]])
        inc = Incidences(codes, num_codes=5)
        self.assertEqual(inc.edge.tolist(), [0, 0, 1, 2])
        self.assertEqual(inc.num_local, 3)
        self.assertEqual(inc.visit_mask.tolist(), [[True, True], [True, False]])
        local = inc.lookup_local(torch.tensor([0, 0, 1]), torch.tensor([3, 4, 4]))
        self.assertEqual(local[0].item() >= 0, True)
        self.assertEqual(local[1].item(), -1)

    def test_unigin_conv(self):
        inc = Incidences(torch.tensor([[[1, 2], [2, 0]]]), num_codes=3)
        conv = UniGINConv(4, 5)
        local, shared = conv(torch.randn(inc.num_local, 4), torch.randn(3, 4), inc)
        self.assertEqual(local.shape, (2, 5))
        self.assertEqual(shared.shape, (3, 5))

    def test_encoder_without_real_visits_is_finite(self):
        enc = PhenotypeEncoder(emb_dim=4, hidden_dim=3)
        out = enc(torch.randn(1, 2, 4), torch.tensor([[False, False]]))
        self.assertTrue(bool(torch.isfinite(out).all()))

    def test_encoder_ignores_padding_visits(self):
        enc = PhenotypeEncoder(emb_dim=4, hidden_dim=3)
        visits = torch.randn(1, 3, 4)
        mask = torch.tensor([[True, True, False]])
        a = enc(visits, mask)
        visits2 = visits.clone()
        visits2[0, 2] = 100.0
        torch.testing.assert_close(a, enc(visits2, mask))

    def test_fidelity_decoder(self):
        dec = FidelityDecoder(latent_dim=6, emb_dim=4, num_codes=7)
        out = dec(torch.randn(2, 6), torch.randn(2, 3, 4))
        self.assertEqual(out.shape, (2, 3, 7))
        self.assertTrue(bool(((out > 0) & (out < 1)).all()))


if __name__ == "__main__":
    unittest.main()
