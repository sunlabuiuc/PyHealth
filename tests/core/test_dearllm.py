"""Tests for the DearLLM model and its mortality task helpers.

All tests use tiny synthetic data: no datasets, downloads or LLMs.
"""

import math
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DearLLM, DearLLMGraphEncoder
from pyhealth.tasks import (
    DearLLMMortalityPredictionMIMIC3,
    DearLLMMortalityPredictionMIMIC4,
)
from pyhealth.tasks.dearllm_mortality import (
    EMPTY_EDGE,
    compute_cf_ipf,
    compute_ipf,
    normalize_perplexity_graphs,
)

SCHEMA = {
    "conditions": "nested_sequence",
    "graph_nodes": "sequence",
    "graph_node_weights": "tensor",
    "graph_edges": "tensor",
}
SMALL = {
    "embedding_dim": 4,
    "hidden_dim": 4,
    "node_dim": 4,
    "graph_hidden_dims": (4, 4),
}


def _samples():
    return [
        {
            "patient_id": "p0",
            "conditions": [["a", "b"], ["b", "c"]],
            "graph_nodes": ["a", "b", "c"],
            "graph_node_weights": [0.1, 0.2, 0.3],
            "graph_edges": [[0, 1, 0.5], [1, 2, 0.9], [2, 0, 0.2]],
            "mortality": 1,
        },
        {
            "patient_id": "p1",
            "conditions": [["d"]],
            "graph_nodes": ["d"],
            "graph_node_weights": [0.4],
            "graph_edges": [EMPTY_EDGE],
            "mortality": 0,
        },
    ]


def _reference_gcn_layer(x, edges, weight, bias, add_self_loops):
    """Per-edge GCNConv formula (as in torch_geometric), for one graph.

    out[j] = bias + sum over edges i->j of w_ij / sqrt(deg_i deg_j) * (x W)[i]
    with deg the weighted in-degree.
    """
    n = x.shape[0]
    edge_list = [(int(s), int(d), float(w)) for s, d, w in edges if w != 0]
    if add_self_loops:
        edge_list += [(i, i, 1.0) for i in range(n)]
    deg = [0.0] * n
    for _, d, w in edge_list:
        deg[d] += w
    h = x @ weight.T
    out = bias.expand(n, -1).clone()
    for s, d, w in edge_list:
        out[d] += w / math.sqrt(deg[s] * deg[d]) * h[s] if deg[s] > 0 else 0
    return out


class TestDearLLMGraphEncoder(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_build_adjacency_and_padding_rows(self):
        edges = torch.tensor([[[0.0, 1.0, 0.5], [0.0, 0.0, 0.0]]])
        adj = DearLLMGraphEncoder.build_adjacency(edges, 2)
        self.assertTrue(torch.equal(adj, torch.tensor([[[0.0, 0.5], [0.0, 0.0]]])))

    def test_invalid_edge_index_raises(self):
        edges = torch.tensor([[[0.0, 5.0, 0.5]]])
        with self.assertRaises(ValueError):
            DearLLMGraphEncoder.build_adjacency(edges, 2)

    def test_matches_reference_gcn(self):
        for self_loops in (False, True):
            enc = DearLLMGraphEncoder(
                num_nodes=6, node_dim=4, hidden_dims=(3,), add_self_loops=self_loops
            )
            with torch.no_grad():
                enc.conv_biases[0].uniform_(-1, 1)
            enc.eval()
            node_ids = torch.tensor([[2, 3, 4, 5]])
            edges = torch.tensor(
                [[[0, 1, 0.5], [1, 2, 0.9], [2, 0, 0.3], [3, 1, 0.7], [0, 2, 0.4]]]
            )
            weights = torch.tensor([[0.1, 0.5, 0.2, 0.9]])
            emb, _ = enc(node_ids, weights, edges)

            x = enc.norms[0](enc.node_embedding(node_ids[0]))
            ref = torch.relu(
                _reference_gcn_layer(
                    x, edges[0], enc.convs[0].weight, enc.conv_biases[0], self_loops
                )
            )
            ref_emb = (torch.softmax(weights[0], dim=0).unsqueeze(1) * ref).sum(0)
            self.assertTrue(torch.allclose(emb[0], ref_emb, atol=1e-5), self_loops)

    def test_padding_invariance(self):
        enc = DearLLMGraphEncoder(num_nodes=8, node_dim=4, hidden_dims=(4, 4))
        enc.eval()
        alone, _ = enc(
            torch.tensor([[2, 3]]),
            torch.tensor([[0.3, 0.6]]),
            torch.tensor([[[0.0, 1.0, 0.8]]]),
        )
        batched, _ = enc(
            torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]]),
            torch.tensor([[0.3, 0.6, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4]]),
            torch.tensor(
                [
                    [[0.0, 1.0, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.5], [1.0, 2.0, 0.5], [2.0, 3.0, 0.5]],
                ]
            ),
        )
        self.assertTrue(torch.allclose(alone[0], batched[0], atol=1e-6))

    def test_pooling_weights(self):
        enc = DearLLMGraphEncoder(num_nodes=5, node_dim=4, hidden_dims=(4,))
        ids = torch.tensor([[1, 2, 3], [4, 0, 0]])
        weights = torch.tensor([[0.1, 0.2, 0.3], [0.5, 0.0, 0.0]])
        edges = torch.zeros(2, 1, 3)
        _, pi = enc(ids, weights, edges)
        self.assertTrue(torch.allclose(pi.sum(dim=1), torch.ones(2)))
        self.assertEqual(pi[1, 1:].abs().sum().item(), 0.0)
        self.assertGreater(pi[0, 2].item(), pi[0, 0].item())

        mean_enc = DearLLMGraphEncoder(
            num_nodes=5, node_dim=4, hidden_dims=(4,), pooling="mean"
        )
        _, pi_mean = mean_enc(ids, weights, edges)
        self.assertTrue(torch.allclose(pi_mean[0], torch.full((3,), 1 / 3)))

    def test_empty_graph_gives_zero_vector(self):
        enc = DearLLMGraphEncoder(num_nodes=5, node_dim=4, hidden_dims=(4,))
        emb, pi = enc(
            torch.tensor([[1, 2], [0, 0]]),
            torch.tensor([[0.1, 0.2], [0.0, 0.0]]),
            torch.zeros(2, 1, 3),
        )
        self.assertFalse(torch.isnan(emb).any())
        self.assertEqual(emb[1].abs().sum().item(), 0.0)
        self.assertEqual(pi[1].abs().sum().item(), 0.0)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            DearLLMGraphEncoder(num_nodes=5, hidden_dims=())
        with self.assertRaises(ValueError):
            DearLLMGraphEncoder(num_nodes=5, pooling="max")


class TestDearLLM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = create_sample_dataset(
            samples=_samples(),
            input_schema=SCHEMA,
            output_schema={"mortality": "binary"},
        )
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=2)))

    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)

    def test_forward_all_backbones(self):
        for backbone in ("gru", "lstm", "transformer"):
            model = DearLLM(self.dataset, backbone=backbone, heads=2, **SMALL)
            out = model(**self.batch)
            self.assertEqual(out["y_prob"].shape, (2, 1))
            self.assertEqual(out["logit"].shape, (2, 1))
            self.assertEqual(out["loss"].dim(), 0)
            self.assertTrue(torch.isfinite(out["loss"]))
            self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())

    def test_variants(self):
        for kwargs in (
            {"add_self_loops": True},
            {"pooling": "mean"},
            {"num_layers": 2},
        ):
            model = DearLLM(self.dataset, **SMALL, **kwargs)
            self.assertTrue(torch.isfinite(model(**self.batch)["loss"]))

    def test_gradients_reach_all_parts(self):
        model = DearLLM(self.dataset, **SMALL)
        model(**self.batch)["loss"].backward()
        for name in (
            "embeddings.conditions.weight",
            "graph_encoder.node_embedding.weight",
            "graph_encoder.convs.0.weight",
            "graph_encoder.convs.1.weight",
            "fc.weight",
        ):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0.0, name)

    def test_embed_output(self):
        model = DearLLM(self.dataset, **SMALL)
        out = model(**self.batch, embed=True)
        self.assertEqual(out["embed"].shape, (2, 4 + 4))
        self.assertEqual(out["node_weights"].shape, (2, 3))

    def test_predict_without_label(self):
        model = DearLLM(self.dataset, **SMALL)
        batch = {k: v for k, v in self.batch.items() if k != "mortality"}
        out = model(**batch)
        self.assertNotIn("loss", out)
        self.assertEqual(out["y_prob"].shape, (2, 1))

    def test_parameter_groups(self):
        model = DearLLM(self.dataset, **SMALL)
        groups = model.parameter_groups(lr=1e-3, graph_lr=1e-4, weight_decay=0.1)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(list(model.parameters())))
        conv = model.graph_encoder.convs[0].weight
        group = next(g for g in groups if any(p is conv for p in g["params"]))
        self.assertEqual(group["lr"], 1e-4)
        self.assertEqual(group["weight_decay"], 0.1)
        torch.optim.Adam(groups).step()

    def test_state_dict_round_trip(self):
        model = DearLLM(self.dataset, **SMALL).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "dearllm.pt")
            torch.save(model.state_dict(), path)
            other = DearLLM(self.dataset, **SMALL).eval()
            other.load_state_dict(torch.load(path))
        self.assertTrue(
            torch.allclose(model(**self.batch)["logit"], other(**self.batch)["logit"])
        )

    def test_schema_validation(self):
        no_graph = create_sample_dataset(
            samples=[
                {"patient_id": "p0", "conditions": [["a"]], "mortality": 1},
                {"patient_id": "p1", "conditions": [["b"]], "mortality": 0},
            ],
            input_schema={"conditions": "nested_sequence"},
            output_schema={"mortality": "binary"},
        )
        with self.assertRaises(ValueError):
            DearLLM(no_graph, **SMALL)
        with self.assertRaises(ValueError):
            DearLLM(self.dataset, backbone="hitanet", **SMALL)


def _event(**kwargs):
    return SimpleNamespace(**kwargs)


class _FakePatient:
    """Minimal stand-in for pyhealth.data.Patient."""

    def __init__(self, patient_id, visits, died, attr="icd9_code"):
        self.patient_id = patient_id
        self.admissions = [
            _event(hadm_id=i, timestamp=i, hospital_expire_flag=0)
            for i in range(len(visits))
        ]
        if self.admissions:
            self.admissions[-1].hospital_expire_flag = int(died)
        self.diagnoses = [
            _event(hadm_id=i, **{attr: code})
            for i, codes in enumerate(visits)
            for code in codes
        ]

    def get_events(self, event_type, filters=None):
        if event_type == "admissions":
            return self.admissions
        ((_, _, hadm_id),) = filters
        return [e for e in self.diagnoses if e.hadm_id == hadm_id]


class TestDearLLMTask(unittest.TestCase):
    def test_compute_ipf(self):
        ipf = compute_ipf({"p0": ["a", "b", "a"], "p1": ["a"], "p2": ["c"]})
        self.assertAlmostEqual(ipf["a"], math.log(3 / 3))
        self.assertAlmostEqual(ipf["b"], math.log(3 / 2))

    def test_compute_cf_ipf(self):
        scores = compute_cf_ipf(
            ["a", "a", "b", "c"], ["a", "b", "z"], {"a": 1.0, "b": 2.0}
        )
        self.assertEqual(scores, [0.5, 0.5, 0.0])
        self.assertEqual(compute_cf_ipf(["a", "b"], ["a"], None), [0.5])
        self.assertEqual(compute_cf_ipf([], ["a"], None), [0.0])

    def test_normalize_perplexity_graphs(self):
        graphs = {
            "p0": [("a", "b", 1.0), ("b", "a", 3.0), ("a", "c", 9.0)],
            "p1": [("c", "d", float("inf")), ("d", "c", 0.5)],
        }
        norm, bounds = normalize_perplexity_graphs(graphs, bounds=(1.0, 5.0))
        self.assertEqual(bounds, (1.0, 5.0))
        self.assertEqual(norm["p0"], [("a", "b", 1.0), ("b", "a", 0.5)])
        self.assertEqual(norm["p1"], [("d", "c", 1.0)])

        _, fitted = normalize_perplexity_graphs(graphs, 0.0, 1.0)
        self.assertEqual(fitted, (0.5, 9.0))
        with self.assertRaises(ValueError):
            normalize_perplexity_graphs({"p0": [("a", "b", float("inf"))]})

    def test_sample_fields(self):
        patient = _FakePatient(
            "7", [["a", "b"], ["b", "c"], ["c", "d"], ["x"]], died=True
        )
        graphs = {
            "7": [
                ("b", "c", 0.8),
                ("c", "d", 0.4),
                ("a", "b", 0.9),  # 'a' is not a node with graph_visits=2
                ("c", "c", 1.0),  # self edge, ignored
            ]
        }
        task = DearLLMMortalityPredictionMIMIC3(graphs=graphs, graph_visits=2)
        (sample,) = task(patient)
        self.assertEqual(sample["mortality"], 1)
        self.assertEqual(sample["conditions"], [["a", "b"], ["b", "c"], ["c", "d"]])
        self.assertEqual(sample["graph_nodes"], ["b", "c", "d"])
        self.assertEqual(sample["graph_edges"], [[0, 1, 0.8], [1, 2, 0.4]])
        self.assertEqual(sample["graph_node_weights"], [2 / 6, 2 / 6, 1 / 6])

    def test_empty_graph_and_filters(self):
        patient = _FakePatient("1", [["a"], ["b"]], died=False)
        (sample,) = DearLLMMortalityPredictionMIMIC3()(patient)
        self.assertEqual(sample["graph_edges"], [EMPTY_EDGE])
        self.assertEqual(sample["mortality"], 0)
        self.assertEqual(
            DearLLMMortalityPredictionMIMIC3(drop_empty_graphs=True)(patient), []
        )
        self.assertEqual(
            DearLLMMortalityPredictionMIMIC3()(_FakePatient("2", [["a"]], True)), []
        )
        self.assertEqual(
            DearLLMMortalityPredictionMIMIC3()(_FakePatient("3", [[], ["a"]], True)), []
        )
        with self.assertRaises(ValueError):
            DearLLMMortalityPredictionMIMIC3(graph_visits=0)

    def test_mimic4_reads_icd_code(self):
        patient = _FakePatient("4", [["I10", "E119"], ["N179"]], True, attr="icd_code")
        (sample,) = DearLLMMortalityPredictionMIMIC4()(patient)
        self.assertEqual(sample["graph_nodes"], ["I10", "E119"])

    def test_task_samples_train_the_model(self):
        task = DearLLMMortalityPredictionMIMIC3(
            graphs={"0": [("a", "b", 0.7)], "1": [("c", "d", 0.2)]}
        )
        samples = task(_FakePatient("0", [["a", "b"], ["z"]], True)) + task(
            _FakePatient("1", [["c"], ["d", "c"], ["z"]], False)
        )
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )
        model = DearLLM(dataset, **SMALL)
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertTrue(torch.isfinite(model(**batch)["loss"]))


if __name__ == "__main__":
    unittest.main()
