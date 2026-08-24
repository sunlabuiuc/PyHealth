"""Regression tests for ``pyhealth.medcode.pretrained_embeddings.kg_emb``.

The suite is behavioural: it exercises construction, indexing and splitting
rather than asserting on type annotations, which are metadata and not a
contract.
"""

from __future__ import annotations

import tempfile
import unittest
from typing import Any

import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import collate_fn_dict_with_padding
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import (
    BaseKGDataset,
    SampleKGDataset,
    split,
)
from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import link_prediction_fn


def make_samples(n: int = 8) -> list[dict[str, Any]]:
    """Build ``n`` synthetic link-prediction samples over a 10-entity graph."""
    return [
        {
            "triple": (i % 10, i % 3, (i + 4) % 10),
            "ground_truth_head": [i % 10, (i + 1) % 10],
            "ground_truth_tail": [(i + 4) % 10],
            "subsampling_weight": torch.tensor([0.25]),
        }
        for i in range(n)
    ]


def make_dataset(n: int = 8, **kwargs: Any) -> SampleKGDataset:
    entity2id = {f"e{i}": i for i in range(10)}
    relation2id = {f"r{i}": i for i in range(3)}
    return SampleKGDataset(
        samples=make_samples(n),
        dataset_name="synthetic",
        task_name="link_prediction",
        entity2id=entity2id,
        relation2id=relation2id,
        negative_sampling=4,
        **kwargs,
    )


class TestKGEmbImports(unittest.TestCase):
    """The module must import cleanly -- the original symptom of issue #952."""

    def test_package_imports(self) -> None:
        import pyhealth.medcode.pretrained_embeddings  # noqa: F401

    def test_model_classes_are_exported(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import (
            ComplEx,
            DistMult,
            KGEBaseModel,
            RotatE,
            TransE,
        )

        for cls in (KGEBaseModel, TransE, RotatE, DistMult, ComplEx):
            self.assertTrue(issubclass(cls, torch.nn.Module))


class TestSampleKGDataset(unittest.TestCase):
    """Construction and indexing -- the failure the rename alone does not fix."""

    def test_construction_and_length(self) -> None:
        dataset = make_dataset(n=8)
        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset.entity_num, 10)
        self.assertEqual(dataset.relation_num, 3)

    def test_getitem_returns_the_sample(self) -> None:
        dataset = make_dataset(n=3)
        self.assertEqual(dataset[0]["triple"], (0, 0, 4))
        self.assertIn("ground_truth_head", dataset[1])

    def test_inverse_vocabularies(self) -> None:
        dataset = make_dataset(n=2)
        self.assertEqual(dataset.id2entity[0], "e0")
        self.assertEqual(dataset.id2relation[2], "r2")

    def test_task_specific_hyperparameters_are_captured(self) -> None:
        dataset = make_dataset(n=2)
        self.assertEqual(dataset.task_spec_param, {"negative_sampling": 4})

    def test_missing_vocabularies_do_not_crash(self) -> None:
        dataset = SampleKGDataset(
            samples=make_samples(2), entity_num=10, relation_num=3
        )
        self.assertEqual(dataset.id2entity, {})
        self.assertIsNone(dataset.task_spec_param)

    def test_contradictory_cardinalities_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SampleKGDataset(
                samples=make_samples(1),
                entity_num=99,
                entity2id={f"e{i}": i for i in range(10)},
            )

    def test_stat_returns_a_report(self) -> None:
        report = make_dataset(n=2).stat()
        self.assertIn("Number of triples: 2", report)

    def test_is_a_map_style_dataset(self) -> None:
        dataset = make_dataset(n=2)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertFalse(hasattr(dataset, "set_shuffle"))


class TestSplit(unittest.TestCase):
    """The splitter must partition the dataset and stay reproducible."""

    def test_partition_sizes(self) -> None:
        train, val, test = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=0)
        self.assertEqual((len(train), len(val), len(test)), (6, 2, 2))

    def test_folds_are_disjoint_and_exhaustive(self) -> None:
        train, val, test = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=0)
        triples = [s["triple"] for s in train + val + test]
        self.assertEqual(len(triples), 10)
        self.assertEqual(len(set(triples)), 10)

    def test_is_reproducible_under_a_fixed_seed(self) -> None:
        first = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=7)[0]
        second = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=7)[0]
        self.assertEqual([s["triple"] for s in first], [s["triple"] for s in second])

    def test_global_numpy_state_is_untouched(self) -> None:
        import numpy as np

        np.random.seed(1234)
        before = np.random.rand()
        np.random.seed(1234)
        split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=99)
        self.assertEqual(before, np.random.rand())

    def test_training_fold_carries_hyperparameters(self) -> None:
        train, val, _ = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=0)
        self.assertTrue(train[0]["train"])
        self.assertEqual(train[0]["hyperparameters"], {"negative_sampling": 4})
        self.assertFalse(val[0]["train"])

    def test_malformed_ratios_are_rejected(self) -> None:
        dataset = make_dataset(n=10)
        for bad in ([0.5, 0.2, 0.2], [0.5, 0.5], [1.2, -0.2, 0.0]):
            with self.subTest(ratios=bad), self.assertRaises(ValueError):
                split(dataset, bad, seed=0)

    def test_ratio_sum_error_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "ratios must sum to 1.0, got 0.9"):
            split(make_dataset(n=10), [0.5, 0.2, 0.2], seed=0)


class TestCollateAndForward(unittest.TestCase):
    """Generic padding collation leaves KG lists intact; one train step runs."""

    def test_variable_length_ground_truth_stays_a_python_list(self) -> None:
        dataset = make_dataset(n=4)
        train, _, _ = split(dataset, [1.0, 0.0, 0.0], seed=0)
        loader = DataLoader(
            train, batch_size=2, shuffle=False, collate_fn=collate_fn_dict_with_padding
        )
        batch = next(iter(loader))
        self.assertIsInstance(batch["triple"], list)
        self.assertIsInstance(batch["ground_truth_head"], list)
        self.assertIsInstance(batch["ground_truth_head"][0], list)
        lengths = [len(h) for h in batch["ground_truth_head"]]
        self.assertTrue(all(length >= 1 for length in lengths))

    def test_transe_train_step(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        dataset = make_dataset(n=4)
        train, _, _ = split(dataset, [1.0, 0.0, 0.0], seed=0)
        loader = DataLoader(
            train, batch_size=2, shuffle=False, collate_fn=collate_fn_dict_with_padding
        )
        model = TransE(dataset=dataset, e_dim=8, r_dim=8, ns="uniform")
        out = model(**next(iter(loader)))
        self.assertIn("loss", out)
        out["loss"].backward()


class TestSetTask(unittest.TestCase):
    """Production path: BaseKGDataset.set_task must return a usable SampleKGDataset."""

    def test_set_task_on_a_synthetic_graph(self) -> None:
        class _ToyKG(BaseKGDataset):
            def raw_graph_process(self):
                self.entity2id = {"a": 0, "b": 1, "c": 2}
                self.relation2id = {"r": 0}
                self.entity_num = 3
                self.relation_num = 1
                self.triples = [(0, 0, 1), (1, 0, 2), (2, 0, 0)]

        with tempfile.TemporaryDirectory() as root:
            base = _ToyKG(root=root, dataset_name="toy", refresh_cache=True)
            sample_ds = base.set_task(
                link_prediction_fn, negative_sampling=4, save=False
            )
        self.assertIsInstance(sample_ds, SampleKGDataset)
        self.assertEqual(len(sample_ds), 3)
        self.assertEqual(sample_ds.task_spec_param, {"negative_sampling": 4})
        self.assertIn("triple", sample_ds[0])


class TestScoringInvariants(unittest.TestCase):
    """Mathematical properties the scoring functions must satisfy by construction."""

    def test_distmult_is_symmetric_in_head_and_tail(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import DistMult

        model = DistMult(dataset=make_dataset(n=4), e_dim=8, r_dim=8, gamma=12.0)
        head, relation, tail = (torch.randn(2, 1, 8) for _ in range(3))
        self.assertTrue(
            torch.allclose(
                model.calc(head, relation, tail),
                model.calc(tail, relation, head),
                atol=1e-6,
            )
        )

    def test_transe_scores_a_perfect_triple_at_the_margin(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=make_dataset(n=4), e_dim=8, r_dim=8, gamma=24.0)
        head = torch.zeros(1, 1, 8)
        relation = torch.ones(1, 1, 8)
        tail = torch.ones(1, 1, 8)  # h + r - t == 0
        self.assertTrue(
            torch.allclose(
                model.calc(head, relation, tail), torch.tensor(24.0), atol=1e-6
            )
        )
