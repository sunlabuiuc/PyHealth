"""Regression tests for ``pyhealth.medcode.pretrained_embeddings.kg_emb``.

The suite is behavioural: it exercises construction, indexing and splitting
rather than asserting on type annotations, which are metadata and not a
contract.
"""

from __future__ import annotations

import hashlib
import logging
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import collate_fn_dict_with_padding
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import (
    BaseKGDataset,
    SampleKGDataset,
    UMLSDataset,
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


# A small graph: entities a..d, relations r1, r2. Names first appear out of
# order (c before a, r2 before r1), so numbering by first appearance would
# differ from the sorted ids a=0, b=1, c=2, d=3 and r1=0, r2=1.
TOY_TRIPLES = [
    ("c", "r2", "a"),
    ("a", "r1", "b"),
    ("a", "r1", "c"),
    ("b", "r2", "c"),
    ("d", "r1", "a"),
]

# Twelve triples: the loader stores rows sorted on the string row number
# ("0", "1", "10", "11", "2", ...), so file order is only recovered by a
# numeric sort once there are more than ten rows.
ORDER_TRIPLES = [(f"e{i}", "r", f"e{i + 1}") for i in range(12)]

TOY_CONFIG = """version: "1.0"
tables:
  triples:
    file_path: "kg.tsv"
    patient_id: null
    timestamp: null
    attributes: [head, relation, tail]
"""


def write_kg(root: Path, triples: list[tuple[str, str, str]]) -> Path:
    """Write ``triples`` as ``kg.tsv`` with its config; return the config path."""
    lines = ["head\trelation\ttail"] + ["\t".join(t) for t in triples]
    (root / "kg.tsv").write_text("\n".join(lines) + "\n")
    config = root / "kg.yaml"
    config.write_text(TOY_CONFIG)
    return config


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
    """The module must import cleanly. This was the original symptom of issue #952."""

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
    """Construction and indexing: the failure a rename alone does not fix."""

    def test_construction_and_length(self) -> None:
        dataset = make_dataset(n=8)
        self.assertEqual(len(dataset), 8)
        self.assertEqual(dataset.entity_num, 10)
        self.assertEqual(dataset.relation_num, 3)

    def test_getitem_returns_the_sample(self) -> None:
        dataset = make_dataset(n=3)
        # "triple" is now a pure LongTensor (the Tensor Trick), not the raw
        # tuple, so this is a torch.equal check rather than a tuple ==.
        self.assertTrue(torch.equal(dataset[0]["triple"], torch.tensor([0, 0, 4])))
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

    def test_is_an_in_memory_sample_dataset(self) -> None:
        """SampleKGDataset is deliberately back under the InMemorySampleDataset
        umbrella (see PR discussion), so `set_shuffle` is now expected to be
        present rather than absent — this supersedes the old standalone-Dataset
        isolation check."""
        from pyhealth.datasets.sample_dataset import InMemorySampleDataset

        dataset = make_dataset(n=2)
        self.assertIsInstance(dataset, torch.utils.data.Dataset)
        self.assertIsInstance(dataset, InMemorySampleDataset)
        self.assertTrue(hasattr(dataset, "set_shuffle"))


class TestSplit(unittest.TestCase):
    """The splitter must partition the dataset and stay reproducible."""

    def test_partition_sizes(self) -> None:
        train, val, test = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=0)
        self.assertEqual((len(train), len(val), len(test)), (6, 2, 2))

    def test_folds_are_disjoint_and_exhaustive(self) -> None:
        train, val, test = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=0)
        # "triple" is now a Tensor, which is neither hashable-by-value nor
        # comparable the way a tuple is; compare/hash via .tolist() instead.
        triples = [tuple(s["triple"].tolist()) for s in train + val + test]
        self.assertEqual(len(triples), 10)
        self.assertEqual(len(set(triples)), 10)

    def test_is_reproducible_under_a_fixed_seed(self) -> None:
        first = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=7)[0]
        second = split(make_dataset(n=10), [0.6, 0.2, 0.2], seed=7)[0]
        self.assertEqual(
            [s["triple"].tolist() for s in first],
            [s["triple"].tolist() for s in second],
        )

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
    """Tensor Trick collation: KG fields arrive pre-padded, with masks; one train step runs."""

    def test_ground_truth_collates_to_padded_tensor_with_mask(self) -> None:
        """Supersedes the old "stays a python list" expectation: since the
        Tensor Trick (KGProcessor), triple/ground_truth_* are pre-padded
        pure tensors by the time they leave SampleKGDataset, not raw Python
        lists collated dynamically per batch."""
        dataset = make_dataset(n=4)
        train, _, _ = split(dataset, [1.0, 0.0, 0.0], seed=0)
        loader = DataLoader(
            train, batch_size=2, shuffle=False, collate_fn=collate_fn_dict_with_padding
        )
        batch = next(iter(loader))

        self.assertIsInstance(batch["triple"], torch.Tensor)
        self.assertEqual(tuple(batch["triple"].shape), (2, 3))

        for field in ("ground_truth_head", "ground_truth_tail"):
            gt = batch[field]
            self.assertIsInstance(gt, dict)
            self.assertEqual(gt["value"].shape, gt["mask"].shape)
            self.assertEqual(gt["value"].shape[0], 2)  # batch size
            # Every sample has at least one real (unmasked) entity.
            self.assertTrue(gt["mask"].bool().any(dim=1).all())

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


class TestBaseKGDataset(unittest.TestCase):
    """BaseKGDataset loads one triples table through the BaseDataset backend."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        config = write_kg(cls.root, TOY_TRIPLES)
        cls.dataset = BaseKGDataset(
            root=str(cls.root), config_path=config, cache_dir=cls.root / "cache"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_each_triple_is_one_record(self) -> None:
        self.assertEqual(len(self.dataset.unique_patient_ids), len(TOY_TRIPLES))

    def test_vocabularies_are_global_and_sorted(self) -> None:
        self.assertEqual(self.dataset.entity2id, {"a": 0, "b": 1, "c": 2, "d": 3})
        self.assertEqual(self.dataset.relation2id, {"r1": 0, "r2": 1})
        self.assertEqual(self.dataset.id2entity[3], "d")
        self.assertEqual(self.dataset.id2relation[1], "r2")
        self.assertEqual((self.dataset.num_entities, self.dataset.num_relations), (4, 2))
        # Pre-2.0 names, still read by older code.
        self.assertEqual((self.dataset.entity_num, self.dataset.relation_num), (4, 2))

    def test_legacy_function_task_still_works_with_a_warning(self) -> None:
        with self.assertWarns(DeprecationWarning):
            sample_ds = self.dataset.set_task(link_prediction_fn, negative_sampling=4)
        self.assertIsInstance(sample_ds, SampleKGDataset)
        self.assertEqual(len(sample_ds), len(TOY_TRIPLES))
        self.assertEqual(sample_ds.task_spec_param, {"negative_sampling": 4})
        self.assertEqual(sample_ds.entity_num, 4)

    def test_legacy_task_fn_keyword_is_accepted(self) -> None:
        with self.assertWarns(DeprecationWarning):
            sample_ds = self.dataset.set_task(task_fn=link_prediction_fn)
        self.assertEqual(len(sample_ds), len(TOY_TRIPLES))

    def test_legacy_path_rejects_2_0_arguments(self) -> None:
        from pyhealth.datasets import PatientSplit

        with self.assertRaisesRegex(TypeError, "split"), warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.dataset.set_task(link_prediction_fn, split=PatientSplit((0.5, 0.5)))

    def test_legacy_members_warn(self) -> None:
        with self.assertWarns(DeprecationWarning):
            report = self.dataset.stat()
        self.assertIn("Number of triples: 5", report)
        with self.assertWarns(DeprecationWarning):
            BaseKGDataset.info()
        with self.assertWarns(DeprecationWarning):
            self.assertEqual(len(self.dataset.triples), len(TOY_TRIPLES))


class TestLegacyTripleOrder(unittest.TestCase):
    """The legacy path hands task functions the triples in file order."""

    def test_more_than_ten_triples_keep_file_order(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config = write_kg(root, ORDER_TRIPLES)
            dataset = BaseKGDataset(root=tmp, config_path=config, cache_dir=root / "cache")
            e, r = dataset.entity2id, dataset.relation2id
            expected = [(e[h], r[rel], e[t]) for h, rel, t in ORDER_TRIPLES]
            with self.assertWarns(DeprecationWarning):
                self.assertEqual(dataset.triples, expected)


class TestBaseKGDatasetValidation(unittest.TestCase):
    """Malformed configs and triples are reported, never silently dropped."""

    def test_config_without_the_triple_attributes_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config = write_kg(root, TOY_TRIPLES)
            config.write_text(TOY_CONFIG.replace("[head, relation, tail]", "[head, tail]"))
            with self.assertRaisesRegex(ValueError, r"missing \['relation'\]"):
                BaseKGDataset(root=tmp, config_path=config, cache_dir=root / "cache")

    def test_a_triple_with_a_missing_field_raises(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config = write_kg(root, TOY_TRIPLES + [("a", "r1", "")])
            dataset = BaseKGDataset(root=tmp, config_path=config, cache_dir=root / "cache")
            with self.assertRaisesRegex(ValueError, r"1 triple\(s\) have a missing"):
                dataset.entity2id

    def test_refresh_cache_is_deprecated(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            config = write_kg(root, TOY_TRIPLES)
            with self.assertWarns(DeprecationWarning):
                BaseKGDataset(
                    root=tmp, config_path=config, cache_dir=root / "cache",
                    refresh_cache=True,
                )


class TestUMLSDataset(unittest.TestCase):
    """UMLSDataset reads the headerless graph.txt through a prepared copy."""

    def test_prepared_copy_leaves_graph_txt_untouched(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            raw = root / "graph.txt"
            raw.write_text("".join(f"{h}\t{r}\t{t}\n" for h, r, t in TOY_TRIPLES))
            digest = hashlib.sha256(raw.read_bytes()).hexdigest()

            dataset = UMLSDataset(root=tmp, cache_dir=root / "cache")

            self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), digest)
            prepared = (root / "umls-pyhealth.tsv").read_text().splitlines()
            self.assertEqual(prepared[0], "head\trelation\ttail")
            self.assertEqual(prepared[1:], raw.read_text().splitlines())
            self.assertEqual(dataset.dataset_name, "umls")
            self.assertEqual(dataset.entity2id, {"a": 0, "b": 1, "c": 2, "d": 3})

    def test_a_replaced_graph_txt_is_prepared_again(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            raw = root / "graph.txt"
            raw.write_text("A\tPAR\tB\nB\tCHD\tA\n")
            first = UMLSDataset(root=tmp, cache_dir=root / "cache")
            self.assertEqual(first.num_entities, 2)

            raw.write_text("A\tPAR\tB\nB\tCHD\tA\nC\tRO\tA\n")
            second = UMLSDataset(root=tmp, cache_dir=root / "cache")
            self.assertEqual((second.num_entities, second.num_relations), (3, 3))

    def test_url_root_is_rejected_with_a_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "graph.txt"):
            UMLSDataset(root="https://storage.googleapis.com/pyhealth/umls/")

    def test_missing_graph_is_reported(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            with self.assertRaises(FileNotFoundError):
                UMLSDataset(root=tmp, cache_dir=Path(tmp) / "cache")


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


class TestGroundTruthUnpadding(unittest.TestCase):
    """Regression test for the padding-sentinel collision the Tensor Trick
    introduces: pad_token_id (0) is not a reserved value, so a real entity id
    of 0 must survive unpadding while an actual padding slot does not."""

    def test_unpad_ground_truth_keeps_real_entity_zero_and_drops_padding(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=make_dataset(n=4), e_dim=8, r_dim=8)

        # Row 0: entities {0, 5} are real (mask=1), trailing slot is padding.
        # Row 1: entity {3} is real, two trailing slots are padding.
        value = torch.tensor([[0, 5, 0], [3, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        unpadded = model._unpad_ground_truth({"value": value, "mask": mask})

        self.assertEqual(unpadded, [[0, 5], [3]])

    def test_unpad_ground_truth_passes_through_plain_lists_unchanged(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=make_dataset(n=4), e_dim=8, r_dim=8)
        raw = [[0, 5], [3]]
        self.assertEqual(model._unpad_ground_truth(raw), raw)


def fit_through_builder(samples, split=None):
    """Fit a kg_triple processor the way set_task does, via SampleBuilder."""
    from pyhealth.datasets.sample_dataset import SampleBuilder

    builder = SampleBuilder(
        input_schema={"triple": ("kg_triple", {"num_entities": 6, "num_relations": 2})},
        output_schema={},
    )
    builder.fit(samples, split=split)
    return builder


class TestKGTripleProcessor(unittest.TestCase):
    """The kg_triple processor encodes triples and keeps training-graph dicts."""

    def setUp(self) -> None:
        from pyhealth.processors import KGTripleProcessor

        self.triples = [(0, 0, 1), (0, 0, 2), (3, 1, 1), (4, 0, 2), (5, 1, 0)]
        self.processor = KGTripleProcessor(num_entities=6, num_relations=2)
        self.processor.fit([{"triple": t} for t in self.triples], "triple")

    def test_process_returns_a_long_triple(self) -> None:
        out = self.processor.process((3, 1, 1))
        self.assertEqual(out.dtype, torch.long)
        self.assertEqual(out.tolist(), [3, 1, 1])

    def test_size_is_the_given_entity_count(self) -> None:
        self.assertEqual(self.processor.size(), 6)
        self.assertEqual(self.processor.num_relations, 2)

    def test_dicts_match_the_fitted_triples(self) -> None:
        self.assertEqual(self.processor.true_tail[(0, 0)], [1, 2])
        self.assertEqual(self.processor.true_head[(0, 2)], [0, 4])
        self.assertEqual(self.processor.true_head[(1, 1)], [3])
        self.assertNotIn((1, 0), self.processor.true_tail)

    def test_counts_and_weights_match_link_prediction_fn(self) -> None:
        """Same frequencies and weights as the 1.x task on the same triples."""
        from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks.link_prediction import (
            count_frequency,
        )

        self.assertEqual(self.processor.count, count_frequency(self.triples))
        legacy = [s["subsampling_weight"].item() for s in link_prediction_fn(self.triples)]
        weights = self.processor.subsampling_weight(torch.tensor(self.triples))
        self.assertTrue(torch.allclose(weights, torch.tensor(legacy)))

    def test_weight_of_a_triple_with_an_unseen_pair_raises(self) -> None:
        # (1, 1) never occurs as (head, relation) in the fitted triples.
        with self.assertRaisesRegex(KeyError, "not seen in fit"):
            self.processor.subsampling_weight(torch.tensor([[1, 1, 5]]))

    def test_malformed_and_out_of_range_triples_raise(self) -> None:
        from pyhealth.processors import KGTripleProcessor

        processor = KGTripleProcessor(num_entities=3, num_relations=1)
        for bad in [(0, 0, 3), (0, 1, 1), (-1, 0, 1), (0, 0)]:
            with self.subTest(triple=bad), self.assertRaises(ValueError):
                processor.fit([{"triple": bad}], "triple")
        with self.assertRaises(ValueError):
            KGTripleProcessor(num_entities=0, num_relations=1)

    def test_refit_replaces_the_dicts(self) -> None:
        self.processor.fit([{"triple": (5, 1, 0)}], "triple")
        self.assertEqual(self.processor.true_tail, {(5, 1): [0]})
        self.assertEqual(self.processor.true_head, {(1, 0): [5]})
        self.assertEqual(self.processor.count, {(5, 1): 4, (0, -2): 4})

    def test_a_fitted_processor_can_be_passed_back_to_set_task(self) -> None:
        """set_task puts vars() of pre-fitted processors in a JSON cache key."""
        import json

        from pyhealth.processors import KGTripleProcessor

        def key(processor):
            return json.dumps({"p": vars(processor)}, sort_keys=True, default=str)

        same = KGTripleProcessor(num_entities=6, num_relations=2)
        same.fit([{"triple": t} for t in reversed(self.triples)], "triple")
        other = KGTripleProcessor(num_entities=6, num_relations=2)
        other.fit([{"triple": t} for t in self.triples[:3]], "triple")

        self.assertEqual(key(self.processor), key(same))
        self.assertNotEqual(key(self.processor), key(other))

    def test_survives_pickling(self) -> None:
        import pickle

        clone = pickle.loads(pickle.dumps(self.processor))
        self.assertEqual(clone.true_tail, self.processor.true_tail)
        self.assertEqual(clone.count, self.processor.count)

    def test_split_fits_on_the_training_part_only(self) -> None:
        """No validation or test triple reaches the dicts or the counts."""
        from pyhealth.datasets import PatientSplit

        samples = [
            {"patient_id": str(i), "triple": t} for i, t in enumerate(self.triples)
        ]
        split = PatientSplit(ratios=(0.6, 0.4), seed=0)
        builder = fit_through_builder(samples, split=split)
        processor = builder.input_processors["triple"]
        train_idx, held_out_idx = builder.split_indices

        train = {self.triples[i] for i in train_idx}
        held_out = {self.triples[i] for i in held_out_idx}
        fitted = {(h, r, t) for (h, r), ts in processor.true_tail.items() for t in ts}
        self.assertEqual(fitted, train)
        self.assertTrue(held_out)
        self.assertFalse(fitted & held_out)
        from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks.link_prediction import (
            count_frequency,
        )

        self.assertEqual(processor.count, count_frequency(sorted(train)))

    def test_fitting_without_split_warns(self) -> None:
        samples = [{"patient_id": str(i), "triple": t} for i, t in enumerate(self.triples)]
        with self.assertLogs("pyhealth.datasets.sample_dataset", level="WARNING") as logs:
            fit_through_builder(samples)
        self.assertTrue(any("triple" in line for line in logs.output))


class TestKGProcessorKeepsLongLists(unittest.TestCase):
    """Filter sets longer than the training maximum are kept whole (no truncation)."""

    def setUp(self) -> None:
        from pyhealth.processors import KGProcessor

        self.processor = KGProcessor(pad_token_id=0)
        # Fitted on "training" lists of at most 2 entities.
        self.processor.fit([{"gt": [1, 2]}, {"gt": [3]}], "gt")

    def test_a_longer_list_keeps_all_entities(self) -> None:
        out = self.processor.process([5, 0, 7, 9])
        self.assertEqual(out["value"].tolist(), [5, 0, 7, 9])
        self.assertEqual(out["mask"].tolist(), [1, 1, 1, 1])

    def test_shorter_lists_are_still_padded_to_the_fitted_length(self) -> None:
        out = self.processor.process([4])
        self.assertEqual(out["value"].tolist(), [4, 0])
        self.assertEqual(out["mask"].tolist(), [1, 0])

    def test_collation_pads_the_batch_with_mask_zero(self) -> None:
        batch = [
            {"gt": self.processor.process([5, 0, 7, 9])},
            {"gt": self.processor.process([4])},
        ]
        gt = collate_fn_dict_with_padding(batch)["gt"]
        self.assertEqual(gt["value"].tolist(), [[5, 0, 7, 9], [4, 0, 0, 0]])
        self.assertEqual(gt["mask"].tolist(), [[1, 1, 1, 1], [1, 0, 0, 0]])
        # The model's unpadding recovers the exact lists, entity 0 included.
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=make_dataset(n=4), e_dim=8, r_dim=8)
        self.assertEqual(model._unpad_ground_truth(gt), [[5, 0, 7, 9], [4]])


# Twelve rows, one of them a duplicate of the first: 11 distinct triples
# over entities a..f (f appears once, so a split can leave it out of training).
TASK_TRIPLES = [
    ("a", "r1", "b"),
    ("a", "r1", "c"),
    ("b", "r1", "c"),
    ("c", "r2", "a"),
    ("d", "r2", "a"),
    ("a", "r2", "d"),
    ("b", "r2", "d"),
    ("e", "r1", "a"),
    ("a", "r1", "b"),
    ("c", "r1", "e"),
    ("d", "r1", "b"),
    ("e", "r2", "f"),
]
TASK_SPLIT_RATIOS = (0.6, 0.2, 0.2)


def unpad(field: dict[str, torch.Tensor]) -> list[int]:
    """The real entities of one processed kg_entity_list field."""
    return field["value"][field["mask"].bool()].tolist()


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TestKGLinkPrediction(unittest.TestCase):
    """KGLinkPrediction through the real set_task, with a triple-level split."""

    @classmethod
    def setUpClass(cls) -> None:
        from pyhealth.datasets import PatientSplit
        from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import KGLinkPrediction

        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls._tmp.name)
        config = write_kg(cls.root, TASK_TRIPLES)
        cls.dataset = BaseKGDataset(
            root=str(cls.root), config_path=config, cache_dir=cls.root / "cache"
        )
        cls.task = KGLinkPrediction(
            num_entities=cls.dataset.num_entities,
            num_relations=cls.dataset.num_relations,
        )
        cls.split = PatientSplit(ratios=TASK_SPLIT_RATIOS, seed=3)
        handler = _ListHandler()
        kg_logger = logging.getLogger("pyhealth.medcode.pretrained_embeddings.kg_emb")
        kg_logger.addHandler(handler)
        kg_logger.setLevel(logging.INFO)
        try:
            cls.parts = cls.dataset.set_task(cls.task, split=cls.split, num_workers=1)
        finally:
            kg_logger.removeHandler(handler)
        cls.log = handler.messages

        e, r = cls.dataset.entity2id, cls.dataset.relation2id
        cls.distinct = sorted({(e[h], r[rel], e[t]) for h, rel, t in TASK_TRIPLES})

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def triples_of(self, part) -> list[tuple[int, int, int]]:
        return [tuple(part[i]["triple"].tolist()) for i in range(len(part))]

    def test_parts_are_streaming_sample_datasets(self) -> None:
        from pyhealth.datasets import SampleDataset
        from pyhealth.datasets.sample_dataset import InMemorySampleDataset

        self.assertEqual(len(self.parts), 3)
        for part in self.parts:
            self.assertIsInstance(part, SampleDataset)
            self.assertNotIsInstance(part, InMemorySampleDataset)

    def test_parts_partition_the_distinct_triples(self) -> None:
        triples = [t for part in self.parts for t in self.triples_of(part)]
        self.assertEqual(sorted(triples), self.distinct)
        self.assertEqual(len(triples), 11)

    def test_duplicates_are_removed_and_logged(self) -> None:
        self.assertTrue(any("Removed 1 duplicate" in m for m in self.log), self.log)

    def test_the_first_duplicate_in_file_order_is_kept(self) -> None:
        # (a, r1, b) is on rows 0 and 8; row 0 must be the record kept, since
        # the record id decides which part the triple falls into.
        e, r = self.dataset.entity2id, self.dataset.relation2id
        duplicate = (e["a"], r["r1"], e["b"])
        kept = [
            part[i]["patient_id"]
            for part in self.parts
            for i in range(len(part))
            if tuple(part[i]["triple"].tolist()) == duplicate
        ]
        self.assertEqual(kept, ["0"])

    def test_ground_truth_covers_the_whole_graph(self) -> None:
        for part in self.parts:
            for i in range(len(part)):
                sample = part[i]
                h, r, t = sample["triple"].tolist()
                with self.subTest(triple=(h, r, t)):
                    self.assertEqual(
                        unpad(sample["ground_truth_head"]),
                        sorted(x for x, y, z in self.distinct if (y, z) == (r, t)),
                    )
                    self.assertEqual(
                        unpad(sample["ground_truth_tail"]),
                        sorted(z for x, y, z in self.distinct if (x, y) == (h, r)),
                    )

    def test_triple_processor_is_fitted_on_training_triples_only(self) -> None:
        processor = self.parts[0].input_processors["triple"]
        fitted = {(h, r, t) for (h, r), ts in processor.true_tail.items() for t in ts}
        self.assertEqual(fitted, set(self.triples_of(self.parts[0])))
        held_out = set(self.triples_of(self.parts[1]) + self.triples_of(self.parts[2]))
        self.assertTrue(held_out)
        self.assertFalse(fitted & held_out)
        self.assertEqual(processor.size(), self.dataset.num_entities)
        self.assertEqual(processor.num_relations, self.dataset.num_relations)

    def test_unseen_entities_are_logged_per_held_out_part(self) -> None:
        processor = self.parts[0].input_processors["triple"]
        seen = {h for h, _ in processor.true_tail} | {
            t for ts in processor.true_tail.values() for t in ts
        }
        for k in (1, 2):
            expected = sum(
                h not in seen or t not in seen for h, _, t in self.triples_of(self.parts[k])
            )
            prefix = f"Part {k}: {expected} of {len(self.parts[k])} triples"
            self.assertTrue(any(m.startswith(prefix) for m in self.log), self.log)

    def test_two_workers_give_the_same_samples(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dataset = BaseKGDataset(
                root=str(self.root),
                config_path=self.root / "kg.yaml",
                cache_dir=Path(tmp),
                num_workers=2,
            )
            parts = dataset.set_task(self.task, split=self.split, num_workers=2)

            def content(part):
                return sorted(
                    (
                        tuple(s["triple"].tolist()),
                        tuple(unpad(s["ground_truth_head"])),
                        tuple(unpad(s["ground_truth_tail"])),
                    )
                    for s in (part[i] for i in range(len(part)))
                )

            for mine, theirs in zip(self.parts, parts):
                self.assertEqual(content(mine), content(theirs))

    def test_without_split_warns_at_every_call(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            dataset = BaseKGDataset(
                root=str(self.root), config_path=self.root / "kg.yaml", cache_dir=Path(tmp)
            )
            with self.assertWarnsRegex(UserWarning, "split=PatientSplit"):
                samples = dataset.set_task(self.task)
            self.assertEqual(len(samples), 11)
            # Second call: the processed cache exists, the warning still fires.
            with self.assertWarnsRegex(UserWarning, "split=PatientSplit"):
                dataset.set_task(self.task)
            # Without a task, the default KGLinkPrediction is used, and warned about.
            with self.assertWarnsRegex(UserWarning, "split=PatientSplit"):
                dataset.set_task()

    def test_default_task_matches_the_graph(self) -> None:
        task = self.dataset.default_task
        self.assertEqual(
            (task.num_entities, task.num_relations),
            (self.dataset.num_entities, self.dataset.num_relations),
        )

    def test_a_task_for_another_graph_is_rejected(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import KGLinkPrediction

        task = KGLinkPrediction(num_entities=99, num_relations=2)
        with self.assertRaisesRegex(ValueError, "expects 99 entities"):
            self.dataset.set_task(task, split=self.split)

    def test_supplied_processors_are_used_as_they_are(self) -> None:
        """A supplied kg_triple processor is neither refitted nor warned about."""
        from pyhealth.processors import KGTripleProcessor

        everything = KGTripleProcessor(
            num_entities=self.dataset.num_entities,
            num_relations=self.dataset.num_relations,
        )
        everything.fit([{"triple": t} for t in self.distinct], "triple")
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            parts = self.dataset.set_task(
                self.task, split=self.split, input_processors={"triple": everything}
            )
            whole = self.dataset.set_task(
                self.task, input_processors={"triple": everything}
            )
        # Fitted on the 11 triples, not refitted on the 6 training ones.
        for result in (parts[0], whole):
            self.assertEqual(result.input_processors["triple"].true_tail, everything.true_tail)


class TestUnseenInTrainingLog(unittest.TestCase):
    """The per-part count of triples whose entity or relation training never saw."""

    def test_counts_heads_tails_and_relations(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import base_kg_dataset
        from pyhealth.processors import KGTripleProcessor

        processor = KGTripleProcessor(num_entities=6, num_relations=3)
        processor.fit([{"triple": (0, 0, 1)}, {"triple": (1, 1, 2)}], "triple")

        class _Train(list):
            input_processors = {"triple": processor}

        def part(*triples):
            return [{"triple": torch.tensor(t)} for t in triples]

        parts = (
            _Train(),
            part((0, 0, 2), (4, 0, 1)),  # seen; unseen head
            part((0, 0, 5), (1, 2, 2), (3, 2, 4)),  # unseen tail; unseen relation; both
        )
        with self.assertLogs(base_kg_dataset.logger, level="INFO") as logs:
            base_kg_dataset._log_unseen_in_training(parts)
        self.assertEqual(
            [line.split(":", 2)[2] for line in logs.output],
            [
                "Part 1: 1 of 2 triples involve an entity absent from the training "
                "triples, 0 a relation absent from them.",
                "Part 2: 2 of 3 triples involve an entity absent from the training "
                "triples, 2 a relation absent from them.",
            ],
        )


class TestModelsOnSampleDatasets(unittest.TestCase):
    """The KGE models consume the SampleDatasets returned by set_task."""

    @classmethod
    def setUpClass(cls) -> None:
        from pyhealth.datasets import PatientSplit
        from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import KGLinkPrediction

        cls._tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls._tmp.name)
        config = write_kg(root, TASK_TRIPLES)
        cls.dataset = BaseKGDataset(root=str(root), config_path=config, cache_dir=root / "cache")
        task = KGLinkPrediction(
            num_entities=cls.dataset.num_entities, num_relations=cls.dataset.num_relations
        )
        cls.train, cls.val, cls.test = cls.dataset.set_task(
            task, split=PatientSplit(ratios=TASK_SPLIT_RATIOS, seed=3)
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def batch(self, part, size=4):
        from pyhealth.datasets import get_dataloader

        return next(iter(get_dataloader(part, batch_size=size)))

    def test_counts_come_from_the_fitted_processor(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import (
            ComplEx,
            DistMult,
            RotatE,
            TransE,
        )

        for cls in (TransE, RotatE, DistMult, ComplEx):
            with self.subTest(model=cls.__name__):
                model = cls(dataset=self.train, e_dim=8, r_dim=4 if cls is RotatE else 8)
                self.assertEqual(model.e_num, self.dataset.num_entities)
                self.assertEqual(model.r_num, self.dataset.num_relations)
                self.assertIs(model.triple_processor, self.train.input_processors["triple"])

    def test_negative_sampling_is_a_model_argument(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        self.assertEqual(TransE(dataset=self.train, e_dim=8, r_dim=8).negative_sampling, 128)
        model = TransE(dataset=self.train, e_dim=8, r_dim=8, negative_sampling=5)
        self.assertEqual(model.negative_sampling, 5)
        # A 1.x SampleKGDataset still provides its own value.
        self.assertEqual(TransE(dataset=make_dataset(n=4), e_dim=8, r_dim=8).negative_sampling, 4)
        with self.assertRaises(ValueError):
            TransE(dataset=self.train, e_dim=8, r_dim=8, negative_sampling=0)

    def test_train_mode_returns_a_loss_and_draws_negatives(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        torch.manual_seed(0)
        model = TransE(dataset=self.train, e_dim=8, r_dim=8, negative_sampling=5)
        model.train()
        out = model(**self.batch(self.train))
        self.assertEqual(set(out), {"loss"})
        out["loss"].backward()
        self.assertIsNotNone(model.E_emb.grad)

    def test_subsampling_weights_come_from_the_processor(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=self.train, e_dim=8, r_dim=8, use_subsampling_weight=True)
        batch = self.batch(self.train)
        weights = model._subsampling_weight(batch, batch["triple"])
        expected = self.train.input_processors["triple"].subsampling_weight(batch["triple"])
        self.assertTrue(torch.equal(weights, expected))
        model.train()
        model(**batch)["loss"].backward()

    def test_eval_mode_ranks_every_entity(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=self.train, e_dim=8, r_dim=8)
        model.eval()
        batch = self.batch(self.test, size=len(self.test))
        with torch.no_grad():
            out = model(**batch)
        n = len(self.test)
        self.assertEqual(tuple(out["y_prob"].shape), (2 * n, self.dataset.num_entities))
        self.assertEqual(out["y_true"].tolist(), batch["triple"][:, 0].tolist() + batch["triple"][:, 2].tolist())

    def test_a_disagreeing_1x_train_flag_warns(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        train, _, _ = split(make_dataset(n=4), [1.0, 0.0, 0.0], seed=0)
        batch = collate_fn_dict_with_padding(train[:2])
        model = TransE(dataset=make_dataset(n=4), e_dim=8, r_dim=8)
        model.eval()
        with self.assertWarnsRegex(UserWarning, "'train' flag is ignored"), torch.no_grad():
            out = model(**batch)
        self.assertIn("y_prob", out)  # model.eval() decides, not the flag


class TestTrainingNegativesUseTrainingTriplesOnly(unittest.TestCase):
    """Regression test for the leak: training negatives were filtered with
    ground truths of the whole graph, so held-out positives were never drawn."""

    def test_a_tail_true_only_in_test_can_be_a_training_negative(self) -> None:
        import numpy as np

        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE
        from pyhealth.processors import KGTripleProcessor

        # Training triple (0, 0, 1); (0, 0, 2) is a test triple.
        processor = KGTripleProcessor(num_entities=3, num_relations=1)
        processor.fit([{"triple": (0, 0, 1)}], "triple")

        class _Train:
            input_processors = {"triple": processor}

        model = TransE(dataset=_Train(), e_dim=4, r_dim=4, negative_sampling=64)
        model.train()
        # The sample's ground truth covers the whole graph, as KGLinkPrediction gives it.
        batch = {
            "triple": torch.tensor([[0, 0, 1]]),
            "ground_truth_head": [[0]],
            "ground_truth_tail": [[1, 2]],
        }
        drawn = {}
        original = model.train_neg_sample_gen

        def record(**kwargs):
            drawn["head"], drawn["tail"] = original(**kwargs)
            return drawn["head"], drawn["tail"]

        model.train_neg_sample_gen = record
        np.random.seed(0)
        model(**batch)["loss"].backward()

        tails = set(drawn["tail"].flatten().tolist())
        self.assertNotIn(1, tails)  # the training positive stays filtered
        self.assertIn(2, tails)  # the test positive is a legitimate negative
        self.assertNotIn(0, set(drawn["head"].flatten().tolist()))


class TestTrainingNegativesOnARealSplit(unittest.TestCase):
    """The same property on set_task output: (b, r2, d) is a test triple and
    (a, r2, d) a training one, so b must stay drawable as a head negative."""

    # Same graph and split as TestModelsOnSampleDatasets, without its tests.
    setUpClass = classmethod(TestModelsOnSampleDatasets.setUpClass.__func__)
    tearDownClass = classmethod(TestModelsOnSampleDatasets.tearDownClass.__func__)

    def test_held_out_heads_are_not_filtered(self) -> None:
        import numpy as np

        from pyhealth.datasets import get_dataloader
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        e, r = self.dataset.entity2id, self.dataset.relation2id
        target = [e["a"], r["r2"], e["d"]]
        batch = next(
            b for b in get_dataloader(self.train, batch_size=1)
            if b["triple"][0].tolist() == target
        )
        test_triples = {tuple(self.test[i]["triple"].tolist()) for i in range(len(self.test))}
        self.assertIn((e["b"], r["r2"], e["d"]), test_triples)
        self.assertEqual(sorted(unpad({k: v[0] for k, v in batch["ground_truth_head"].items()})),
                         [e["a"], e["b"]])

        model = TransE(dataset=self.train, e_dim=4, r_dim=4, negative_sampling=256)
        model.train()
        gt_head, _ = model._training_filters(batch, batch["triple"])
        self.assertEqual(gt_head, [[e["a"]]])
        np.random.seed(0)
        heads, _ = model.train_neg_sample_gen(gt_head, [[]], 256)
        self.assertIn(e["b"], heads.flatten().tolist())
        self.assertNotIn(e["a"], heads.flatten().tolist())


class TestDeprecatedEntryPoints(unittest.TestCase):
    """The 1.x entry points keep their import paths and work, with a warning."""

    def test_old_import_paths_resolve(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb import datasets, tasks

        for name in ("BaseKGDataset", "SampleKGDataset", "UMLSDataset", "split"):
            self.assertTrue(hasattr(datasets, name), name)
        self.assertTrue(callable(tasks.link_prediction_fn))

    def test_link_prediction_fn_warns(self) -> None:
        with self.assertWarns(DeprecationWarning):
            samples = link_prediction_fn([(0, 0, 1)])
        self.assertEqual(len(samples), 1)

    def test_sample_kg_dataset_warns(self) -> None:
        with self.assertWarns(DeprecationWarning):
            make_dataset(n=2)

    def test_split_warns(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dataset = make_dataset(n=10)
        with self.assertWarns(DeprecationWarning):
            split(dataset, [0.6, 0.2, 0.2], seed=0)

    def test_a_model_on_a_1x_dataset_warns(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dataset = make_dataset(n=4)
        with self.assertWarnsRegex(DeprecationWarning, "kg_triple"):
            TransE(dataset=dataset, e_dim=4, r_dim=4)


class TestNegativeSamplingReachesTheSampler(unittest.TestCase):
    """The number of negatives drawn is the one the caller chose."""

    def spy(self, model):
        shapes = []
        original = model.train_neg_sample_gen

        def record(**kwargs):
            out = original(**kwargs)
            shapes.append(tuple(out[0].shape))
            return out

        model.train_neg_sample_gen = record
        return shapes

    def legacy_batch(self, per_sample=None, **dataset_kwargs):
        samples = make_samples(4)
        if per_sample is not None:
            for s in samples:
                s["hyperparameters"] = {"negative_sampling": per_sample}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            dataset = SampleKGDataset(
                samples=samples, entity_num=10, relation_num=3, **dataset_kwargs
            )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn_dict_with_padding)
        return dataset, next(iter(loader))

    def build(self, dataset, **kwargs):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = TransE(dataset=dataset, e_dim=4, r_dim=4, **kwargs)
        model.train()
        return model

    def test_the_model_argument_sets_the_sample_count(self) -> None:
        from pyhealth.processors import KGTripleProcessor

        processor = KGTripleProcessor(num_entities=10, num_relations=3)
        processor.fit([{"triple": s["triple"]} for s in make_samples(4)], "triple")

        class _Train:
            input_processors = {"triple": processor}

        model = self.build(_Train(), negative_sampling=7)
        shapes = self.spy(model)
        model(triple=torch.tensor([[0, 0, 4], [1, 1, 5]]))
        self.assertEqual(shapes, [(2, 7)])

    def test_numpy_integers_are_accepted(self) -> None:
        import numpy as np

        dataset, _ = self.legacy_batch()
        self.assertEqual(self.build(dataset, negative_sampling=np.int64(6)).negative_sampling, 6)
        dataset, _ = self.legacy_batch(negative_sampling=np.int32(5))
        self.assertEqual(self.build(dataset).negative_sampling, 5)
        with self.assertRaises(ValueError):
            self.build(dataset, negative_sampling=True)

    def test_a_1x_per_sample_value_applies_when_none_was_chosen(self) -> None:
        dataset, batch = self.legacy_batch(per_sample=3)
        model = self.build(dataset)
        shapes = self.spy(model)
        model(**batch)
        self.assertEqual(shapes, [(2, 3)])

    def test_a_chosen_value_wins_over_the_samples_with_a_warning(self) -> None:
        dataset, batch = self.legacy_batch(per_sample=3)
        model = self.build(dataset, negative_sampling=5)
        shapes = self.spy(model)
        with self.assertWarnsRegex(UserWarning, "negative_sampling=3 is ignored"):
            model(**batch)
        self.assertEqual(shapes, [(2, 5)])
