"""Regression tests for ``pyhealth.medcode.pretrained_embeddings.kg_emb``.

The suite is behavioural: it exercises construction, indexing and splitting
rather than asserting on type annotations, which are metadata and not a
contract.
"""

from __future__ import annotations

import hashlib
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
