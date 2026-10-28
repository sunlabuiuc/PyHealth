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

from pyhealth.datasets import collate_fn_dict_with_padding
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import (
    BaseKGDataset,
    UMLSDataset,
)

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


def processor_dataset(num_entities: int, num_relations: int) -> Any:
    """Stands in for a set_task part: the models only read its triple processor."""
    from types import SimpleNamespace

    from pyhealth.processors import KGTripleProcessor

    processor = KGTripleProcessor(num_entities=num_entities, num_relations=num_relations)
    return SimpleNamespace(input_processors={"triple": processor})


class TestKGEmbImports(unittest.TestCase):
    """The module must import cleanly. This was the original symptom of issue #952."""

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

    def test_a_concurrent_up_to_date_copy_is_accepted(self) -> None:
        # Windows refuses os.replace while another job replaces the same file.
        from unittest import mock

        from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import umls

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "graph.txt").write_text("A\tPAR\tB\n")
            UMLSDataset.prepare_metadata(tmp)  # the other job's copy
            busy = PermissionError("the target is being replaced")
            with mock.patch.object(umls.os, "replace", side_effect=busy):
                UMLSDataset.prepare_metadata(tmp)
            self.assertEqual(
                sorted(f.name for f in root.iterdir()), ["graph.txt", "umls-pyhealth.tsv"]
            )

    def test_a_failed_replacement_without_an_up_to_date_copy_raises(self) -> None:
        from unittest import mock

        from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import umls

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            root = Path(tmp)
            (root / "graph.txt").write_text("A\tPAR\tB\n")
            busy = PermissionError("the target is being replaced")
            replace = mock.patch.object(umls.os, "replace", side_effect=busy)
            with replace, self.assertRaises(PermissionError):
                UMLSDataset.prepare_metadata(tmp)
            self.assertEqual([f.name for f in root.iterdir()], ["graph.txt"])

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

        model = DistMult(dataset=processor_dataset(5, 2), e_dim=8, r_dim=8, gamma=12.0)
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

        model = TransE(dataset=processor_dataset(5, 2), e_dim=8, r_dim=8, gamma=24.0)
        head = torch.zeros(1, 1, 8)
        relation = torch.ones(1, 1, 8)
        tail = torch.ones(1, 1, 8)  # h + r - t == 0
        self.assertTrue(
            torch.allclose(
                model.calc(head, relation, tail), torch.tensor(24.0), atol=1e-6
            )
        )


class TestScoringReferenceValues(unittest.TestCase):
    """Each score function on hand-set embeddings, against its formula worked by hand.

    The values are chosen so that a sign error, a missing conjugate or a wrong
    norm gives a different number. Shapes are (batch, candidates, dim).
    """

    def check(self, model, head, relation, tail, expected) -> None:
        h, r, t = (torch.tensor([[x]]) for x in (head, relation, tail))
        for mode in ("pos", "head", "tail"):
            with self.subTest(model=type(model).__name__, mode=mode):
                self.assertAlmostEqual(model.calc(h, r, t, mode=mode).item(), expected, places=5)

    def test_transe(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        # gamma - ||h + r - t||_1 = 24 - (|1 + 0.5 - 2| + |2 + 0.25 - 1|) = 24 - 1.75;
        # a negative component, so a norm without absolute values differs.
        model = TransE(processor_dataset(2, 1), e_dim=2, r_dim=2)
        self.check(model, [1.0, 2.0], [0.5, 0.25], [2.0, 1.0], 22.25)

    def test_distmult(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import DistMult

        # sum(h * r * t) = 1 * 3 * 0.5 + 2 * (-1) * 4
        model = DistMult(processor_dataset(2, 1), e_dim=2, r_dim=2)
        self.check(model, [1.0, 2.0], [3.0, -1.0], [0.5, 4.0], -6.5)

    def test_complex(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import ComplEx

        # Re(h * r * conj(t)) with h = 1 + 2i, r = 3 + 5i, t = 7 + 11i:
        # Re((-7 + 11i)(7 - 11i)) = 72, every real and imaginary part being
        # non-zero. Without the conjugate it would be -170.
        model = ComplEx(processor_dataset(2, 1), e_dim=2, r_dim=2)
        self.check(model, [1.0, 2.0], [3.0, 5.0], [7.0, 11.0], 72.0)

    def test_rotate(self) -> None:
        import cmath
        import math

        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import RotatE

        # gamma - |h * exp(i theta) - t| with h = 1 and theta = pi / 2, the phase
        # being relation / (r_emb_range / pi): t = i scores 24, t = 1 scores 24 - sqrt 2.
        model = RotatE(processor_dataset(2, 1), e_dim=2, r_dim=1)
        quarter_turn = [model.r_emb_range.item() / 2]
        self.check(model, [1.0, 0.0], quarter_turn, [0.0, 1.0], 24.0)
        self.check(model, [1.0, 0.0], quarter_turn, [1.0, 0.0], 24.0 - math.sqrt(2))
        # A general case, with no zero part: h = 1 + 2i, theta = pi / 4, t = 3 - i.
        eighth_turn = [model.r_emb_range.item() / 4]
        rotated = complex(1, 2) * cmath.exp(1j * math.pi / 4)
        self.check(model, [1.0, 2.0], eighth_turn, [3.0, -1.0], 24.0 - abs(rotated - complex(3, -1)))


class TestGroundTruthUnpadding(unittest.TestCase):
    """Regression test for the padding-sentinel collision the Tensor Trick
    introduces: pad_token_id (0) is not a reserved value, so a real entity id
    of 0 must survive unpadding while an actual padding slot does not."""

    def test_unpad_ground_truth_keeps_real_entity_zero_and_drops_padding(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=processor_dataset(5, 2), e_dim=8, r_dim=8)

        # Row 0: entities {0, 5} are real (mask=1), trailing slot is padding.
        # Row 1: entity {3} is real, two trailing slots are padding.
        value = torch.tensor([[0, 5, 0], [3, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        unpadded = model._unpad_ground_truth({"value": value, "mask": mask})

        self.assertEqual(unpadded, [[0, 5], [3]])

    def test_unpad_ground_truth_passes_through_plain_lists_unchanged(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=processor_dataset(5, 2), e_dim=8, r_dim=8)
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

    def test_counts_and_weights_follow_the_reference_formula(self) -> None:
        """Worked by hand: a pair's count starts at 4 and grows by one per
        further triple, and a triple weighs 1 / sqrt(count[(h, r)] +
        count[(t, -r - 1)]), as in the reference implementation of RotatE."""
        self.assertEqual(
            self.processor.count,
            {(0, 0): 5, (1, -1): 4, (2, -1): 5, (3, 1): 4,
             (1, -2): 4, (4, 0): 4, (5, 1): 4, (0, -2): 4},
        )
        weights = self.processor.subsampling_weight(torch.tensor(self.triples))
        expected = torch.tensor([9.0, 10.0, 8.0, 9.0, 8.0]).rsqrt()
        self.assertTrue(torch.allclose(weights, expected))

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
        from pyhealth.processors import KGTripleProcessor

        on_train_only = KGTripleProcessor(num_entities=6, num_relations=2)
        on_train_only.fit([{"triple": t} for t in sorted(train)], "triple")
        self.assertEqual(processor.count, on_train_only.count)

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

        model = TransE(dataset=processor_dataset(5, 2), e_dim=8, r_dim=8)
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


_TASK_SPLIT: dict[str, Any] = {}


def task_split() -> dict[str, Any]:
    """set_task on TASK_TRIPLES with a triple-level split, built once.

    Tests may call set_task on the shared dataset again: another task or other
    processors get their own cache entry, so the parts held here are never
    rewritten. Tests must not refit or modify the parts' processors. The log
    records are captured while it is built, and the kg_emb logger is left as
    it was found.
    """
    if not _TASK_SPLIT:
        from pyhealth.datasets import PatientSplit
        from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import KGLinkPrediction

        tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(tmp.name)
        config = write_kg(root, TASK_TRIPLES)
        dataset = BaseKGDataset(root=str(root), config_path=config, cache_dir=root / "cache")
        task = KGLinkPrediction(
            num_entities=dataset.num_entities, num_relations=dataset.num_relations
        )
        split = PatientSplit(ratios=TASK_SPLIT_RATIOS, seed=3)
        handler = _ListHandler()
        kg_logger = logging.getLogger("pyhealth.medcode.pretrained_embeddings.kg_emb")
        level = kg_logger.level
        kg_logger.addHandler(handler)
        kg_logger.setLevel(logging.INFO)
        try:
            parts = dataset.set_task(task, split=split, num_workers=1)
        finally:
            kg_logger.removeHandler(handler)
            kg_logger.setLevel(level)
        _TASK_SPLIT.update(
            tmp=tmp, root=root, dataset=dataset, task=task, split=split,
            parts=parts, log=handler.messages,
        )
    return _TASK_SPLIT


def tearDownModule() -> None:
    if _TASK_SPLIT:
        _TASK_SPLIT["tmp"].cleanup()


class TestKGLinkPrediction(unittest.TestCase):
    """KGLinkPrediction through the real set_task, with a triple-level split."""

    @classmethod
    def setUpClass(cls) -> None:
        shared = task_split()
        cls.root, cls.dataset, cls.task = shared["root"], shared["dataset"], shared["task"]
        cls.split, cls.parts, cls.log = shared["split"], shared["parts"], shared["log"]

        e, r = cls.dataset.entity2id, cls.dataset.relation2id
        cls.distinct = sorted({(e[h], r[rel], e[t]) for h, rel, t in TASK_TRIPLES})

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
        shared = task_split()
        cls.dataset = shared["dataset"]
        cls.train, cls.val, cls.test = shared["parts"]

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

    def test_every_model_trains_and_evaluates(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import (
            ComplEx,
            DistMult,
            RotatE,
            TransE,
        )

        train_batch = self.batch(self.train)
        test_batch = self.batch(self.test, size=len(self.test))
        for cls in (TransE, RotatE, DistMult, ComplEx):
            with self.subTest(model=cls.__name__):
                torch.manual_seed(0)
                model = cls(
                    dataset=self.train, e_dim=8, r_dim=4 if cls is RotatE else 8,
                    negative_sampling=4,
                )
                model.train()
                loss = model(**train_batch)["loss"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.E_emb.grad).all())

                model.eval()
                with torch.no_grad():
                    y_prob = model(**test_batch)["y_prob"]
                n = len(self.test)
                self.assertEqual(tuple(y_prob.shape), (2 * n, self.dataset.num_entities))
                self.assertTrue(torch.isfinite(y_prob).all())

    def test_uniform_negative_weighting_trains(self) -> None:
        # ns="uniform" is valid for DistMult, whose score does not depend on ns:
        # it reaches the loss branch without self-adversarial weighting.
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import DistMult

        torch.manual_seed(0)
        model = DistMult(dataset=self.train, e_dim=8, r_dim=8, ns="uniform", negative_sampling=4)
        model.train()
        loss = model(**self.batch(self.train))["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_negative_sampling_is_a_model_argument(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        self.assertEqual(TransE(dataset=self.train, e_dim=8, r_dim=8).negative_sampling, 128)
        model = TransE(dataset=self.train, e_dim=8, r_dim=8, negative_sampling=5)
        self.assertEqual(model.negative_sampling, 5)
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

    def test_training_does_not_call_np_in1d(self) -> None:
        # np.in1d no longer exists in NumPy 2.4, which pyproject.toml allows.
        from unittest import mock

        import numpy as np

        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=self.train, e_dim=8, r_dim=8, negative_sampling=5)
        model.train()
        removed = AssertionError("np.in1d was removed in NumPy 2.4")
        with mock.patch.object(np, "in1d", side_effect=removed, create=True):
            model(**self.batch(self.train))["loss"].backward()

    def test_subsampling_weights_come_from_the_processor(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=self.train, e_dim=8, r_dim=8, use_subsampling_weight=True)
        batch = self.batch(self.train)
        weights = model._subsampling_weight(batch["triple"])
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


class TestFilteredRankingByHand(unittest.TestCase):
    """forward in eval mode, then the rank metrics, on a TransE scored by hand.

    A filtered candidate must fall below the true entity, and the padding of a
    kg_entity_list field must not be taken for entity 0.
    """

    def test_filtered_scores_ranks_and_mrr(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE
        from pyhealth.metrics import multiclass_metrics_fn

        model = TransE(processor_dataset(4, 1), e_dim=1, r_dim=1, gamma=10.0)
        with torch.no_grad():
            model.E_emb.copy_(torch.tensor([[0.0], [1.2], [2.5], [4.0]]))
            model.R_emb.copy_(torch.tensor([[1.0]]))
        model.eval()
        # Both rows rank (0, r, 2); the score is 10 - |h + 1 - t|.
        #   candidate heads, t = 2.5: 8.5 (true), 9.7, 9.0, 7.5
        #   candidate tails, h = 0:   9.0, 9.8, 8.5 (true), 7.0
        # Row 0: heads [0] and tails [2, 1], each followed by a padding 0.
        # Row 1: entity 0 is a real member of both lists.
        batch = {
            "triple": torch.tensor([[0, 0, 2], [0, 0, 2]]),
            "ground_truth_head": {
                "value": torch.tensor([[0, 0], [0, 2]]),
                "mask": torch.tensor([[1, 0], [1, 1]]),
            },
            "ground_truth_tail": {
                "value": torch.tensor([[2, 1, 0], [2, 1, 0]]),
                "mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
            },
        }
        with torch.no_grad():
            out = model(**batch)

        # A filtered candidate is scored as the true entity minus 1.
        expected = torch.tensor([
            [8.5, 9.7, 9.0, 7.5],  # head, row 0: nothing to filter
            [8.5, 9.7, 7.5, 7.5],  # head, row 1: head 2 filtered
            [9.0, 7.5, 8.5, 7.0],  # tail, row 0: tail 1 filtered, the padding 0 is not
            [7.5, 7.5, 8.5, 7.0],  # tail, row 1: tails 1 and 0 filtered
        ])
        self.assertTrue(torch.allclose(out["y_prob"], expected, atol=1e-5), out["y_prob"])
        self.assertEqual(out["y_true"].tolist(), [0, 0, 2, 2])

        scores = multiclass_metrics_fn(
            out["y_true"].numpy(), out["y_prob"].numpy(), metrics=["mean_rank", "hits@n"]
        )
        # Filtered ranks 3, 2, 2, 1; unfiltered they would be 3, 3, 3, 3.
        self.assertAlmostEqual(scores["mean_rank"], 2.0)
        self.assertAlmostEqual(scores["mean_reciprocal_rank"], (1 / 3 + 1 / 2 + 1 / 2 + 1) / 4)
        self.assertAlmostEqual(scores["HITS@1"], 0.25)


def record_negatives(model) -> dict:
    """Wraps train_neg_sample_gen to keep what forward draws."""
    drawn = {}
    original = model.train_neg_sample_gen

    def record(**kwargs):
        drawn["head"], drawn["tail"] = original(**kwargs)
        return drawn["head"], drawn["tail"]

    model.train_neg_sample_gen = record
    return drawn


class TestTrainingNegativesUseTrainingTriplesOnly(unittest.TestCase):
    """Regression test for the leak: training negatives were filtered with
    ground truths of the whole graph, so held-out positives were never drawn."""

    def test_held_out_positives_are_drawable_and_training_ones_are_not(self) -> None:
        import numpy as np

        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE
        from pyhealth.processors import KGTripleProcessor

        # Training: (0, 0, 1), (0, 0, 3), (4, 0, 1). Test: (0, 0, 2), (5, 0, 1).
        processor = KGTripleProcessor(num_entities=6, num_relations=1)
        processor.fit(
            [{"triple": t} for t in [(0, 0, 1), (0, 0, 3), (4, 0, 1)]], "triple"
        )

        class _Train:
            input_processors = {"triple": processor}

        model = TransE(dataset=_Train(), e_dim=4, r_dim=4, negative_sampling=256)
        model.train()
        # Ground truths of the whole graph, as KGLinkPrediction gives them.
        batch = {
            "triple": torch.tensor([[0, 0, 1]]),
            "ground_truth_head": [[0, 4, 5]],
            "ground_truth_tail": [[1, 2, 3]],
        }
        drawn = record_negatives(model)
        np.random.seed(0)
        model(**batch)["loss"].backward()

        # Every training positive of the pair stays out (reference
        # semantics); the test-only positives 2 and 5 are drawable.
        self.assertEqual(set(drawn["tail"].flatten().tolist()), {0, 2, 4, 5})
        self.assertEqual(set(drawn["head"].flatten().tolist()), {1, 2, 3, 5})

    def test_negatives_are_drawn_with_replacement(self) -> None:
        # With assume_unique=True, the repeated draws 5 and 7 would be dropped:
        # for these sizes NumPy takes its sort path, where the flag matters.
        from types import SimpleNamespace
        from unittest import mock

        import numpy as np

        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE
        from pyhealth.processors import KGTripleProcessor

        processor = KGTripleProcessor(num_entities=2000, num_relations=1)
        part = SimpleNamespace(input_processors={"triple": processor})
        model = TransE(dataset=part, e_dim=2, r_dim=2)
        known = list(range(100, 2000, 100))
        draws = np.array([5, 5, 7, 7, 100, 200, 8, 9])
        with mock.patch.object(np.random, "randint", return_value=draws):
            head, tail = model.train_neg_sample_gen(
                gt_head=[known], gt_tail=[known], negative_sampling=4
            )
        self.assertEqual(head.tolist(), [[5, 5, 7, 7]])
        self.assertEqual(tail.tolist(), [[5, 5, 7, 7]])


class TestTrainingNegativesOnARealSplit(unittest.TestCase):
    """The same property on set_task output, through forward: (b, r2, d) is a
    test triple and (a, r2, d) a training one, so b must be drawable as a
    head negative of (a, r2, d)."""

    # Same graph and split as TestModelsOnSampleDatasets, without its tests.
    setUpClass = classmethod(TestModelsOnSampleDatasets.setUpClass.__func__)

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
        self.assertEqual(
            sorted(unpad({k: v[0] for k, v in batch["ground_truth_head"].items()})),
            [e["a"], e["b"]],
        )

        model = TransE(dataset=self.train, e_dim=4, r_dim=4, negative_sampling=256)
        model.train()
        drawn = record_negatives(model)
        np.random.seed(0)
        model(**batch)["loss"].backward()
        heads = set(drawn["head"].flatten().tolist())
        self.assertIn(e["b"], heads)
        self.assertNotIn(e["a"], heads)

    def test_a_model_cannot_be_built_from_the_base_dataset(self) -> None:
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        # A base dataset has no processor fitted on training triples.
        with self.assertRaisesRegex(TypeError, "training part"):
            TransE(dataset=self.dataset, e_dim=4, r_dim=4)


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

    def build(self, dataset, **kwargs):
        from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

        model = TransE(dataset=dataset, e_dim=4, r_dim=4, **kwargs)
        model.train()
        return model

    def test_the_model_argument_sets_the_sample_count(self) -> None:
        from pyhealth.processors import KGTripleProcessor

        processor = KGTripleProcessor(num_entities=10, num_relations=3)
        processor.fit([{"triple": t} for t in [(0, 0, 4), (1, 1, 5), (2, 2, 6)]], "triple")

        class _Train:
            input_processors = {"triple": processor}

        model = self.build(_Train(), negative_sampling=7)
        shapes = self.spy(model)
        model(triple=torch.tensor([[0, 0, 4], [1, 1, 5]]))
        self.assertEqual(shapes, [(2, 7)])

    def test_numpy_integers_are_accepted(self) -> None:
        import numpy as np

        dataset = processor_dataset(10, 3)
        self.assertEqual(self.build(dataset, negative_sampling=np.int64(6)).negative_sampling, 6)
        self.assertEqual(self.build(dataset, negative_sampling=np.int32(5)).negative_sampling, 5)
        with self.assertRaises(ValueError):
            self.build(dataset, negative_sampling=True)
