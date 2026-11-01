"""Tests for VSACDataset and the ValueSetCompletion task.

Author: Bryan Chance (NetID: bchance2)

Uses five synthetic value sets written to a temporary directory and a
bag-of-words stub encoder, so no network access or model download.
"""

import json
import shutil
import tempfile
import unittest
import zlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from pyhealth.datasets import ValueSetSplit, VSACDataset
from pyhealth.datasets.vsac import CODES_FILE, VALUESETS_FILE
from pyhealth.tasks import ValueSetCompletion, build_retrieval_index

ICD10 = "http://hl7.org/fhir/sid/icd-10-cm"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"

# oid -> (title, split, [(system, code)], expansion.total)
VALUE_SETS = {
    "1.1-20240101": (
        "type 2 diabetes",
        "train",
        [(ICD10, "E11.9"), (ICD10, "E11.65"), (ICD10, "E11.8"), (ICD10, "E13.9")],
        4,
    ),
    "1.2-20240101": (
        "type 1 diabetes",
        "val",
        [(ICD10, "E10.9"), (ICD10, "E10.65"), (ICD10, "E11.9")],
        3,
    ),
    "1.3-20240101": (
        "hypertension",
        "test",
        [(ICD10, "I10"), (ICD10, "I11.9"), (ICD10, "I12.9")],
        3,
    ),
    "1.4-20240101": (
        "hypertension drugs",
        "corpus",
        [(RXNORM, "197361"), (RXNORM, "314076"), (ICD10, "I10")],
        5,
    ),
    "1.5-20240101": (
        "hypertension disorder of kidney",
        "corpus",
        [(ICD10, "I10"), (ICD10, "I15.0")],
        2,
    ),
}


def stub_encoder(texts: Sequence[str]) -> np.ndarray:
    """Unit-norm bag-of-words hashing; shared words mean higher similarity."""
    vecs = np.zeros((len(texts), 16), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, zlib.crc32(word.encode()) % 16] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def write_fixture(root: Path) -> None:
    exp_dir = root / "expansions"
    exp_dir.mkdir(parents=True)
    manifest = []
    for oid, (title, split, members, total) in VALUE_SETS.items():
        resource = {
            "resourceType": "ValueSet",
            "id": oid,
            "title": title,
            "version": oid.rsplit("-", 1)[1],
            "status": "active",
            "publisher": "VSAC Publisher",
            "expansion": {
                "total": total,
                "contains": [
                    {"system": system, "code": code, "display": f"{code} name"}
                    for system, code in members
                ],
            },
        }
        (exp_dir / f"{oid}.json").write_text(json.dumps(resource))
        manifest.append(
            {"oid": oid, "split": split, "type": "", "publisher": "Steward"}
        )
    (root / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest)
    )


_SHARED: dict = {}


def setUpModule():
    """Builds the dataset and retrieval index once for every test class."""
    tmp = Path(tempfile.mkdtemp())
    write_fixture(tmp / "vsac")
    dataset = VSACDataset(root=str(tmp / "vsac"), cache_dir=tmp / "cache")
    index_dir = build_retrieval_index(
        dataset,
        stub_encoder,
        str(tmp / "index"),
        max_k=2,
        source_splits=("train", "val", "test", "corpus"),
    )
    _SHARED.update(tmp=tmp, dataset=dataset, index_dir=index_dir)


def tearDownModule():
    shutil.rmtree(_SHARED["tmp"])


class VSACFixture(unittest.TestCase):
    """Exposes the module fixture as class attributes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = _SHARED["tmp"]
        cls.dataset = _SHARED["dataset"]
        cls.index_dir = _SHARED["index_dir"]

    def samples_for(self, oid: str, **task_args) -> dict:
        task = ValueSetCompletion(self.index_dir, **{"k": 2, **task_args})
        samples = task(self.dataset.get_patient(oid))
        return {(s["system"], s["code"]): s for s in samples}


class TestVSACDataset(VSACFixture):
    def test_writes_flat_tables(self):
        for name in (VALUESETS_FILE, CODES_FILE):
            self.assertTrue((self.tmp / "vsac" / name).exists())

    def test_one_patient_per_value_set(self):
        self.assertEqual(sorted(self.dataset.unique_patient_ids), sorted(VALUE_SETS))

    def test_valueset_event(self):
        event = self.dataset.get_patient("1.4-20240101").get_events("valuesets")[0]
        self.assertEqual(event["title"], "hypertension drugs")
        self.assertEqual(event["split"], "corpus")
        self.assertEqual(event["version"], "20240101")
        self.assertEqual(event["publisher"], "Steward")  # manifest wins
        self.assertEqual(event["type"], "Condition/Diagnosis")  # inferred

    def test_code_events_keep_expansion_order(self):
        codes = self.dataset.get_patient("1.1-20240101").get_events("codes")
        ranked = sorted(codes, key=lambda e: int(e["rank"]))
        self.assertEqual(
            [e["code"] for e in ranked], ["E11.9", "E11.65", "E11.8", "E13.9"]
        )
        self.assertEqual(ranked[0]["display"], "E11.9 name")

    def test_missing_expansions_raises(self):
        with self.assertRaises(FileNotFoundError):
            VSACDataset.prepare_metadata(str(self.tmp / "empty"))

    def test_manifest_split(self):
        split = self.dataset.manifest_split()
        parts = split.split_indices({oid: [i] for i, oid in enumerate(VALUE_SETS)})
        self.assertEqual([p.tolist() for p in parts], [[0], [1], [2]])

    def test_split_key_tracks_assignments(self):
        a = ValueSetSplit({"x": "train"}).to_dict()
        b = ValueSetSplit({"x": "test"}).to_dict()
        self.assertNotEqual(a, b)


class TestValueSetCompletion(VSACFixture):
    def test_index_excludes_self(self):
        task_index = json.loads((Path(self.index_dir) / "meta.json").read_text())
        self.assertEqual(task_index["n_queries"], 5)
        samples = self.samples_for("1.1-20240101", k=1)
        # Nearest other set is "type 1 diabetes"; its codes are the pool.
        self.assertEqual(
            set(samples), {(ICD10, "E10.9"), (ICD10, "E10.65"), (ICD10, "E11.9")}
        )

    def test_labels(self):
        samples = self.samples_for("1.1-20240101", k=1)
        labels = {code: s["label"] for (_, code), s in samples.items()}
        self.assertEqual(labels, {"E10.9": 0, "E10.65": 0, "E11.9": 1})
        self.assertEqual(samples[(ICD10, "E11.9")]["n_true_codes"], 4)

    def test_feature_shapes(self):
        sample = next(iter(self.samples_for("1.3-20240101", k=2).values()))
        self.assertEqual(sample["title_embedding"].shape, (16,))
        self.assertEqual(sample["candidate_embedding"].shape, (16,))
        self.assertEqual(sample["code_system"].sum(), 1.0)
        self.assertEqual(sample["score"].shape, (1,))

    def test_dedup_keeps_best_score(self):
        samples = self.samples_for("1.3-20240101", k=2)
        # Both neighbors supply I10; the closer one ("hypertension drugs",
        # cosine 1/sqrt(2)) sets its score.
        self.assertEqual(sum(code == "I10" for _, code in samples), 1)
        self.assertAlmostEqual(
            float(samples[(ICD10, "I10")]["score"][0]), 2**-0.5, places=5
        )
        self.assertEqual(float(samples[(ICD10, "I15.0")]["score"][0]), 0.5)
        self.assertEqual(samples[(ICD10, "I10")]["label"], 1)

    def test_cap_limits_labels_and_pool(self):
        capped = self.samples_for("1.1-20240101", k=1, max_codes_per_set=2, min_codes=1)
        self.assertEqual(set(capped), {(ICD10, "E10.9"), (ICD10, "E10.65")})
        self.assertEqual(capped[(ICD10, "E10.9")]["n_true_codes"], 2)

    def test_min_codes_after_cap(self):
        self.assertEqual(self.samples_for("1.1-20240101", max_codes_per_set=2), {})

    def test_non_target_split_skipped(self):
        self.assertEqual(self.samples_for("1.4-20240101"), {})

    def test_feature_subset(self):
        task = ValueSetCompletion(
            self.index_dir, k=2, features=["score", "code_system"]
        )
        self.assertEqual(list(task.input_schema), ["code_system", "score"])
        sample = task(self.dataset.get_patient("1.3-20240101"))[0]
        self.assertNotIn("title_embedding", sample)

    def test_invalid_arguments(self):
        for kwargs in ({"k": 3}, {"features": ["nope"]}, {"max_codes_per_set": 0}):
            with self.assertRaises(ValueError):
                ValueSetCompletion(self.index_dir, **kwargs)

    def test_cap_changes_task_cache_key(self):
        capped = vars(ValueSetCompletion(self.index_dir, k=2))
        full = vars(ValueSetCompletion(self.index_dir, k=2, max_codes_per_set=None))
        self.assertNotEqual(capped, full)

    def test_set_task_with_manifest_split(self):
        task = ValueSetCompletion(self.index_dir, k=2, min_codes=1)
        train, val, test = self.dataset.set_task(
            task, num_workers=1, split=self.dataset.manifest_split()
        )
        self.assertEqual({s["patient_id"] for s in train}, {"1.1-20240101"})
        self.assertEqual({s["patient_id"] for s in test}, {"1.3-20240101"})
        self.assertEqual(train[0]["title_embedding"].shape[-1], 16)
        self.assertIsInstance(train[0]["code"], str)
        self.assertGreater(train[0]["n_true_codes"], 0)
        self.assertGreater(len(val), 0)


if __name__ == "__main__":
    unittest.main()
