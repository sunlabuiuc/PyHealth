"""Tests for VSACDataset and the ValueSetCompletion task.

Author: Bryan Chance (NetID: bchance2)

Uses five synthetic value sets written to a temporary directory and a
bag-of-words stub encoder, so no network access or model download.
"""

import json
import logging
import shutil
import tempfile
import unittest
import zlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import polars as pl
import requests
import torch

from pyhealth.datasets import VSACDataset
from pyhealth.datasets.vsac import CODES_FILE, MANIFEST_FILE, VALUESETS_FILE
from pyhealth.tasks import ValueSetCompletion, build_retrieval_index
from pyhealth.tasks.value_set_completion import (
    FEATURES,
    normalize_code_system,
    sapbert_encoder,
)

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


PURPOSE = "(Clinical Focus: diabetes mellitus type 2 diagnoses.)"


def stub_encoder(texts: Sequence[str]) -> np.ndarray:
    """Unit-norm bag-of-words hashing; shared words mean higher similarity."""
    vecs = np.zeros((len(texts), 16), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            vecs[i, zlib.crc32(word.encode()) % 16] += 1.0
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1.0, norms)


def write_fixture(root: Path) -> None:
    """Writes expansions, one ValueSet read with a purpose, and a manifest."""
    exp_dir = root / "expansions"
    exp_dir.mkdir(parents=True)
    (root / "resources").mkdir()
    (root / "resources" / "1.1-20240101.json").write_text(
        json.dumps(
            {"resourceType": "ValueSet", "id": "1.1-20240101", "purpose": PURPOSE}
        )
    )
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
                    if code != "I15.0"
                    else {"system": system, "code": code}
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


class FakeResponse:
    """Minimal ``requests.Response`` stand-in."""

    def __init__(self, body: dict, status: int = 200) -> None:
        self.body, self.status_code = body, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self.body


class FakeSession:
    """Serves a 3-code value set in pages of 2 and records each request."""

    def __init__(self) -> None:
        self.codes = [{"system": ICD10, "code": c} for c in ("A00", "A01", "A02")]
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url: str, params: dict | None = None, timeout: int = 0):
        self.calls.append((url, params))
        if "9.9" in url:
            return FakeResponse({}, 404)
        if not url.endswith("$expand"):
            return FakeResponse({"resourceType": "ValueSet", "purpose": "p"})
        offset = params["offset"]
        page = self.codes[offset : offset + 2]
        return FakeResponse(
            {"resourceType": "ValueSet", "expansion": {"total": 3, "contains": page}}
        )


class FakeBatch(dict):
    """Tokenizer output with the ``.to(device)`` that ``BatchEncoding`` has."""

    def to(self, device: str) -> "FakeBatch":
        return self


class FakeTokenizer:
    """Encodes each text as its length and records each batch."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def __call__(self, texts: list[str], **kwargs: Any) -> FakeBatch:
        self.batches.append(texts)
        return FakeBatch(lengths=torch.tensor([float(len(t)) for t in texts]))


class FakeModel:
    """Returns a CLS vector of (length, 1, 0, 0) and noise for other tokens."""

    hidden = 4
    config = SimpleNamespace(hidden_size=hidden)

    def to(self, device: str) -> "FakeModel":
        return self

    def eval(self) -> "FakeModel":
        return self

    def __call__(self, lengths: torch.Tensor) -> SimpleNamespace:
        state = torch.rand(len(lengths), 3, self.hidden)
        state[:, 0] = 0.0
        state[:, 0, 0], state[:, 0, 1] = lengths, 1.0
        return SimpleNamespace(last_hidden_state=state)


# BaseDataset builds the event table on a Dask cluster of worker processes
# (about 1 s to start); the threaded cluster it uses in notebooks keeps the
# whole module setup under 1 s.
_THREADED_DASK = mock.patch(
    "pyhealth.datasets.base_dataset.in_notebook", return_value=True
)


def setUpModule() -> None:
    """Builds the dataset and retrieval index once for every test class."""
    _THREADED_DASK.start()
    tmp = Path(tempfile.mkdtemp())
    _SHARED["tmp"] = tmp
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


def tearDownModule() -> None:
    """Removes the temporary directory and restores process-based Dask."""
    _THREADED_DASK.stop()
    shutil.rmtree(_SHARED["tmp"], ignore_errors=True)


class VSACFixture(unittest.TestCase):
    """Exposes the module fixture as class attributes."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = _SHARED["tmp"]
        cls.dataset = _SHARED["dataset"]
        cls.index_dir = _SHARED["index_dir"]

    def samples_for(self, oid: str, **task_args: Any) -> dict:
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

    def test_manifest_split_needs_manifest(self):
        root = self.tmp / "no_manifest"
        write_fixture(root)
        (root / MANIFEST_FILE).unlink()
        VSACDataset.prepare_metadata(str(root))
        with self.assertRaisesRegex(ValueError, MANIFEST_FILE):
            VSACDataset.manifest_split(SimpleNamespace(root=str(root)))

    def test_description_falls_back_to_purpose(self):
        event = self.dataset.get_patient("1.1-20240101").get_events("valuesets")[0]
        self.assertEqual(event["description"], PURPOSE)

    def test_display_falls_back_to_code(self):
        codes = self.dataset.get_patient("1.5-20240101").get_events("codes")
        self.assertEqual({e["code"]: e["display"] for e in codes}["I15.0"], "I15.0")

    def test_prepare_warns_and_uses_file_name(self):
        root = self.tmp / "prepare"
        write_fixture(root)
        path = root / "expansions" / "1.2-20240101.json"
        path.write_text(path.read_text().replace('"id": "1.2-20240101"', '"id": "x"'))
        (root / "expansions" / "1.3-20240101.json").unlink()
        with self.assertLogs("pyhealth.datasets.vsac", logging.WARNING) as logs:
            VSACDataset.prepare_metadata(str(root))
        text = " ".join(logs.output)
        for phrase in ("no expansion file", "expansion.total", "file name"):
            self.assertIn(phrase, text)
        oids = (root / VALUESETS_FILE).read_text()
        self.assertIn("1.2-20240101", oids)
        self.assertNotIn("1.3-20240101", oids)

    def test_invalid_files_raise(self):
        root = self.tmp / "invalid"
        (root / "expansions").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            VSACDataset.prepare_metadata(str(root))
        (root / "expansions" / "bad.json").write_text("{not json")
        with self.assertRaisesRegex(ValueError, "bad.json"):
            VSACDataset.prepare_metadata(str(root))
        (root / MANIFEST_FILE).write_text('{"oid": "../1.1", "split": "train"}\n')
        with self.assertRaisesRegex(ValueError, "invalid manifest row"):
            VSACDataset.prepare_metadata(str(root))

    def test_download_rejects_path_like_oids(self):
        session = FakeSession()
        with self.assertRaisesRegex(ValueError, "Not VSAC OIDs"):
            VSACDataset.download_expansions(
                str(self.tmp / "unsafe"), ["2.1", "../2.1"], session=session
            )
        self.assertEqual(session.calls, [])
        self.assertFalse((self.tmp / "unsafe").exists())

    def test_download_pages_and_metadata(self):
        root = self.tmp / "download"
        session = FakeSession()
        with mock.patch("pyhealth.datasets.vsac._REQUEST_DELAY", 0):
            failed = VSACDataset.download_expansions(
                str(root), ["2.1", "9.9"], session=session
            )
            self.assertEqual(failed, ["9.9"])
            saved = json.loads((root / "expansions" / "2.1.json").read_text())
            self.assertEqual(len(saved["expansion"]["contains"]), 3)
            self.assertEqual([c[1]["offset"] for c in session.calls[:2]], [0, 2])
            meta = json.loads((root / "resources" / "2.1.json").read_text())
            self.assertEqual(meta["purpose"], "p")
            calls = len(session.calls)
            VSACDataset.download_expansions(str(root), ["2.1"], session=session)
            self.assertEqual(len(session.calls), calls)  # already on disk

    def test_download_without_paging_keeps_first_page(self):
        root = self.tmp / "single_page"
        with mock.patch("pyhealth.datasets.vsac._REQUEST_DELAY", 0):
            VSACDataset.download_expansions(
                str(root),
                ["2.1"],
                paginate=False,
                metadata=False,
                session=FakeSession(),
            )
        saved = json.loads((root / "expansions" / "2.1.json").read_text())
        self.assertEqual(len(saved["expansion"]["contains"]), 2)
        self.assertFalse((root / "resources").exists())

    def test_download_fails_on_empty_page(self):
        root = self.tmp / "short"
        session = FakeSession()
        session.codes = session.codes[:2]  # total stays 3
        with mock.patch("pyhealth.datasets.vsac._REQUEST_DELAY", 0):
            failed = VSACDataset.download_expansions(
                str(root), ["2.1"], metadata=False, session=session
            )
        self.assertEqual(failed, ["2.1"])
        self.assertFalse((root / "expansions" / "2.1.json").exists())

    def test_interrupted_write_leaves_no_file(self):
        root = self.tmp / "interrupted"
        with (
            mock.patch("pyhealth.datasets.vsac._REQUEST_DELAY", 0),
            mock.patch("pyhealth.datasets.vsac.os.replace", side_effect=OSError),
            self.assertRaises(OSError),
        ):
            VSACDataset.download_expansions(
                str(root), ["2.1"], metadata=False, session=FakeSession()
            )
        self.assertFalse((root / "expansions" / "2.1.json").exists())

    def test_download_needs_key(self):
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            self.assertRaises(ValueError),
        ):
            VSACDataset.download_expansions(str(self.tmp / "nokey"), ["2.1"])


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
        for kwargs in (
            {"k": 3},
            {"k": 0},
            {"features": ["nope"]},
            {"max_codes_per_set": 0},
            {"min_codes": 0},
            {"target_splits": ()},
        ):
            with self.assertRaises(ValueError):
                ValueSetCompletion(self.index_dir, **{"k": 2, **kwargs})
        with self.assertRaises(FileNotFoundError):
            ValueSetCompletion(str(self.tmp / "no_index"))

    def test_normalize_code_system(self):
        self.assertEqual(normalize_code_system(ICD10), "ICD-10-CM")
        self.assertEqual(normalize_code_system("urn:x:rxnorm"), "RxNorm")
        self.assertEqual(normalize_code_system("urn:unknown"), "OTHER")

    def test_sapbert_encoder_pools_cls_and_normalizes(self):
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with (
            mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
            mock.patch("transformers.AutoModel.from_pretrained", return_value=model),
        ):
            encode = sapbert_encoder(device="cpu", batch_size=2)
        out = encode(["a", "bbb", "cc"])
        self.assertEqual(out.shape, (3, FakeModel.hidden))
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
        # Row order survives batching: the CLS vector is (len(text), 1, 0, 0).
        np.testing.assert_allclose(out[:, 0] / out[:, 1], [1.0, 3.0, 2.0], rtol=1e-6)
        self.assertEqual(tokenizer.batches, [["a", "bbb"], ["cc"]])
        self.assertEqual(encode([]).shape, (0, FakeModel.hidden))

    def test_source_splits_restrict_retrieval(self):
        index_dir = build_retrieval_index(
            self.dataset,
            stub_encoder,
            str(self.tmp / "corpus_index"),
            max_k=2,
            source_splits=("corpus",),
        )
        neighbors = pl.read_parquet(Path(index_dir) / "neighbors.parquet")
        self.assertEqual(set(neighbors["neighbor"]), {"1.4-20240101", "1.5-20240101"})
        with self.assertRaises(ValueError):
            build_retrieval_index(self.dataset, stub_encoder, index_dir, max_k=0)

    def test_description_embedding(self):
        task = ValueSetCompletion(
            self.index_dir, k=2, features=[*FEATURES, "description_embedding"]
        )
        with_text = task(self.dataset.get_patient("1.1-20240101"))[0]
        without = task(self.dataset.get_patient("1.3-20240101"))[0]
        self.assertGreater(np.abs(with_text["description_embedding"]).sum(), 0)
        self.assertEqual(np.abs(without["description_embedding"]).sum(), 0)
        old = self.tmp / "old_index"
        shutil.copytree(self.index_dir, old)
        meta = json.loads((old / "meta.json").read_text())
        del meta["n_descriptions"]
        (old / "meta.json").write_text(json.dumps(meta))
        with self.assertRaises(ValueError):
            ValueSetCompletion(str(old), k=2, features=["description_embedding"])

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
        positives = sum(float(s["label"]) for s in train)
        self.assertAlmostEqual(
            float(train.output_processors["label"].weight),
            (len(train) - positives) / positives,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
