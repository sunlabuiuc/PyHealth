"""Tests for ValueSetPairClassification and build_hybrid_retrieval_index.

Author: Wenchang Qi (NetID: qi14)

Covers the text-pair samples, max-score pooling, BM25 and reciprocal rank
fusion, and the task feeding RASCCrossEncoder end to end. Five synthetic
value sets (the fixture of test_vsac.py), stub encoders and a random
two-layer BERT; no network access.
"""

import json
import shutil
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import torch

from pyhealth.datasets import VSACDataset, get_dataloader
from pyhealth.models import RASCCrossEncoder
from pyhealth.tasks import (
    ValueSetCompletion,
    ValueSetPairClassification,
    build_hybrid_retrieval_index,
    build_retrieval_index,
)
from pyhealth.tasks.value_set_pair_classification import (
    _bm25_scores,
    _hybrid_neighbors,
    _load_index,
)
from tests.core.test_rasc import write_tiny_bert
from tests.core.test_vsac import _THREADED_DASK, ICD10, stub_encoder, write_fixture

_SHARED: dict = {}


def misleading_encoder(texts: Sequence[str]) -> np.ndarray:
    """Puts "hypertension" next to the diabetes titles, away from its own kin.

    Titles mentioning drugs or kidney map to one axis and every other text to
    the other, so dense search alone retrieves the wrong value sets.
    """
    vecs = np.zeros((len(texts), 2), dtype=np.float32)
    for i, text in enumerate(texts):
        vecs[i, int("drugs" in text or "kidney" in text)] = 1.0
    return vecs


def setUpModule() -> None:
    """Builds the dataset, the three retrieval indexes and the tiny encoder."""
    _THREADED_DASK.start()
    tmp = Path(tempfile.mkdtemp())
    write_fixture(tmp / "vsac")
    vsac = VSACDataset(root=str(tmp / "vsac"), cache_dir=tmp / "cache")
    splits = ("train", "val", "test", "corpus")
    _SHARED.update(
        tmp=tmp,
        bert=write_tiny_bert(tmp / "bert"),
        vsac=vsac,
        index_dir=build_retrieval_index(
            vsac, stub_encoder, str(tmp / "index"), max_k=2, source_splits=splits
        ),
        flat_dir=build_retrieval_index(
            vsac, misleading_encoder, str(tmp / "flat"), max_k=2, source_splits=splits
        ),
        hybrid_dir=build_hybrid_retrieval_index(
            vsac, misleading_encoder, str(tmp / "hybrid"), max_k=2, source_splits=splits
        ),
    )


def tearDownModule() -> None:
    """Removes the temporary directory and restores process-based Dask."""
    _THREADED_DASK.stop()
    shutil.rmtree(_SHARED["tmp"], ignore_errors=True)


class TestPairSamples(unittest.TestCase):
    def samples_for(self, oid: str, index: str = "index_dir", **task_args) -> dict:
        task = ValueSetPairClassification(_SHARED[index], **{"k": 2, **task_args})
        samples = task(_SHARED["vsac"].get_patient(oid))
        return {(s["system"], s["code"]): s for s in samples}

    def test_text_inputs_hold_title_and_display(self):
        samples = self.samples_for("1.3-20240101")
        sample = samples[(ICD10, "I10")]
        self.assertEqual(sample["title_text"], "hypertension")
        self.assertEqual(sample["candidate_text"], "I10 name")
        # A code without a display name falls back to the code itself.
        self.assertEqual(samples[(ICD10, "I15.0")]["candidate_text"], "I15.0")
        self.assertEqual(sample["n_true_codes"], 3)
        for absent in ("score", "code_system", "title_embedding"):
            self.assertNotIn(absent, sample)

    def test_schema(self):
        task = ValueSetPairClassification(_SHARED["index_dir"], k=2)
        self.assertEqual(
            task.input_schema, {"title_text": "text", "candidate_text": "text"}
        )
        self.assertEqual(
            task.output_schema, {"label": ("binary", {"pos_weight": "balanced"})}
        )
        fused = ValueSetPairClassification(
            _SHARED["index_dir"], k=2, fusion_features=["score"]
        )
        self.assertEqual(
            fused.input_schema,
            {"score": "tensor", "title_text": "text", "candidate_text": "text"},
        )

    def test_fusion_inputs(self):
        samples = self.samples_for(
            "1.3-20240101", fusion_features=["score", "code_system"]
        )
        sample = samples[(ICD10, "I10")]
        self.assertEqual(sample["score"].shape, (1,))
        self.assertEqual(sample["code_system"].tolist(), [0, 1, 0, 0, 0, 0, 0, 0])

    def test_same_pool_and_labels_as_the_embedding_task(self):
        patient = _SHARED["vsac"].get_patient("1.3-20240101")
        embedded = ValueSetCompletion(_SHARED["index_dir"], k=2)(patient)
        pairs = ValueSetPairClassification(
            _SHARED["index_dir"], k=2, fusion_features=["score"]
        )(patient)
        self.assertEqual(
            [(s["code"], s["system"], s["label"]) for s in pairs],
            [(s["code"], s["system"], s["label"]) for s in embedded],
        )
        for pair, vector in zip(pairs, embedded):
            self.assertEqual(float(pair["score"][0]), float(vector["score"][0]))

    def test_cap_cuts_the_candidate_pool(self):
        # "type 1 diabetes" lists E10.9, E10.65, E11.9; the cap drops E11.9.
        oid = "1.1-20240101"
        full = self.samples_for(oid, k=1, min_codes=1)
        self.assertEqual(full[(ICD10, "E11.9")]["label"], 1)
        capped = self.samples_for(oid, k=1, max_codes_per_set=2, min_codes=1)
        self.assertEqual(set(capped), {(ICD10, "E10.9"), (ICD10, "E10.65")})
        self.assertEqual({s["label"] for s in capped.values()}, {0})
        self.assertEqual({s["n_true_codes"] for s in capped.values()}, {2})

    def test_cap_cuts_the_labels(self):
        # The target's third code, E11.9, is a candidate from "type 2
        # diabetes"; under a cap of 2 it is no longer a true code.
        oid = "1.2-20240101"
        full = self.samples_for(oid, k=1, min_codes=1)
        self.assertEqual(full[(ICD10, "E11.9")]["label"], 1)
        self.assertEqual(full[(ICD10, "E11.9")]["n_true_codes"], 3)
        capped = self.samples_for(oid, k=1, max_codes_per_set=2, min_codes=1)
        self.assertEqual(capped[(ICD10, "E11.9")]["label"], 0)
        self.assertEqual(capped[(ICD10, "E11.9")]["n_true_codes"], 2)
        self.assertNotIn((ICD10, "E11.8"), capped)  # third code of the source

    def test_cap_matches_the_embedding_task(self):
        for oid in ("1.1-20240101", "1.2-20240101", "1.3-20240101"):
            for cap in (1, 2, 3, None):
                args = {"k": 2, "max_codes_per_set": cap, "min_codes": 1}
                patient = _SHARED["vsac"].get_patient(oid)
                pairs = ValueSetPairClassification(_SHARED["index_dir"], **args)
                embedded = ValueSetCompletion(_SHARED["index_dir"], **args)
                self.assertEqual(
                    [
                        (s["code"], s["system"], s["label"], s["n_true_codes"])
                        for s in pairs(patient)
                    ],
                    [
                        (s["code"], s["system"], s["label"], s["n_true_codes"])
                        for s in embedded(patient)
                    ],
                    msg=(oid, cap),
                )

    def test_min_codes_counts_codes_after_the_cap(self):
        self.assertEqual(self.samples_for("1.1-20240101", max_codes_per_set=2), {})

    def test_not_a_target_yields_nothing(self):
        patient = _SHARED["vsac"].get_patient("1.3-20240101")
        split = patient.get_events(event_type="valuesets")[0]["split"]
        other = next(s for s in ("train", "val", "test") if s != split)
        task = ValueSetPairClassification(
            _SHARED["index_dir"], k=2, target_splits=[other]
        )
        self.assertEqual(task(patient), [])
        strict = ValueSetPairClassification(_SHARED["index_dir"], k=2, min_codes=99)
        self.assertEqual(strict(patient), [])

    def test_rejects_bad_arguments(self):
        for kwargs in (
            {"k": 0},
            {"k": 3},  # the index holds two neighbors
            {"fusion_features": ["title_embedding"]},
            {"max_codes_per_set": 0},
            {"min_codes": 0},
            {"target_splits": []},
        ):
            with self.assertRaises(ValueError, msg=kwargs):
                ValueSetPairClassification(_SHARED["index_dir"], **{"k": 2, **kwargs})
        with self.assertRaises(FileNotFoundError):
            ValueSetPairClassification(str(_SHARED["tmp"] / "missing"))

    def test_pool_keeps_best_score_in_any_neighbor_order(self):
        task = ValueSetPairClassification(
            _SHARED["index_dir"], k=2, fusion_features=["score"]
        )
        index = _load_index(task.index_dir, task.fingerprint)
        oid = "1.3-20240101"
        expected = float(
            self.samples_for(oid, fusion_features=["score"])[(ICD10, "I10")]["score"][0]
        )
        with mock.patch.dict(index.neighbors, {oid: index.neighbors[oid][::-1]}):
            reordered = task(_SHARED["vsac"].get_patient(oid))
        scores = {s["code"]: float(s["score"][0]) for s in reordered}
        self.assertEqual(scores["I10"], expected)
        self.assertAlmostEqual(expected, 2**-0.5, places=5)

    def test_task_to_model_pipeline(self):
        task = ValueSetPairClassification(
            _SHARED["index_dir"],
            k=2,
            min_codes=1,
            fusion_features=["score", "code_system"],
        )
        train, _, test = _SHARED["vsac"].set_task(
            task, num_workers=1, split=_SHARED["vsac"].manifest_split()
        )
        self.assertIsInstance(train[0]["title_text"], str)
        model = RASCCrossEncoder(train, model_name=_SHARED["bert"], max_length=16)
        self.assertEqual(model.fusion_dim, 9)  # score + 8 code systems
        out = model.eval()(**next(iter(get_dataloader(test, batch_size=8))))
        self.assertEqual(out["y_prob"].shape, (len(test), 1))
        self.assertTrue(torch.isfinite(out["loss"]))


class TestHybridRetrieval(unittest.TestCase):
    def neighbors(self, index: str) -> dict:
        frame = pl.read_parquet(Path(_SHARED[index]) / "neighbors.parquet")
        out: dict = {}
        for oid, neighbor, position in (
            frame.sort("oid", "position")
            .select("oid", "neighbor", "position")
            .iter_rows()
        ):
            out.setdefault(oid, []).append(neighbor)
        return out

    def test_writes_the_dense_index_layout(self):
        dense, hybrid = Path(_SHARED["flat_dir"]), Path(_SHARED["hybrid_dir"])
        self.assertEqual(
            sorted(p.name for p in hybrid.iterdir()),
            sorted(p.name for p in dense.iterdir()),
        )
        meta = json.loads((hybrid / "meta.json").read_text())
        self.assertEqual(meta["retriever"], "hybrid")
        self.assertEqual(meta["max_k"], 2)
        self.assertEqual(meta["rrf_k"], 60)
        self.assertIn("ValueSetPairClassification", meta["note"])

    def test_hybrid_neighbors_are_not_in_score_order(self):
        # Why a hybrid index is for ValueSetPairClassification only: "b" is
        # the closer vector, but "a" also shares the query's word, so it
        # ranks first with the lower dense score.
        frame = _hybrid_neighbors(
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32),
            ["q"],
            ["a", "b"],
            ["asthma"],
            ["asthma drugs", "kidney disorder"],
            max_k=2,
        ).sort("position")
        self.assertEqual(frame["neighbor"].to_list(), ["a", "b"])
        scores = frame["score"].to_list()
        self.assertAlmostEqual(scores[0], 0.6, places=6)
        self.assertAlmostEqual(scores[1], 0.8, places=6)

    def test_bm25_ranks_shared_terms(self):
        scores = _bm25_scores(
            ["hypertension", "asthma"],
            [
                "type 2 diabetes",
                "hypertension drugs",
                "hypertension disorder of kidney",
            ],
        ).toarray()
        self.assertEqual(scores.shape, (2, 3))
        self.assertEqual(scores[0, 0], 0.0)
        self.assertGreater(scores[0, 1], scores[0, 2])  # shorter title wins
        self.assertEqual(float(scores[1].sum()), 0.0)  # unseen term

    def test_bm25_recovers_what_dense_search_misses(self):
        # Dense search pairs "hypertension" with the two diabetes sets; BM25
        # brings the sets that share the word to the front.
        self.assertEqual(
            set(self.neighbors("flat_dir")["1.3-20240101"]),
            {"1.1-20240101", "1.2-20240101"},
        )
        self.assertEqual(
            self.neighbors("hybrid_dir")["1.3-20240101"],
            ["1.4-20240101", "1.5-20240101"],
        )
        self.assertEqual(
            self.neighbors("hybrid_dir")["1.1-20240101"][0], "1.2-20240101"
        )

    def test_hybrid_excludes_self_and_respects_max_k(self):
        for oid, found in self.neighbors("hybrid_dir").items():
            self.assertNotIn(oid, found)
            self.assertEqual(len(found), 2)

    def test_score_stays_the_dense_similarity(self):
        frame = pl.read_parquet(Path(_SHARED["hybrid_dir"]) / "neighbors.parquet")
        scores = {(o, n): s for o, n, s, _ in frame.iter_rows()}
        # Fused rank picks the neighbor; its score is still the inner product.
        self.assertEqual(scores[("1.3-20240101", "1.4-20240101")], 0.0)
        self.assertEqual(scores[("1.4-20240101", "1.5-20240101")], 1.0)

    def test_hybrid_changes_the_candidate_pool(self):
        def pool(index: str) -> set:
            task = ValueSetPairClassification(_SHARED[index], k=2)
            patient = _SHARED["vsac"].get_patient("1.3-20240101")
            return {s["code"]: s["label"] for s in task(patient)}

        # The dense pool holds no true code; the hybrid one holds I10.
        self.assertEqual(sum(pool("flat_dir").values()), 0)
        self.assertEqual(
            pool("hybrid_dir"), {"197361": 0, "314076": 0, "I10": 1, "I15.0": 0}
        )

    def test_hybrid_index_has_its_own_cache_key(self):
        flat = ValueSetPairClassification(_SHARED["flat_dir"], k=2)
        hybrid = ValueSetPairClassification(_SHARED["hybrid_dir"], k=2)
        self.assertNotEqual(flat.fingerprint, hybrid.fingerprint)

    def test_rejects_bad_arguments(self):
        for kwargs in ({"max_k": 0}, {"rrf_k": 0}, {"source_splits": ["nowhere"]}):
            with self.assertRaises(ValueError, msg=kwargs):
                build_hybrid_retrieval_index(
                    _SHARED["vsac"], stub_encoder, str(_SHARED["tmp"] / "bad"), **kwargs
                )


if __name__ == "__main__":
    unittest.main()
