"""Unit tests for the BulkRNABert downstream cancer-classification workflow.

Covers:

* :class:`pyhealth.models.BulkRNABertClassifier` — forward contract,
  learning signal, mode inference from the sample dataset.
* :meth:`pyhealth.models.BulkRNABert.encode` — mean-pooled embedding shape
  for both expression modes.
* :func:`pyhealth.datasets.load_tcga_cancer_classification_5cohort` — label
  assignment from a synthetic mapping CSV, and the embedding / identifier
  CSV row-alignment contract.
* :func:`pyhealth.datasets.stratified_split_indices` — per-class proportions
  are preserved.
* End-to-end Trainer smoke test on a synthetic problem.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np
import torch

from pyhealth.datasets import (
    TCGARNASeqEmbeddingDataset,
    create_sample_dataset,
    get_dataloader,
    load_tcga_cancer_classification_5cohort,
    stratified_split_indices,
)
from pyhealth.models import BulkRNABert, BulkRNABertClassifier, BulkRNABertConfig
from pyhealth.tasks import LABEL_MAP, TCGACancerClassification5Cohort
from pyhealth.trainer import Trainer


def _tiny_config(expression_mode: str = "discrete") -> BulkRNABertConfig:
    return BulkRNABertConfig(
        n_genes=16,
        n_bins=8,
        embed_dim=12,
        num_layers=1,
        num_heads=2,
        ffn_embed_dim=16,
        init_gene_embed_dim=12,
        expression_mode=expression_mode,
        continuous_hidden_dim=12 if expression_mode == "continuous" else None,
    )


def _make_classifier_dataset(n_per_class: int = 20, embed_dim: int = 12, n_classes: int = 5):
    """Build a small in-memory classifier dataset with linearly separable means."""
    rng = np.random.default_rng(0)
    centers = rng.normal(0.0, 3.0, size=(n_classes, embed_dim)).astype(np.float32)
    samples: List[dict] = []
    for cls in range(n_classes):
        for i in range(n_per_class):
            emb = centers[cls] + 0.1 * rng.standard_normal(embed_dim).astype(np.float32)
            samples.append(
                {
                    "patient_id": f"s{cls}_{i}",
                    "embedding": emb,
                    "label": int(cls),
                }
            )
    task = TCGACancerClassification5Cohort()
    return create_sample_dataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        task_name=task.task_name,
        in_memory=True,
    )


class TestBulkRNABertEncode(unittest.TestCase):
    def test_discrete_shape(self):
        cfg = _tiny_config("discrete")
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        tokens = torch.randint(0, cfg.n_bins, (3, cfg.n_genes))
        emb = model.encode(tokens)
        self.assertEqual(emb.shape, (3, cfg.embed_dim))
        self.assertEqual(emb.dtype, torch.float32)

    def test_continuous_shape(self):
        cfg = _tiny_config("continuous")
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        values = torch.rand(3, cfg.n_genes) * 5.0
        emb = model.encode(values)
        self.assertEqual(emb.shape, (3, cfg.embed_dim))

    def test_deterministic_eval(self):
        """encode must run no-mask — same input -> same output."""
        cfg = _tiny_config("discrete")
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        tokens = torch.randint(0, cfg.n_bins, (2, cfg.n_genes))
        a = model.encode(tokens)
        b = model.encode(tokens)
        torch.testing.assert_close(a, b)

    def test_training_mode_restored(self):
        cfg = _tiny_config("discrete")
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        model.train()
        tokens = torch.randint(0, cfg.n_bins, (1, cfg.n_genes))
        model.encode(tokens)
        self.assertTrue(model.training)

    def test_shape_mismatch_raises(self):
        cfg = _tiny_config("discrete")
        model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
        bad = torch.randint(0, cfg.n_bins, (2, cfg.n_genes - 1))
        with self.assertRaises(ValueError):
            model.encode(bad)


class TestBulkRNABertClassifier(unittest.TestCase):
    def setUp(self):
        self.dataset = _make_classifier_dataset(n_per_class=10, embed_dim=12, n_classes=5)
        self.model = BulkRNABertClassifier(
            dataset=self.dataset,
            hidden_sizes=(16, 8),
            embed_dim=12,
        )

    def test_mode_and_num_classes(self):
        self.assertEqual(self.model.mode, "multiclass")
        self.assertEqual(self.model.num_classes, 5)

    def test_forward_contract(self):
        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))
        out = self.model(**batch)
        self.assertIn("loss", out)
        self.assertIn("y_prob", out)
        self.assertIn("y_true", out)
        self.assertIn("logit", out)
        self.assertEqual(out["logit"].shape, (4, 5))
        self.assertEqual(out["y_true"].shape, (4,))
        self.assertEqual(out["y_prob"].shape, (4, 5))
        self.assertEqual(out["loss"].dim(), 0)

    def test_backward_learns(self):
        loader = get_dataloader(self.dataset, batch_size=8, shuffle=True)
        optim = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        first_losses = []
        for i, batch in enumerate(loader):
            out = self.model(**batch)
            optim.zero_grad()
            out["loss"].backward()
            optim.step()
            first_losses.append(float(out["loss"].detach()))
            if i >= 2:
                break
        last_losses = []
        for _ in range(20):
            for batch in loader:
                out = self.model(**batch)
                optim.zero_grad()
                out["loss"].backward()
                optim.step()
                last_losses.append(float(out["loss"].detach()))
        self.assertLess(
            sum(last_losses[-3:]) / 3,
            sum(first_losses) / len(first_losses),
        )

    def test_wrong_input_dim_raises(self):
        bad_batch = {
            "embedding": torch.randn(2, 8),  # wrong embed_dim
            "label": torch.zeros(2, dtype=torch.long),
        }
        with self.assertRaises(ValueError):
            self.model(**bad_batch)


class TestTCGALabelAssignment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        self.mapping_path = root / "tcga_file_mapping.csv"
        with open(self.mapping_path, "w") as f:
            f.write("project,file_name,sample_type\n")
            f.write("TCGA-BLCA,id0.counts.tsv,Primary Tumor\n")
            f.write("TCGA-BRCA,id1.counts.tsv,Primary Tumor\n")
            f.write("TCGA-GBM,id2.counts.tsv,Primary Tumor\n")
            f.write("TCGA-LGG,id3.counts.tsv,Primary Tumor\n")
            f.write("TCGA-LUAD,id4.counts.tsv,Primary Tumor\n")
            f.write("TCGA-UCEC,id5.counts.tsv,Primary Tumor\n")
            # Filtered out: non-target cohort and non-primary tumor.
            f.write("TCGA-KIRC,id6.counts.tsv,Primary Tumor\n")
            f.write("TCGA-BRCA,id7.counts.tsv,Solid Tissue Normal\n")

        self.identifier_path = root / "tcga_preprocessed.csv"
        with open(self.identifier_path, "w") as f:
            f.write("geneA,geneB,identifier\n")
            # Row order matters — must match the embeddings array.
            for i in range(8):
                f.write(f"0.0,0.0,id{i}\n")

        self.embeddings_path = root / "emb.npy"
        self.embeddings = np.arange(8 * 4, dtype=np.float32).reshape(8, 4)
        np.save(self.embeddings_path, self.embeddings)

    def tearDown(self):
        self.tmp.cleanup()

    def test_expected_labels(self):
        ds = load_tcga_cancer_classification_5cohort(
            embeddings_path=self.embeddings_path,
            identifier_csv=self.identifier_path,
            mapping_csv=self.mapping_path,
        )
        self.assertEqual(len(ds), 6)

        labels = []
        embeddings = []
        for sample in ds:
            labels.append(int(sample["label"].item()))
            embeddings.append(sample["embedding"].numpy())
        # 6 rows retained (id0..id5), filtered out id6 (wrong cohort) and id7
        # (not Primary Tumor). Row i in the embeddings matrix aligns with idi.
        self.assertEqual(labels, [0, 1, 2, 2, 3, 4])
        np.testing.assert_allclose(np.stack(embeddings), self.embeddings[:6])

    def test_label_map_has_5_distinct_classes(self):
        self.assertEqual(set(LABEL_MAP.values()), {0, 1, 2, 3, 4})


class TestStratifiedSplit(unittest.TestCase):
    def test_three_way_class_proportions_preserved(self):
        labels = np.array([0] * 20 + [1] * 30 + [2] * 10)
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels, val_ratio=0.1, test_ratio=0.2, seed=0
        )
        # Every class appears in all three splits.
        for cls in (0, 1, 2):
            self.assertGreater(int((labels[train_idx] == cls).sum()), 0)
            self.assertGreater(int((labels[val_idx] == cls).sum()), 0)
            self.assertGreater(int((labels[test_idx] == cls).sum()), 0)
        # No overlap between any pair of splits.
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())
        self.assertEqual(train_set & val_set, set())
        self.assertEqual(train_set & test_set, set())
        self.assertEqual(val_set & test_set, set())
        # Total coverage: the three splits partition [0, n).
        self.assertEqual(
            sorted(train_set | val_set | test_set), list(range(len(labels)))
        )

    def test_per_class_ratio_floors_match_formula(self):
        # 60 samples in class 0, 20 in class 1 — exercise int() flooring for
        # both val and test. With val=0.1, test=0.2:
        #   class 0 (n=60): n_test=12, n_val=6, n_train=42
        #   class 1 (n=20): n_test=4,  n_val=2, n_train=14
        labels = np.array([0] * 60 + [1] * 20)
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels, val_ratio=0.1, test_ratio=0.2, seed=0
        )
        self.assertEqual(int((labels[test_idx] == 0).sum()), 12)
        self.assertEqual(int((labels[val_idx] == 0).sum()), 6)
        self.assertEqual(int((labels[train_idx] == 0).sum()), 42)
        self.assertEqual(int((labels[test_idx] == 1).sum()), 4)
        self.assertEqual(int((labels[val_idx] == 1).sum()), 2)
        self.assertEqual(int((labels[train_idx] == 1).sum()), 14)

    def test_tiny_class_gets_at_least_one_test_sample(self):
        # 2-sample class 0 rounds up to 1 test; no room for val so it falls
        # back to 0 val / 1 train — mirrors the pre-3-way behavior.
        labels = np.array([0, 0, 1, 1, 1, 1, 1])
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels, val_ratio=0.1, test_ratio=0.1, seed=0
        )
        self.assertEqual(int((labels[test_idx] == 0).sum()), 1)
        self.assertEqual(int((labels[val_idx] == 0).sum()), 0)
        self.assertEqual(int((labels[train_idx] == 0).sum()), 1)
        # Class 1 (n=5): n_test=1, n_val=1, n_train=3
        self.assertEqual(int((labels[test_idx] == 1).sum()), 1)
        self.assertEqual(int((labels[val_idx] == 1).sum()), 1)
        self.assertEqual(int((labels[train_idx] == 1).sum()), 3)

    def test_invalid_ratios_raise(self):
        labels = np.array([0, 0, 1, 1])
        with self.assertRaises(ValueError):
            stratified_split_indices(labels, val_ratio=-0.1, test_ratio=0.2)
        with self.assertRaises(ValueError):
            stratified_split_indices(labels, val_ratio=0.5, test_ratio=0.5)


class TestTrainerSmoke(unittest.TestCase):
    def test_train_converges(self):
        torch.manual_seed(0)
        dataset = _make_classifier_dataset(n_per_class=30, embed_dim=12, n_classes=3)
        labels = [int(s["label"].item()) for s in dataset]
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels, val_ratio=0.1, test_ratio=0.2, seed=0
        )
        train_loader = get_dataloader(
            dataset.subset(train_idx.tolist()), batch_size=8, shuffle=True
        )
        val_loader = get_dataloader(
            dataset.subset(val_idx.tolist()), batch_size=8, shuffle=False
        )
        test_loader = get_dataloader(
            dataset.subset(test_idx.tolist()), batch_size=8, shuffle=False
        )
        model = BulkRNABertClassifier(
            dataset=dataset, hidden_sizes=(16,), embed_dim=12
        )
        trainer = Trainer(
            model=model,
            metrics=["accuracy", "f1_weighted"],
            device="cpu",
            enable_logging=False,
        )
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=5,
            optimizer_params={"lr": 1e-2},
            monitor="f1_weighted",
            monitor_criterion="max",
            load_best_model_at_last=False,
        )
        scores = trainer.evaluate(test_loader)
        # Linearly separable toy problem — the head should comfortably beat chance.
        self.assertGreater(scores["accuracy"], 0.5)


def _build_synthetic_sources(root: Path, n: int = 6, embed_dim: int = 4):
    """Write the trio of files the Dataset / factory expect."""
    mapping = root / "tcga_file_mapping.csv"
    with open(mapping, "w") as f:
        f.write("project,file_name,sample_type\n")
        f.write("TCGA-BLCA,id0.counts.tsv,Primary Tumor\n")
        f.write("TCGA-BRCA,id1.counts.tsv,Primary Tumor\n")
        f.write("TCGA-GBM,id2.counts.tsv,Primary Tumor\n")
        f.write("TCGA-LGG,id3.counts.tsv,Primary Tumor\n")
        f.write("TCGA-LUAD,id4.counts.tsv,Primary Tumor\n")
        f.write("TCGA-UCEC,id5.counts.tsv,Primary Tumor\n")

    identifier_csv = root / "tcga_preprocessed.csv"
    with open(identifier_csv, "w") as f:
        f.write("geneA,geneB,identifier\n")
        for i in range(n):
            f.write(f"0.0,0.0,id{i}\n")

    embeddings_path = root / "emb.npy"
    embeddings = np.arange(n * embed_dim, dtype=np.float32).reshape(n, embed_dim)
    np.save(embeddings_path, embeddings)
    return mapping, identifier_csv, embeddings_path, embeddings


class _DummyEvent:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _DummyPatient:
    def __init__(self, patient_id, events):
        self.patient_id = patient_id
        self._events = events

    def get_events(self, event_type=None):
        if event_type is None:
            return list(self._events)
        return [e for e in self._events if getattr(e, "event_type", None) == event_type]


class TestTaskCall(unittest.TestCase):
    def test_in_map_returns_single_sample(self):
        event = _DummyEvent(
            event_type="rnaseq_embedding",
            cohort="TCGA-BLCA",
            embedding_json="[1.0, 2.0, 3.0]",
        )
        patient = _DummyPatient("id0", [event])
        task = TCGACancerClassification5Cohort()
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s["patient_id"], "id0")
        self.assertEqual(s["label"], 0)
        np.testing.assert_allclose(s["embedding"], [1.0, 2.0, 3.0])
        self.assertEqual(s["embedding"].dtype, np.float32)

    def test_out_of_map_returns_empty(self):
        event = _DummyEvent(
            event_type="rnaseq_embedding",
            cohort="TCGA-KIRC",  # not in LABEL_MAP
            embedding_json="[1.0]",
        )
        task = TCGACancerClassification5Cohort()
        self.assertEqual(task(_DummyPatient("x", [event])), [])

    def test_missing_event_returns_empty(self):
        task = TCGACancerClassification5Cohort()
        self.assertEqual(task(_DummyPatient("x", [])), [])


class TestTCGARNASeqEmbeddingDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = Path(self.tmp.name) / "cache"
        mapping, id_csv, emb_path, self.emb = _build_synthetic_sources(
            self.root, n=6, embed_dim=4
        )
        self.dataset = TCGARNASeqEmbeddingDataset(
            root=str(self.root),
            embeddings_path=emb_path,
            identifier_csv=id_csv,
            mapping_csv=mapping,
            cache_dir=str(self.cache),
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_merged_csv_materialized(self):
        self.assertTrue((self.root / "tcga_rnaseq_embedding.csv").exists())

    def test_unique_patient_ids(self):
        # 6 rows, all map to a cohort in LABEL_MAP -> 6 patients.
        self.assertEqual(set(self.dataset.unique_patient_ids), {f"id{i}" for i in range(6)})

    def test_set_task_end_to_end(self):
        samples = self.dataset.set_task(TCGACancerClassification5Cohort())
        self.assertEqual(len(samples), 6)
        labels = sorted(int(s["label"].item()) for s in samples)
        # LABEL_MAP ordering yields {0,1,2,2,3,4} across id0..id5.
        self.assertEqual(labels, [0, 1, 2, 2, 3, 4])

    def test_default_task(self):
        self.assertIsInstance(
            self.dataset.default_task, TCGACancerClassification5Cohort
        )


if __name__ == "__main__":
    unittest.main()
