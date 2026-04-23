"""Tests for KEEP embedding export and medcode file generation.

Tests both the text-vector export (keep_snomed.txt, keep_icd9.txt)
and the medcode CSV generation (SNOMED.csv, ICD9CM_to_SNOMED.csv)
using small synthetic data.
"""

import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np


def _make_snomed_graph():
    """Small SNOMED graph with concept_code and concept_name attributes."""
    G = nx.DiGraph()
    G.add_node(100, concept_code="404684003", concept_name="Clinical finding")
    G.add_node(200, concept_code="49601007", concept_name="Cardiovascular disorder")
    G.add_node(300, concept_code="84114007", concept_name="Heart failure")
    G.add_node(400, concept_code="38341003", concept_name="Hypertensive disorder")
    G.add_edge(200, 100)
    G.add_edge(300, 200)
    G.add_edge(400, 200)
    return G


def _make_embeddings():
    """Tiny 4-node embedding matrix."""
    np.random.seed(42)
    return np.random.randn(4, 4).astype(np.float32)


NODE_IDS = [100, 200, 300, 400]
ICD9_TO_SNOMED = {"428.0": [300], "401.9": [400]}
ICD9_TO_SNOMED_MULTI = {
    "428.0": [300],
    "401.9": [400],
    "250.01": [300, 400],
}


class _KeepExportBase(unittest.TestCase):
    """Shared setUp for KEEP export tests."""

    def setUp(self):
        self.snomed_graph = _make_snomed_graph()
        self.embeddings = _make_embeddings()
        self.node_ids = NODE_IDS
        self.icd9_to_snomed = dict(ICD9_TO_SNOMED)
        self.icd9_to_snomed_multi = dict(ICD9_TO_SNOMED_MULTI)
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


class TestExportToText(_KeepExportBase):

    def test_creates_file(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_to_text,
        )
        path = export_to_text(
            self.embeddings[:2],
            ["token_a", "token_b"],
            self.tmpdir / "test.txt",
        )
        self.assertTrue(path.exists())
        lines = path.read_text().strip().split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("token_a "))
        # token + 4 dims
        self.assertEqual(len(lines[0].split()), 5)

    def test_round_trip_with_pyhealth(self):
        """Verify exported file can be loaded by init_embedding_with_pretrained."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_to_text,
        )
        from pyhealth.models.embedding import _iter_text_vectors

        tokens = ["84114007", "38341003"]
        emb = self.embeddings[:2]
        path = export_to_text(emb, tokens, self.tmpdir / "test.txt")

        vectors = _iter_text_vectors(
            str(path),
            embedding_dim=4,
            wanted_tokens={"84114007", "38341003"},
        )
        self.assertIn("84114007", vectors)
        self.assertIn("38341003", vectors)
        np.testing.assert_allclose(
            vectors["84114007"].numpy(), emb[0], atol=1e-5,
        )


class TestExportSnomed(_KeepExportBase):

    def test_uses_concept_codes_as_tokens(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_snomed,
        )
        path = export_snomed(
            self.embeddings, self.node_ids, self.snomed_graph,
            self.tmpdir / "keep_snomed.txt",
        )
        lines = path.read_text().strip().split("\n")
        tokens = [line.split()[0] for line in lines]
        self.assertIn("84114007", tokens)  # Heart failure
        self.assertIn("49601007", tokens)  # Cardiovascular


class TestExportIcd(_KeepExportBase):

    def test_maps_snomed_to_icd_tokens(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_icd,
        )
        path = export_icd(
            self.embeddings, self.node_ids, self.icd9_to_snomed,
            self.tmpdir / "keep_icd9.txt",
        )
        lines = path.read_text().strip().split("\n")
        tokens = [line.split()[0] for line in lines]
        self.assertIn("428.0", tokens)
        self.assertIn("401.9", tokens)

    def test_icd_gets_same_vector_as_snomed(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_icd,
        )
        from pyhealth.models.embedding import _iter_text_vectors

        path = export_icd(
            self.embeddings, self.node_ids, self.icd9_to_snomed,
            self.tmpdir / "keep_icd9.txt",
        )
        vectors = _iter_text_vectors(str(path), 4, {"428.0"})
        # 428.0 maps to SNOMED concept 300, which is node_ids[2]
        np.testing.assert_allclose(
            vectors["428.0"].numpy(), self.embeddings[2], atol=1e-5,
        )

    def test_multi_target_icd_averages_embeddings(self):
        """Multi-target ICD codes get the average of their SNOMED vectors."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_icd,
        )
        from pyhealth.models.embedding import _iter_text_vectors

        path = export_icd(
            self.embeddings, self.node_ids, self.icd9_to_snomed_multi,
            self.tmpdir / "keep_icd9.txt",
        )
        vectors = _iter_text_vectors(str(path), 4, {"250.01"})
        # "250.01" maps to [300, 400] — averages their vectors
        expected = (self.embeddings[2] + self.embeddings[3]) / 2.0
        np.testing.assert_allclose(
            vectors["250.01"].numpy(), expected, atol=1e-5,
        )


class TestGenerateMedcodeFiles(_KeepExportBase):

    def test_generate_snomed_csv(self):
        import pandas as pd
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_snomed_csv,
        )

        path = generate_snomed_csv(self.snomed_graph, str(self.tmpdir))
        self.assertTrue(path.exists())
        df = pd.read_csv(path, dtype=str)
        self.assertIn("code", df.columns)
        self.assertIn("name", df.columns)
        self.assertIn("parent_code", df.columns)
        self.assertEqual(len(df), 4)
        hf_row = df[df["code"] == "84114007"]
        self.assertEqual(len(hf_row), 1)
        self.assertEqual(hf_row.iloc[0]["name"], "Heart failure")

    def test_generate_crossmap_csv(self):
        import pandas as pd
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_crossmap_csv,
        )

        path = generate_crossmap_csv(
            self.icd9_to_snomed, self.snomed_graph,
            source_vocabulary="ICD9CM",
            output_dir=str(self.tmpdir),
        )
        self.assertTrue(path.exists())
        df = pd.read_csv(path, dtype=str)
        self.assertIn("ICD9CM", df.columns)
        self.assertIn("SNOMED", df.columns)
        row = df[df["ICD9CM"] == "428.0"]
        self.assertEqual(len(row), 1)
        self.assertEqual(row.iloc[0]["SNOMED"], "84114007")

    def test_generate_all(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_all_medcode_files,
        )

        generate_all_medcode_files(
            self.snomed_graph, self.icd9_to_snomed, output_dir=str(self.tmpdir),
        )
        self.assertTrue((self.tmpdir / "SNOMED.csv").exists())
        self.assertTrue((self.tmpdir / "ICD9CM_to_SNOMED.csv").exists())

    def test_generate_all_with_icd10(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_all_medcode_files,
        )

        icd10_to_snomed = {"I50.9": [300]}
        generate_all_medcode_files(
            self.snomed_graph, self.icd9_to_snomed, icd10_to_snomed,
            output_dir=str(self.tmpdir),
        )
        self.assertTrue((self.tmpdir / "SNOMED.csv").exists())
        self.assertTrue((self.tmpdir / "ICD9CM_to_SNOMED.csv").exists())
        self.assertTrue((self.tmpdir / "ICD10CM_to_SNOMED.csv").exists())


if __name__ == "__main__":
    unittest.main()
