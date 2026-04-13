"""Tests for KEEP embedding export and medcode file generation.

Tests both the text-vector export (keep_snomed.txt, keep_icd9.txt)
and the medcode CSV generation (SNOMED.csv, ICD9CM_to_SNOMED.csv)
using small synthetic data.
"""

import os
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pytest


@pytest.fixture
def snomed_graph():
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


@pytest.fixture
def embeddings():
    """Tiny 3-node embedding matrix."""
    np.random.seed(42)
    return np.random.randn(4, 4).astype(np.float32)


@pytest.fixture
def node_ids():
    return [100, 200, 300, 400]


@pytest.fixture
def icd9_to_snomed():
    """Multi-target format: ICD code -> list of SNOMED IDs."""
    return {"428.0": [300], "401.9": [400]}


@pytest.fixture
def icd9_to_snomed_multi():
    """Fixture with a multi-target code for testing averaging."""
    # "250.01" maps to both 300 (Heart failure) and 400 (Hypertension)
    return {
        "428.0": [300],
        "401.9": [400],
        "250.01": [300, 400],
    }


# ---------------------------------------------------------------------------
# Test export_to_text
# ---------------------------------------------------------------------------

class TestExportToText:

    def test_creates_file(self, embeddings):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_to_text,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_text(
                embeddings[:2],
                ["token_a", "token_b"],
                Path(tmpdir) / "test.txt",
            )
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert lines[0].startswith("token_a ")
            assert len(lines[0].split()) == 5  # token + 4 dims

    def test_round_trip_with_pyhealth(self, embeddings):
        """Verify exported file can be loaded by init_embedding_with_pretrained."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_to_text,
        )

        tokens = ["84114007", "38341003"]
        emb = embeddings[:2]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_to_text(emb, tokens, Path(tmpdir) / "test.txt")

            # Simulate what init_embedding_with_pretrained does
            from pyhealth.models.embedding import _iter_text_vectors

            vectors = _iter_text_vectors(
                str(path),
                embedding_dim=4,
                wanted_tokens={"84114007", "38341003"},
            )
            assert "84114007" in vectors
            assert "38341003" in vectors
            np.testing.assert_allclose(
                vectors["84114007"].numpy(), emb[0], atol=1e-5,
            )


# ---------------------------------------------------------------------------
# Test export_snomed
# ---------------------------------------------------------------------------

class TestExportSnomed:

    def test_uses_concept_codes_as_tokens(
        self, embeddings, node_ids, snomed_graph,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_snomed,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_snomed(
                embeddings, node_ids, snomed_graph,
                Path(tmpdir) / "keep_snomed.txt",
            )
            lines = path.read_text().strip().split("\n")
            tokens = [line.split()[0] for line in lines]
            # Should use concept_code, not concept_id
            assert "84114007" in tokens  # Heart failure
            assert "49601007" in tokens  # Cardiovascular


# ---------------------------------------------------------------------------
# Test export_icd
# ---------------------------------------------------------------------------

class TestExportIcd:

    def test_maps_snomed_to_icd_tokens(
        self, embeddings, node_ids, icd9_to_snomed,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_icd,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_icd(
                embeddings, node_ids, icd9_to_snomed,
                Path(tmpdir) / "keep_icd9.txt",
            )
            lines = path.read_text().strip().split("\n")
            tokens = [line.split()[0] for line in lines]
            assert "428.0" in tokens
            assert "401.9" in tokens

    def test_icd_gets_same_vector_as_snomed(
        self, embeddings, node_ids, icd9_to_snomed,
    ):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_icd, export_to_text,
        )
        from pyhealth.models.embedding import _iter_text_vectors

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_icd(
                embeddings, node_ids, icd9_to_snomed,
                Path(tmpdir) / "keep_icd9.txt",
            )
            vectors = _iter_text_vectors(str(path), 4, {"428.0"})
            # 428.0 maps to SNOMED concept 300, which is node_ids[2]
            np.testing.assert_allclose(
                vectors["428.0"].numpy(), embeddings[2], atol=1e-5,
            )

    def test_multi_target_icd_averages_embeddings(
        self, embeddings, node_ids, icd9_to_snomed_multi,
    ):
        """Multi-target ICD codes get the average of their SNOMED vectors."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.export_embeddings import (
            export_icd,
        )
        from pyhealth.models.embedding import _iter_text_vectors

        with tempfile.TemporaryDirectory() as tmpdir:
            path = export_icd(
                embeddings, node_ids, icd9_to_snomed_multi,
                Path(tmpdir) / "keep_icd9.txt",
            )
            vectors = _iter_text_vectors(str(path), 4, {"250.01"})
            # "250.01" maps to [300, 400] — averages their vectors
            # 300 is node_ids[2], 400 is node_ids[3]
            expected = (embeddings[2] + embeddings[3]) / 2.0
            np.testing.assert_allclose(
                vectors["250.01"].numpy(), expected, atol=1e-5,
            )


# ---------------------------------------------------------------------------
# Test generate_medcode_files
# ---------------------------------------------------------------------------

class TestGenerateMedcodeFiles:

    def test_generate_snomed_csv(self, snomed_graph):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_snomed_csv,
        )
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_snomed_csv(snomed_graph, tmpdir)
            assert path.exists()
            df = pd.read_csv(path, dtype=str)
            assert "code" in df.columns
            assert "name" in df.columns
            assert "parent_code" in df.columns
            assert len(df) == 4
            # Heart failure's code should be "84114007"
            hf_row = df[df["code"] == "84114007"]
            assert len(hf_row) == 1
            assert hf_row.iloc[0]["name"] == "Heart failure"

    def test_generate_crossmap_csv(self, snomed_graph, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_crossmap_csv,
        )
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_crossmap_csv(
                icd9_to_snomed, snomed_graph,
                source_vocabulary="ICD9CM",
                output_dir=tmpdir,
            )
            assert path.exists()
            df = pd.read_csv(path, dtype=str)
            assert "ICD9CM" in df.columns
            assert "SNOMED" in df.columns
            # 428.0 should map to concept_code "84114007"
            row = df[df["ICD9CM"] == "428.0"]
            assert len(row) == 1
            assert row.iloc[0]["SNOMED"] == "84114007"

    def test_generate_all(self, snomed_graph, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_all_medcode_files,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_all_medcode_files(
                snomed_graph, icd9_to_snomed, output_dir=tmpdir,
            )
            assert (Path(tmpdir) / "SNOMED.csv").exists()
            assert (Path(tmpdir) / "ICD9CM_to_SNOMED.csv").exists()

    def test_generate_all_with_icd10(self, snomed_graph, icd9_to_snomed):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.generate_medcode_files import (
            generate_all_medcode_files,
        )

        icd10_to_snomed = {"I50.9": [300]}

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_all_medcode_files(
                snomed_graph, icd9_to_snomed, icd10_to_snomed,
                output_dir=tmpdir,
            )
            assert (Path(tmpdir) / "SNOMED.csv").exists()
            assert (Path(tmpdir) / "ICD9CM_to_SNOMED.csv").exists()
            assert (Path(tmpdir) / "ICD10CM_to_SNOMED.csv").exists()
