"""Tests for KEEP OMOP graph construction.

Uses small synthetic Athena CSV fixtures (5 concepts, 4 relationships)
to verify graph building and ICD-to-SNOMED mapping logic without
requiring real Athena data or network access.
"""

import tempfile
from pathlib import Path

import networkx as nx
import pytest

# ---------------------------------------------------------------------------
# Synthetic Athena-format fixtures
# ---------------------------------------------------------------------------

# Minimal CONCEPT.csv content (tab-separated, matching Athena format)
# We create a tiny SNOMED hierarchy:
#   Root (100) -> Cardiovascular (200) -> Heart Failure (300)
#                                      -> Hypertension (400)
#              -> Metabolic (500) -> Diabetes (600)
# Plus ICD9 and ICD10 codes that map to SNOMED concepts.
CONCEPT_CSV = (
    "concept_id\tconcept_name\tdomain_id\tvocabulary_id\t"
    "concept_class_id\tstandard_concept\tconcept_code\t"
    "valid_start_date\tvalid_end_date\tinvalid_reason\n"
    # SNOMED Condition concepts (standard)
    "100\tClinical finding\tCondition\tSNOMED\tClinical Finding\tS\t404684003\t20020131\t20991231\t\n"
    "200\tDisorder of cardiovascular system\tCondition\tSNOMED\tDisorder\tS\t49601007\t20020131\t20991231\t\n"
    "300\tHeart failure\tCondition\tSNOMED\tDisorder\tS\t84114007\t20020131\t20991231\t\n"
    "400\tHypertensive disorder\tCondition\tSNOMED\tDisorder\tS\t38341003\t20020131\t20991231\t\n"
    "500\tMetabolic disease\tCondition\tSNOMED\tDisorder\tS\t75934005\t20020131\t20991231\t\n"
    "600\tDiabetes mellitus\tCondition\tSNOMED\tDisorder\tS\t73211009\t20020131\t20991231\t\n"
    # An invalid SNOMED concept (should be filtered out)
    "700\tDeprecated concept\tCondition\tSNOMED\tDisorder\tS\t99999999\t20020131\t20201231\tD\n"
    # ICD9CM concepts
    "1001\tCongestive heart failure\tCondition\tICD9CM\t4-dig nonbill code\t\t428.0\t19700101\t20991231\t\n"
    "1002\tEssential hypertension\tCondition\tICD9CM\t4-dig nonbill code\t\t401.9\t19700101\t20991231\t\n"
    "1003\tDiabetes mellitus type 2\tCondition\tICD9CM\t5-dig billing code\t\t250.00\t19700101\t20991231\t\n"
    # ICD10CM concepts
    "2001\tHeart failure unspecified\tCondition\tICD10CM\t4-char nonbill code\t\tI50.9\t20150101\t20991231\t\n"
    "2002\tType 2 diabetes mellitus\tCondition\tICD10CM\t3-char nonbill code\t\tE11\t20150101\t20991231\t\n"
)

# Minimal CONCEPT_RELATIONSHIP.csv (tab-separated)
RELATIONSHIP_CSV = (
    "concept_id_1\tconcept_id_2\trelationship_id\t"
    "valid_start_date\tvalid_end_date\tinvalid_reason\n"
    # "Is a" edges (child -> parent)
    "200\t100\tIs a\t20020131\t20991231\t\n"  # Cardiovascular Is a Clinical finding
    "300\t200\tIs a\t20020131\t20991231\t\n"  # Heart failure Is a Cardiovascular
    "400\t200\tIs a\t20020131\t20991231\t\n"  # Hypertension Is a Cardiovascular
    "500\t100\tIs a\t20020131\t20991231\t\n"  # Metabolic Is a Clinical finding
    "600\t500\tIs a\t20020131\t20991231\t\n"  # Diabetes Is a Metabolic
    # "Maps to" edges (ICD -> SNOMED)
    "1001\t300\tMaps to\t20020131\t20991231\t\n"  # ICD9 428.0 -> Heart failure
    "1002\t400\tMaps to\t20020131\t20991231\t\n"  # ICD9 401.9 -> Hypertension
    "1003\t600\tMaps to\t20020131\t20991231\t\n"  # ICD9 250.00 -> Diabetes
    "2001\t300\tMaps to\t20020131\t20991231\t\n"  # ICD10 I50.9 -> Heart failure
    "2002\t600\tMaps to\t20020131\t20991231\t\n"  # ICD10 E11 -> Diabetes
    # An invalid relationship (should be filtered)
    "300\t100\tIs a\t20020131\t20201231\tD\n"
)


@pytest.fixture
def athena_dir(tmp_path):
    """Create a temporary directory with synthetic Athena CSV files."""
    concept_path = tmp_path / "CONCEPT.csv"
    concept_path.write_text(CONCEPT_CSV)
    rel_path = tmp_path / "CONCEPT_RELATIONSHIP.csv"
    rel_path.write_text(RELATIONSHIP_CSV)
    return tmp_path


class TestLoadConcepts:
    """Tests for load_concepts()."""

    def test_returns_dataframe_with_required_columns(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )

        df = load_concepts(athena_dir / "CONCEPT.csv")
        assert "concept_id" in df.columns
        assert "concept_name" in df.columns
        assert "vocabulary_id" in df.columns
        assert len(df) > 0

    def test_filters_by_vocabulary(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )

        df = load_concepts(
            athena_dir / "CONCEPT.csv", vocabulary_ids=["SNOMED"]
        )
        assert all(df["vocabulary_id"] == "SNOMED")

    def test_filters_by_domain(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )

        df = load_concepts(
            athena_dir / "CONCEPT.csv", domain_ids=["Condition"]
        )
        assert all(df["domain_id"] == "Condition")

    def test_excludes_invalid_concepts(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )

        df = load_concepts(athena_dir / "CONCEPT.csv")
        # concept_id 700 has invalid_reason="D", should be excluded
        assert 700 not in df["concept_id"].values


class TestLoadRelationships:
    """Tests for load_relationships()."""

    def test_returns_dataframe_with_required_columns(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_relationships,
        )

        df = load_relationships(athena_dir / "CONCEPT_RELATIONSHIP.csv")
        assert "concept_id_1" in df.columns
        assert "concept_id_2" in df.columns
        assert "relationship_id" in df.columns

    def test_filters_by_relationship_type(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_relationships,
        )

        df = load_relationships(
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            relationship_ids=["Is a"],
        )
        assert all(df["relationship_id"] == "Is a")

    def test_excludes_invalid_relationships(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_relationships,
        )

        df = load_relationships(athena_dir / "CONCEPT_RELATIONSHIP.csv")
        # There is 1 invalid "Is a" relationship -- should be excluded
        is_a_edges = df[df["relationship_id"] == "Is a"]
        assert len(is_a_edges) == 5  # 5 valid, 1 invalid excluded


class TestBuildHierarchyGraph:
    """Tests for build_hierarchy_graph()."""

    def test_returns_networkx_digraph(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        assert isinstance(G, nx.DiGraph)

    def test_graph_contains_expected_nodes(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        # Our fixture has 6 valid SNOMED concepts (100-600)
        assert G.number_of_nodes() == 6
        assert 100 in G  # Root
        assert 300 in G  # Heart failure
        assert 600 in G  # Diabetes

    def test_graph_excludes_invalid_concepts(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        assert 700 not in G  # Invalid concept

    def test_edges_are_child_to_parent(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        # Heart failure (300) -> Cardiovascular (200)
        assert G.has_edge(300, 200)
        # Cardiovascular (200) -> Root (100)
        assert G.has_edge(200, 100)

    def test_node_attributes(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        assert G.nodes[300]["concept_name"] == "Heart failure"

    def test_depth_limit(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        # With max_depth=1, only root + its direct children
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            max_depth=1,
        )
        # Root(100), Cardiovascular(200), Metabolic(500) = 3 nodes
        assert G.number_of_nodes() == 3
        assert 100 in G
        assert 200 in G
        assert 500 in G
        # Heart failure (300) is at depth 2, should be excluded
        assert 300 not in G


class TestBuildIcdToSnomed:
    """Tests for build_icd_to_snomed()."""

    def test_icd9_mapping(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        assert mapping["428.0"] == 300   # Heart failure
        assert mapping["401.9"] == 400   # Hypertension
        assert mapping["250.00"] == 600  # Diabetes

    def test_icd10_mapping(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD10CM",
        )
        assert mapping["I50.9"] == 300  # Heart failure
        assert mapping["E11"] == 600    # Diabetes

    def test_filter_to_graph_nodes(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        # Only keep mappings to SNOMED IDs {300, 400} (not 600)
        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
            snomed_concept_ids={300, 400},
        )
        assert "428.0" in mapping   # Maps to 300, which is in our set
        assert "401.9" in mapping   # Maps to 400, which is in our set
        assert "250.00" not in mapping  # Maps to 600, excluded


class TestBuildAllMappings:
    """Tests for build_all_mappings()."""

    def test_returns_both_mappings(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_all_mappings,
        )

        icd9_map, icd10_map = build_all_mappings(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        assert len(icd9_map) == 3   # 3 ICD-9 codes
        assert len(icd10_map) == 2  # 2 ICD-10 codes
