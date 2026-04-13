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
    # ORPHAN: Condition concept whose only direct parent is in Observation domain.
    # Its only "Is a" edge points to 900 (Observation), which gets filtered out
    # by our domain filter. Without rescue, 800 is unreachable from root.
    # With rescue via CONCEPT_ANCESTOR: 800's closest in-graph ancestor is 300
    # (Heart failure) at distance 2, so a rescue edge 800 -> 300 is added.
    "800\tDrug-induced heart failure\tCondition\tSNOMED\tDisorder\tS\t88888888\t20020131\t20991231\t\n"
    "900\tAdverse drug reaction\tObservation\tSNOMED\tObservation\tS\t282100009\t20020131\t20991231\t\n"
    # ICD9CM concepts
    "1001\tCongestive heart failure\tCondition\tICD9CM\t4-dig nonbill code\t\t428.0\t19700101\t20991231\t\n"
    "1002\tEssential hypertension\tCondition\tICD9CM\t4-dig nonbill code\t\t401.9\t19700101\t20991231\t\n"
    "1003\tDiabetes mellitus type 2\tCondition\tICD9CM\t5-dig billing code\t\t250.00\t19700101\t20991231\t\n"
    # ICD10CM concepts
    "2001\tHeart failure unspecified\tCondition\tICD10CM\t4-char nonbill code\t\tI50.9\t20150101\t20991231\t\n"
    "2002\tType 2 diabetes mellitus\tCondition\tICD10CM\t3-char nonbill code\t\tE11\t20150101\t20991231\t\n"
    # Multi-target ICD: combination code that maps to TWO SNOMED concepts.
    # Example pattern: ICD code "A01.04 Typhoid arthritis" maps to BOTH
    # "Typhoid fever" AND "Secondary inflammatory arthritis" in Athena.
    # Here we use "250.01" as a synthetic multi-target code mapping to
    # both Diabetes (600) and Heart failure (300).
    "1004\tDiabetes with heart complications\tCondition\tICD9CM\t5-dig billing code\t\t250.01\t19700101\t20991231\t\n"
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
    # Orphan edge: drug-induced HF (800) -> Adverse drug reaction (900, Observation)
    # This edge gets filtered by domain filter, orphaning 800.
    "800\t900\tIs a\t20020131\t20991231\t\n"
    # "Maps to" edges (ICD -> SNOMED)
    "1001\t300\tMaps to\t20020131\t20991231\t\n"  # ICD9 428.0 -> Heart failure
    "1002\t400\tMaps to\t20020131\t20991231\t\n"  # ICD9 401.9 -> Hypertension
    "1003\t600\tMaps to\t20020131\t20991231\t\n"  # ICD9 250.00 -> Diabetes
    "2001\t300\tMaps to\t20020131\t20991231\t\n"  # ICD10 I50.9 -> Heart failure
    "2002\t600\tMaps to\t20020131\t20991231\t\n"  # ICD10 E11 -> Diabetes
    # Multi-target: ICD9 250.01 "Diabetes with heart complications"
    # maps to BOTH Diabetes (600) AND Heart failure (300).
    "1004\t600\tMaps to\t20020131\t20991231\t\n"
    "1004\t300\tMaps to\t20020131\t20991231\t\n"
    # An invalid relationship (should be filtered)
    "300\t100\tIs a\t20020131\t20201231\tD\n"
)

# Minimal CONCEPT_ANCESTOR.csv (tab-separated).
# Contains transitive closure of "Is a" relationships for orphan rescue.
# The orphan 800 "Drug-induced heart failure" has these ancestors:
#   - 900 "Adverse drug reaction" (Observation, distance 1) — filtered
#   - 300 "Heart failure" (Condition, distance 2 via a conceptual path) — rescue target
#   - 200 "Cardiovascular" (distance 3)
#   - 100 "Clinical finding" (distance 4)
# The closest in-graph ancestor is 300, so rescue adds edge 800 -> 300.
ANCESTOR_CSV = (
    "ancestor_concept_id\tdescendant_concept_id\t"
    "min_levels_of_separation\tmax_levels_of_separation\n"
    # Ancestors of 800 (the orphan) — only these matter for rescue
    "900\t800\t1\t1\n"   # direct parent (Observation, filtered)
    "300\t800\t2\t2\n"   # rescue target (Condition, in graph)
    "200\t800\t3\t3\n"   # further ancestor
    "100\t800\t4\t4\n"   # root's grandchild
    # Other ancestor relationships (for completeness, not strictly needed for tests)
    "100\t200\t1\t1\n"
    "100\t300\t2\t2\n"
    "100\t400\t2\t2\n"
    "100\t500\t1\t1\n"
    "100\t600\t2\t2\n"
    "200\t300\t1\t1\n"
    "200\t400\t1\t1\n"
    "500\t600\t1\t1\n"
)


@pytest.fixture
def athena_dir(tmp_path):
    """Create a temporary directory with synthetic Athena CSV files."""
    concept_path = tmp_path / "CONCEPT.csv"
    concept_path.write_text(CONCEPT_CSV)
    rel_path = tmp_path / "CONCEPT_RELATIONSHIP.csv"
    rel_path.write_text(RELATIONSHIP_CSV)
    ancestor_path = tmp_path / "CONCEPT_ANCESTOR.csv"
    ancestor_path.write_text(ANCESTOR_CSV)
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
        # 6 valid "Is a" edges (5 in main hierarchy + 1 orphan edge 800->900),
        # 1 invalid excluded.
        assert len(is_a_edges) == 6


class TestBuildHierarchyGraph:
    """Tests for build_hierarchy_graph().

    Uses root_concept_id=100 (our fixture's root) as a stand-in for
    the paper's 4274025 "Disease" concept.
    """

    def test_returns_networkx_digraph(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        assert isinstance(G, nx.DiGraph)

    def test_graph_contains_expected_nodes(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        # Our fixture has 6 valid SNOMED concepts (100-600) all under root 100
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
            root_concept_id=100,
        )
        assert 700 not in G  # Invalid concept

    def test_edges_are_child_to_parent(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
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
            root_concept_id=100,
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
            root_concept_id=100,
        )
        # Root(100), Cardiovascular(200), Metabolic(500) = 3 nodes
        assert G.number_of_nodes() == 3
        assert 100 in G
        assert 200 in G
        assert 500 in G
        # Heart failure (300) is at depth 2, should be excluded
        assert 300 not in G


class TestBuildHierarchyGraphSingleRoot:
    """Tests for paper-faithful single-root graph construction.

    Paper Appendix A.1.1: root is concept_id 4274025 "Disease".
    We BFS from this single concept, not from every node with
    out-degree 0. Graph must be a single connected DAG.
    """

    def test_root_concept_is_in_graph(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        # Using 100 as the "root" in our fixture (stand-in for 4274025)
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        assert 100 in G.nodes()

    def test_single_root_only(self, athena_dir):
        """Graph must have exactly one root node (out-degree 0)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        roots = [n for n in G.nodes() if G.out_degree(n) == 0]
        assert roots == [100], f"Expected single root [100], got {roots}"

    def test_all_nodes_reachable_from_root(self, athena_dir):
        """Every node must be reachable from the root via BFS.

        This catches the orphan bug: concepts whose only parent is in a
        different domain would be unreachable from the root in a naive
        implementation.
        """
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        # Reverse graph so root -> descendants direction
        reverse = G.reverse()
        reachable = {100} | nx.descendants(reverse, 100)
        assert reachable == set(G.nodes()), (
            f"{len(set(G.nodes()) - reachable)} nodes unreachable from root"
        )

    def test_excludes_nodes_outside_root_subtree(self, athena_dir):
        """Nodes not descended from the root must be excluded.

        Using root_concept_id=200 (Cardiovascular) should give us only
        {200, 300, 400} — the Metabolic subtree (500, 600) must be excluded.
        """
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=200,  # Cardiovascular subtree only
        )
        assert set(G.nodes()) == {200, 300, 400}
        assert 500 not in G.nodes()  # Metabolic root
        assert 600 not in G.nodes()  # Diabetes (under Metabolic)

    def test_default_root_is_paper_disease_concept(self, athena_dir):
        """Default root should be 4274025 per paper Appendix A.1.1.

        Since our fixture doesn't contain 4274025, the graph should
        be empty (or raise) when no explicit root is passed and 4274025
        is not in the CSV.
        """
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )

        # Default root=4274025 doesn't exist in our fixture → empty graph
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        assert G.number_of_nodes() == 0


class TestOrphanRescue:
    """Tests for orphan rescue via CONCEPT_ANCESTOR.

    Some SNOMED Condition concepts have direct "Is a" parents only in
    the Observation domain. Our domain-filtered BFS can't reach them.
    The rescue step uses CONCEPT_ANCESTOR to find each orphan's closest
    in-graph ancestor and adds a direct edge.

    Our fixture has orphan concept 800 "Drug-induced heart failure"
    whose only direct parent (900 "Adverse drug reaction") is in the
    Observation domain. Without rescue, 800 is unreachable from root
    100. With rescue, 800 gets a rescue edge to its closest in-graph
    ancestor: 300 "Heart failure" (distance 2 in CONCEPT_ANCESTOR).
    """

    def test_orphan_not_in_graph_without_rescue(self, athena_dir):
        """Sanity check: without rescue, orphan is excluded."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        # No CONCEPT_ANCESTOR passed → no rescue
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        assert 800 not in G.nodes()

    def test_orphan_rescued_with_ancestor_csv(self, athena_dir):
        """With CONCEPT_ANCESTOR, orphan is rescued."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
            ancestor_csv=athena_dir / "CONCEPT_ANCESTOR.csv",
        )
        assert 800 in G.nodes(), "Orphan 800 should be rescued"

    def test_rescue_edge_points_to_closest_ancestor(self, athena_dir):
        """Rescue edge must go to the closest in-graph ancestor."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
            ancestor_csv=athena_dir / "CONCEPT_ANCESTOR.csv",
        )
        # 800's ancestors in graph: 300 (dist 2), 200 (dist 3), 100 (dist 4)
        # Closest is 300 → edge 800 -> 300 must exist
        assert G.has_edge(800, 300), "Rescue edge 800 -> 300 missing"

    def test_rescued_orphan_reachable_from_root(self, athena_dir):
        """After rescue, every node is reachable from root."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
            ancestor_csv=athena_dir / "CONCEPT_ANCESTOR.csv",
        )
        reverse = G.reverse()
        reachable = {100} | nx.descendants(reverse, 100)
        assert 800 in reachable, "Rescued orphan unreachable from root"


class TestBuildIcdToSnomed:
    """Tests for build_icd_to_snomed() with multi-target support.

    Returns Dict[str, List[int]] where each ICD code maps to a sorted
    list of SNOMED concept IDs. This handles combination codes like
    "A01.04 Typhoid arthritis" that map to multiple atomic SNOMED
    concepts (typhoid fever + inflammatory arthritis).
    """

    def test_icd9_mapping_returns_lists(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        # Every value is a non-empty list of ints
        for icd, snomeds in mapping.items():
            assert isinstance(snomeds, list), f"{icd} value is not a list"
            assert all(isinstance(s, int) for s in snomeds)
            assert len(snomeds) > 0

    def test_icd9_single_target_codes(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        # Single-target codes wrap the one target in a list
        assert mapping["428.0"] == [300]   # Heart failure
        assert mapping["401.9"] == [400]   # Hypertension
        assert mapping["250.00"] == [600]  # Diabetes

    def test_icd9_multi_target_code(self, athena_dir):
        """ICD9 250.01 maps to BOTH Diabetes (600) AND Heart failure (300)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        # Multi-target code preserves both; list is sorted for determinism
        assert mapping["250.01"] == [300, 600]

    def test_icd9_mapping_is_sorted(self, athena_dir):
        """List entries must be sorted for reproducibility."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        for icd, snomeds in mapping.items():
            assert snomeds == sorted(snomeds), f"{icd} list not sorted"

    def test_icd10_mapping(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD10CM",
        )
        assert mapping["I50.9"] == [300]  # Heart failure
        assert mapping["E11"] == [600]    # Diabetes

    def test_filter_to_graph_nodes_keeps_partial_multi_target(self, athena_dir):
        """When filter is applied, multi-target codes may lose some targets."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        # 250.01 maps to [300, 600]. Filter to {300, 400} means we keep
        # only the 300 target; 600 is filtered out.
        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
            snomed_concept_ids={300, 400},
        )
        assert mapping["428.0"] == [300]    # Maps to 300, in set
        assert mapping["401.9"] == [400]    # Maps to 400, in set
        assert "250.00" not in mapping       # Maps to 600, excluded entirely
        assert mapping["250.01"] == [300]   # Multi-target: kept 300, dropped 600

    def test_filter_drops_code_when_no_targets_match(self, athena_dir):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )

        # Filter to {400} only — drops everything that doesn't map to 400
        mapping = build_icd_to_snomed(
            athena_dir / "CONCEPT.csv",
            athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
            snomed_concept_ids={400},
        )
        assert "401.9" in mapping   # Maps to 400
        assert "428.0" not in mapping
        assert "250.00" not in mapping
        assert "250.01" not in mapping


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
        assert len(icd9_map) == 4   # 4 ICD-9 codes (3 single + 1 multi-target)
        assert len(icd10_map) == 2  # 2 ICD-10 codes
