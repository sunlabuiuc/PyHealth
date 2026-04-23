"""Tests for KEEP OMOP graph construction.

Uses small synthetic Athena CSV fixtures (5 concepts, 4 relationships)
to verify graph building and ICD-to-SNOMED mapping logic without
requiring real Athena data or network access.
"""

import tempfile
import unittest
from pathlib import Path

import networkx as nx


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
    "1004\tDiabetes with heart complications\tCondition\tICD9CM\t5-dig billing code\t\t250.01\t19700101\t20991231\t\n"
)

RELATIONSHIP_CSV = (
    "concept_id_1\tconcept_id_2\trelationship_id\t"
    "valid_start_date\tvalid_end_date\tinvalid_reason\n"
    # "Is a" edges (child -> parent)
    "200\t100\tIs a\t20020131\t20991231\t\n"
    "300\t200\tIs a\t20020131\t20991231\t\n"
    "400\t200\tIs a\t20020131\t20991231\t\n"
    "500\t100\tIs a\t20020131\t20991231\t\n"
    "600\t500\tIs a\t20020131\t20991231\t\n"
    # Orphan edge: drug-induced HF (800) -> Adverse drug reaction (900, Observation)
    "800\t900\tIs a\t20020131\t20991231\t\n"
    # "Maps to" edges (ICD -> SNOMED)
    "1001\t300\tMaps to\t20020131\t20991231\t\n"
    "1002\t400\tMaps to\t20020131\t20991231\t\n"
    "1003\t600\tMaps to\t20020131\t20991231\t\n"
    "2001\t300\tMaps to\t20020131\t20991231\t\n"
    "2002\t600\tMaps to\t20020131\t20991231\t\n"
    # Multi-target: ICD9 250.01 maps to BOTH Diabetes (600) AND Heart failure (300).
    "1004\t600\tMaps to\t20020131\t20991231\t\n"
    "1004\t300\tMaps to\t20020131\t20991231\t\n"
    # An invalid relationship (should be filtered)
    "300\t100\tIs a\t20020131\t20201231\tD\n"
)

ANCESTOR_CSV = (
    "ancestor_concept_id\tdescendant_concept_id\t"
    "min_levels_of_separation\tmax_levels_of_separation\n"
    # Ancestors of 800 (the orphan) — only these matter for rescue
    "900\t800\t1\t1\n"
    "300\t800\t2\t2\n"
    "200\t800\t3\t3\n"
    "100\t800\t4\t4\n"
    # Other ancestor relationships (for completeness)
    "100\t200\t1\t1\n"
    "100\t300\t2\t2\n"
    "100\t400\t2\t2\n"
    "100\t500\t1\t1\n"
    "100\t600\t2\t2\n"
    "200\t300\t1\t1\n"
    "200\t400\t1\t1\n"
    "500\t600\t1\t1\n"
)


class AthenaFixtureMixin:
    """Shared setUp for tests that need a synthetic Athena directory."""

    def setUp(self):
        """Create a temp dir with CONCEPT/CONCEPT_RELATIONSHIP/CONCEPT_ANCESTOR CSVs."""
        self._tmp = tempfile.TemporaryDirectory()
        self.athena_dir = Path(self._tmp.name)
        (self.athena_dir / "CONCEPT.csv").write_text(CONCEPT_CSV)
        (self.athena_dir / "CONCEPT_RELATIONSHIP.csv").write_text(RELATIONSHIP_CSV)
        (self.athena_dir / "CONCEPT_ANCESTOR.csv").write_text(ANCESTOR_CSV)

    def tearDown(self):
        self._tmp.cleanup()


class TestLoadConcepts(AthenaFixtureMixin, unittest.TestCase):
    """Tests for load_concepts()."""

    def test_returns_dataframe_with_required_columns(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )
        df = load_concepts(self.athena_dir / "CONCEPT.csv")
        self.assertIn("concept_id", df.columns)
        self.assertIn("concept_name", df.columns)
        self.assertIn("vocabulary_id", df.columns)
        self.assertGreater(len(df), 0)

    def test_filters_by_vocabulary(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )
        df = load_concepts(
            self.athena_dir / "CONCEPT.csv", vocabulary_ids=["SNOMED"]
        )
        self.assertTrue(all(df["vocabulary_id"] == "SNOMED"))

    def test_filters_by_domain(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )
        df = load_concepts(
            self.athena_dir / "CONCEPT.csv", domain_ids=["Condition"]
        )
        self.assertTrue(all(df["domain_id"] == "Condition"))

    def test_excludes_invalid_concepts(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_concepts,
        )
        df = load_concepts(self.athena_dir / "CONCEPT.csv")
        # concept_id 700 has invalid_reason="D", should be excluded
        self.assertNotIn(700, df["concept_id"].values)


class TestLoadRelationships(AthenaFixtureMixin, unittest.TestCase):
    """Tests for load_relationships()."""

    def test_returns_dataframe_with_required_columns(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_relationships,
        )
        df = load_relationships(self.athena_dir / "CONCEPT_RELATIONSHIP.csv")
        self.assertIn("concept_id_1", df.columns)
        self.assertIn("concept_id_2", df.columns)
        self.assertIn("relationship_id", df.columns)

    def test_filters_by_relationship_type(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_relationships,
        )
        df = load_relationships(
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            relationship_ids=["Is a"],
        )
        self.assertTrue(all(df["relationship_id"] == "Is a"))

    def test_excludes_invalid_relationships(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            load_relationships,
        )
        df = load_relationships(self.athena_dir / "CONCEPT_RELATIONSHIP.csv")
        is_a_edges = df[df["relationship_id"] == "Is a"]
        # 6 valid "Is a" edges (5 in main hierarchy + 1 orphan edge 800->900),
        # 1 invalid excluded.
        self.assertEqual(len(is_a_edges), 6)


class TestBuildHierarchyGraph(AthenaFixtureMixin, unittest.TestCase):
    """Tests for build_hierarchy_graph()."""

    def test_returns_networkx_digraph(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertIsInstance(G, nx.DiGraph)

    def test_graph_contains_expected_nodes(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertEqual(G.number_of_nodes(), 6)
        self.assertIn(100, G)
        self.assertIn(300, G)
        self.assertIn(600, G)

    def test_graph_excludes_invalid_concepts(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertNotIn(700, G)

    def test_edges_are_child_to_parent(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertTrue(G.has_edge(300, 200))
        self.assertTrue(G.has_edge(200, 100))

    def test_node_attributes(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertEqual(G.nodes[300]["concept_name"], "Heart failure")

    def test_depth_limit(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            max_depth=1,
            root_concept_id=100,
        )
        # Root(100), Cardiovascular(200), Metabolic(500) = 3 nodes
        self.assertEqual(G.number_of_nodes(), 3)
        self.assertIn(100, G)
        self.assertIn(200, G)
        self.assertIn(500, G)
        # Heart failure (300) is at depth 2, should be excluded
        self.assertNotIn(300, G)


class TestBuildHierarchyGraphSingleRoot(AthenaFixtureMixin, unittest.TestCase):
    """Tests for paper-faithful single-root graph construction."""

    def test_root_concept_is_in_graph(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertIn(100, G.nodes())

    def test_single_root_only(self):
        """Graph must have exactly one root node (out-degree 0)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        roots = [n for n in G.nodes() if G.out_degree(n) == 0]
        self.assertEqual(roots, [100])

    def test_all_nodes_reachable_from_root(self):
        """Every node must be reachable from the root via BFS."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        reverse = G.reverse()
        reachable = {100} | nx.descendants(reverse, 100)
        self.assertEqual(reachable, set(G.nodes()))

    def test_excludes_nodes_outside_root_subtree(self):
        """Nodes not descended from the root must be excluded."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=200,
        )
        self.assertEqual(set(G.nodes()), {200, 300, 400})
        self.assertNotIn(500, G.nodes())
        self.assertNotIn(600, G.nodes())

    def test_default_root_is_paper_disease_concept(self):
        """Default root should be 4274025 per paper Appendix A.1.1."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        self.assertEqual(G.number_of_nodes(), 0)


class TestOrphanRescue(AthenaFixtureMixin, unittest.TestCase):
    """Tests for orphan rescue via CONCEPT_ANCESTOR."""

    def test_orphan_not_in_graph_without_rescue(self):
        """Sanity check: without rescue, orphan is excluded."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
        )
        self.assertNotIn(800, G.nodes())

    def test_orphan_rescued_with_ancestor_csv(self):
        """With CONCEPT_ANCESTOR, orphan is rescued."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
            ancestor_csv=self.athena_dir / "CONCEPT_ANCESTOR.csv",
        )
        self.assertIn(800, G.nodes())

    def test_rescue_edge_points_to_closest_ancestor(self):
        """Rescue edge must go to the closest in-graph ancestor."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
            ancestor_csv=self.athena_dir / "CONCEPT_ANCESTOR.csv",
        )
        self.assertTrue(G.has_edge(800, 300))

    def test_rescued_orphan_reachable_from_root(self):
        """After rescue, every node is reachable from root."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_hierarchy_graph,
        )
        G = build_hierarchy_graph(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            root_concept_id=100,
            ancestor_csv=self.athena_dir / "CONCEPT_ANCESTOR.csv",
        )
        reverse = G.reverse()
        reachable = {100} | nx.descendants(reverse, 100)
        self.assertIn(800, reachable)


class TestBuildIcdToSnomed(AthenaFixtureMixin, unittest.TestCase):
    """Tests for build_icd_to_snomed() with multi-target support."""

    def test_icd9_mapping_returns_lists(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        for icd, snomeds in mapping.items():
            self.assertIsInstance(snomeds, list, f"{icd} value is not a list")
            self.assertTrue(all(isinstance(s, int) for s in snomeds))
            self.assertGreater(len(snomeds), 0)

    def test_icd9_single_target_codes(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        self.assertEqual(mapping["428.0"], [300])
        self.assertEqual(mapping["401.9"], [400])
        self.assertEqual(mapping["250.00"], [600])

    def test_icd9_multi_target_code(self):
        """ICD9 250.01 maps to BOTH Diabetes (600) AND Heart failure (300)."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        self.assertEqual(mapping["250.01"], [300, 600])

    def test_icd9_mapping_is_sorted(self):
        """List entries must be sorted for reproducibility."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
        )
        for icd, snomeds in mapping.items():
            self.assertEqual(snomeds, sorted(snomeds), f"{icd} list not sorted")

    def test_icd10_mapping(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD10CM",
        )
        self.assertEqual(mapping["I50.9"], [300])
        self.assertEqual(mapping["E11"], [600])

    def test_filter_to_graph_nodes_keeps_partial_multi_target(self):
        """When filter is applied, multi-target codes may lose some targets."""
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
            snomed_concept_ids={300, 400},
        )
        self.assertEqual(mapping["428.0"], [300])
        self.assertEqual(mapping["401.9"], [400])
        self.assertNotIn("250.00", mapping)
        self.assertEqual(mapping["250.01"], [300])

    def test_filter_drops_code_when_no_targets_match(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_icd_to_snomed,
        )
        mapping = build_icd_to_snomed(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
            source_vocabulary="ICD9CM",
            snomed_concept_ids={400},
        )
        self.assertIn("401.9", mapping)
        self.assertNotIn("428.0", mapping)
        self.assertNotIn("250.00", mapping)
        self.assertNotIn("250.01", mapping)


class TestBuildAllMappings(AthenaFixtureMixin, unittest.TestCase):
    """Tests for build_all_mappings()."""

    def test_returns_both_mappings(self):
        from pyhealth.medcode.pretrained_embeddings.keep_emb.build_omop_graph import (
            build_all_mappings,
        )
        icd9_map, icd10_map = build_all_mappings(
            self.athena_dir / "CONCEPT.csv",
            self.athena_dir / "CONCEPT_RELATIONSHIP.csv",
        )
        self.assertEqual(len(icd9_map), 4)
        self.assertEqual(len(icd10_map), 2)


if __name__ == "__main__":
    unittest.main()
