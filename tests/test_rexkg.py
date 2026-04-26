"""Tests for ReXKGModel (NER head, relation head, and KG builder).

Run with:
    pytest tests/test_rexkg.py -v
"""

import pytest
import torch

from pyhealth.models.rexkg import (
    ReXKGModel,
    _NER_ID2LABEL,
    _REL_ID2LABEL,
    NER_LABELS,
    RELATION_TYPES,
)
from pyhealth.tasks.rexkg_extraction import RadiologyKGExtractionTask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    """Instantiate ReXKGModel without a SampleDataset (inference only)."""
    return ReXKGModel(
        dataset=None,
        bert_model_name="bert-base-uncased",
        max_span_length=4,
    )


SAMPLE_REPORTS = [
    "No acute cardiopulmonary process.",
    "Mild cardiomegaly. Small left pleural effusion is noted.",
]


# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------


def test_ner_label_count():
    """NER label map must cover all labels plus the 'O' class."""
    assert len(_NER_ID2LABEL) == len(NER_LABELS) + 1
    assert _NER_ID2LABEL[0] == "O"


def test_rel_label_count():
    """Relation label map must cover all types plus 'none'."""
    assert len(_REL_ID2LABEL) == len(RELATION_TYPES) + 1
    assert _REL_ID2LABEL[0] == "none"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


def test_task_schema():
    task = RadiologyKGExtractionTask()
    assert "text" in task.input_schema
    assert "entities" in task.output_schema
    assert "relations" in task.output_schema


def test_task_skips_short_text():
    """Reports shorter than min_text_length should be excluded."""
    from pyhealth.data import Patient

    task = RadiologyKGExtractionTask(min_text_length=50)

    # Build a minimal mock Patient
    class _MockEvent(dict):
        pass

    class _MockPatient:
        patient_id = "p001"

        def get_events(self, event_type):
            evt = _MockEvent()
            evt["chexpert_plus/section_findings"] = "Short."
            evt["chexpert_plus/section_impression"] = ""
            return [evt]

    samples = task(_MockPatient())
    assert samples == []


# ---------------------------------------------------------------------------
# Model: NER head
# ---------------------------------------------------------------------------


def test_model_predict_entities_returns_list(model):
    """predict_entities should return one list per input report."""
    results = model.predict_entities(SAMPLE_REPORTS, batch_size=2)
    assert len(results) == len(SAMPLE_REPORTS)
    for entity_list in results:
        assert isinstance(entity_list, list)


def test_model_entity_dict_keys(model):
    """Each predicted entity must have the required keys."""
    results = model.predict_entities(SAMPLE_REPORTS[:1])
    for entity in results[0]:
        assert {"start", "end", "text", "type"} == set(entity.keys())
        assert entity["type"] in NER_LABELS


# ---------------------------------------------------------------------------
# Model: RE head
# ---------------------------------------------------------------------------


def test_model_predict_relations_returns_list(model):
    """predict_relations should return one list per input report."""
    entity_lists = model.predict_entities(SAMPLE_REPORTS, batch_size=2)
    rel_lists = model.predict_relations(SAMPLE_REPORTS, entity_lists, batch_size=2)
    assert len(rel_lists) == len(SAMPLE_REPORTS)


def test_model_relation_dict_keys(model):
    """Each predicted relation must have subject, object, and relation keys."""
    entity_lists = model.predict_entities(SAMPLE_REPORTS)
    rel_lists = model.predict_relations(SAMPLE_REPORTS, entity_lists)
    for rel_list in rel_lists:
        for rel in rel_list:
            assert {"subject", "object", "relation"} == set(rel.keys())
            assert rel["relation"] in RELATION_TYPES


# ---------------------------------------------------------------------------
# Model: KG builder
# ---------------------------------------------------------------------------


def test_build_kg_structure(model):
    """build_kg should return a dict with nodes, edges, and subgraphs."""
    kg = model.build_kg(SAMPLE_REPORTS, patient_ids=["p001", "p002"])
    assert "nodes" in kg
    assert "edges" in kg
    assert "subgraphs" in kg
    assert set(kg["subgraphs"].keys()) == {"p001", "p002"}


def test_build_kg_node_deduplication(model):
    """Identical entities across reports should map to the same node."""
    reports = [
        "Cardiomegaly is present.",
        "Cardiomegaly is noted bilaterally.",
    ]
    kg = model.build_kg(reports, patient_ids=["a", "b"])

    matching_nodes = [
        n
        for n in kg["nodes"]
        if str(
            n.get("text")
            or n.get("label")
            or n.get("name")
            or n.get("entity")
            or ""
        ).lower()
        == "cardiomegaly"
    ]
    assert len(matching_nodes) == 1

    shared_node_id = matching_nodes[0]["id"]
    for patient_id in ["a", "b"]:
        subgraph = kg["subgraphs"][patient_id]
        if isinstance(subgraph, dict) and "nodes" in subgraph:
            subgraph_node_ids = [
                node["id"] if isinstance(node, dict) else node
                for node in subgraph["nodes"]
            ]
            assert shared_node_id in subgraph_node_ids
def test_save_kg(model, tmp_path):
    """save_kg should write a valid JSON file."""
    import json

    kg = model.build_kg(SAMPLE_REPORTS[:1])
    out = str(tmp_path / "kg_test.json")
    model.save_kg(kg, out)
    with open(out) as f:
        loaded = json.load(f)
    assert "nodes" in loaded
