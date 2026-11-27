"""Synthetic smoke test for the GiantMed example's offline CLI flow."""

import sys
from types import SimpleNamespace

import torch

from examples.drug_recommendation import giantmed_mimic3 as example


def test_demo_refinement_runs_example_pipeline_offline(monkeypatch, capsys):
    """Exercise CLI orchestration and the real refiner with a stub backbone."""
    raw = {
        name: {
            "patient_id": name,
            "visit_id": "visit-" + name,
            "conditions": [["diagnosis"]],
            "procedures": [["procedure"]],
            "drugs_hist": [[]],
            "drugs": ["A10A"],
        }
        for name in ("train", "validation", "test")
    }

    class Partition:
        def __init__(self, name):
            self.name = name
            self.patient_to_index = {name: [0]}
            self.output_processors = {
                "drugs": SimpleNamespace(label_vocab={"A10A": 0, "C03C": 1})
            }

        def __len__(self):
            return 1

    class Dataset:
        def __init__(self, **kwargs):
            pass

        def set_task(self, task, split):
            return tuple(Partition(name) for name in raw)

        def get_patient(self, patient_id):
            return raw[patient_id]

    class Task:
        def __init__(self, **kwargs):
            pass

        def __call__(self, patient):
            return [patient]

    class Backbone:
        def __init__(self, dataset):
            self.safedrug = SimpleNamespace(ddi_adj=torch.zeros((2, 2)))

        def eval(self):
            pass

        def __call__(self, **batch):
            return {
                "y_prob": torch.tensor([[0.4, 0.6]]),
                "y_true": torch.tensor([[1, 0]]),
            }

    class Trainer:
        def __init__(self, **kwargs):
            pass

        def train(self, *args, **kwargs):
            assert kwargs["epochs"] == 1

    class Terminology:
        def __contains__(self, item):
            return False

    def dataloader(part, **kwargs):
        sample = raw[part.name]
        return [
            {"patient_id": [sample["patient_id"]], "visit_id": [sample["visit_id"]]}
        ]

    def forbid_network(*args, **kwargs):
        raise AssertionError("The demo responder must not make network requests")

    monkeypatch.setattr(example, "MIMIC3Dataset", Dataset)
    monkeypatch.setattr(example, "DrugRecommendationMIMIC3", Task)
    monkeypatch.setattr(example, "SafeDrug", Backbone)
    monkeypatch.setattr(example, "Trainer", Trainer)
    monkeypatch.setattr(example, "get_dataloader", dataloader)
    monkeypatch.setattr(
        example, "load_reference_ndc_atc3", lambda *args: {"ndc": "A10A"}
    )
    monkeypatch.setattr(example, "ndc_coverage", lambda *args: (1, 1))
    monkeypatch.setattr(example, "ATC", Terminology)
    monkeypatch.setattr(example, "ICD9CM", Terminology)
    monkeypatch.setattr(example, "ICD9PROC", Terminology)
    monkeypatch.setattr("urllib.request.urlopen", forbid_network)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "giantmed_mimic3.py",
            "--ndc-rxcui",
            "unused",
            "--ndc-atc",
            "unused",
            "--demo-refinement",
        ],
    )
    example.main()
    out = capsys.readouterr().out
    assert "Using a fake LLM" in out
    assert "'complete': 1" in out
    assert "accepted 2/2 boundary scores; changed 2" in out
    assert "SafeDrug" in out and "GiantMed" in out
