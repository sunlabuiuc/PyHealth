from pyhealth.tasks.clinical_abbreviation import ClinicalAbbreviationTask


def test_task_without_context() -> None:
    task = ClinicalAbbreviationTask(use_context=False)

    sample = {
        "abbr": "SOB",
        "context": "Patient presents with SOB.",
        "label": "shortness of breath",
    }

    result = task(sample)

    assert result["input"] == "SOB"
    assert result["label"] == "shortness of breath"


def test_task_with_context() -> None:
    task = ClinicalAbbreviationTask(use_context=True)

    sample = {
        "abbr": "SOB",
        "context": "Patient presents with SOB.",
        "label": "shortness of breath",
    }

    result = task(sample)

    assert result["input"] == "SOB"
    assert result["label"] == "shortness of breath"