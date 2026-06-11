from pyhealth.datasets.medlingo import MedLingoDataset

# All test samples are synthetic and defined inline to avoid relying on
# external data files or real clinical records.

def test_medlingo_dataset_structure():
    dataset = MedLingoDataset(root=".")

    # synthetic samples inline
    samples = [
        {
            "abbr": "SOB",
            "context": "Patient presents with SOB.",
            "label": "shortness of breath",
        },
        {
            "abbr": "BP",
            "context": "BP remained stable overnight.",
            "label": "blood pressure",
        },
    ]

    # pass samples into dataset
    dataset = MedLingoDataset(samples=samples)

    output = dataset.process()

    # actual assertions to test implementation
    assert len(output) == 2
    assert output[0]["medlingo"][0]["abbr"] == "SOB"
    assert output[1]["medlingo"][0]["label"] == "blood pressure"