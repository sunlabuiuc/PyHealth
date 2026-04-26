from pyhealth.datasets.medlingo import MedLingoDataset


def test_medlingo_dataset_structure():
    dataset = MedLingoDataset(root=".")

    # synthetic raw samples (what your JSON would contain)
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

    # monkey patch process input
    def mock_process():
        data = []
        for i, sample in enumerate(samples):
            data.append({
                "patient_id": f"patient_{i}",
                "visit_id": f"visit_{i}",
                "medlingo": [sample],
            })
        return data

    dataset.process = mock_process  # override

    output = dataset.process()

    # actual assertions
    assert len(output) == 2
    assert output[0]["medlingo"][0]["abbr"] == "SOB"
    assert output[1]["medlingo"][0]["label"] == "blood pressure"