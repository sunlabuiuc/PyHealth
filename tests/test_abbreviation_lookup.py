from pyhealth.models.abbreviation_lookup import AbbreviationLookupModel


def test_lookup_model_predicts_known_abbreviation() -> None:
    samples = [
        {"abbr": "SOB", "label": "shortness of breath"},
        {"abbr": "BP", "label": "blood pressure"},
    ]

    model = AbbreviationLookupModel(normalize=True)
    model.fit(samples)

    assert model.predict("SOB") == "shortness of breath"
    assert model.predict("sob") == "shortness of breath"
    assert model.predict("BP!!!") == "blood pressure"