"""
Task: Query-Focused EHR Summarization
Authors: Abhitej Bokka (abhitej2), Liam Shen (liams4)
Yields: Sentence-level samples with ICD code relevance labels
"""
def ehr_summarization_task(dataset):
    """
    Generator function that yields extractive summarization samples.
    Each sample contains:
        - patient_id: str
        - visit_id: str
        - sentences: List[str]
        - labels: List[int] (0 or 1)
        - queries: List[str] (ICD codes)

    Args:
        dataset (SampleEHRDataset): EHR dataset processed with sentence-level labels

    Yields:
        dict: Sample dictionary for each note
    """
    for patient in dataset:
        for encounter in patient["encounters"]:
            for note in encounter.get("notes", []):
                yield {
                    "patient_id": patient["patient_id"],
                    "visit_id": encounter["visit_id"],
                    "sentences": note["sentences"],
                    "labels": note["labels"],
                    "queries": encounter["diagnoses"]
                }