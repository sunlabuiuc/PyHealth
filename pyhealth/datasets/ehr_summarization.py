"""
Dataset: Query-Focused EHR Summarization
Authors: Abhitej Bokka (abhitej2), Liam Shen (liams4)
Paper: Query-Focused EHR Summarization to Aid Imaging Diagnosis
Link: https://arxiv.org/abs/2004.04645
Description: Loads MIMIC-III, splits notes into sentences, and labels each sentence with a binary indicator
             of whether it is relevant to ICD codes appearing in future encounters (30-day horizon).
"""

from pyhealth.datasets import SampleEHRDataset
from datetime import timedelta
import pandas as pd
import spacy

class EHRSummarizationDataset(SampleEHRDataset):
    """Processes MIMIC-III for query-focused summarization using future ICD codes."""

    def process_notes(self):
        """Splits notes into sentences and labels each using future ICD codes."""
        nlp = spacy.load("en_core_web_sm")
        for patient in self.patients.values():
            for encounter in patient["encounters"]:
                enc_date = pd.to_datetime(encounter["date"])
                future_codes = []
                for future_enc in patient["encounters"]:
                    fut_date = pd.to_datetime(future_enc["date"])
                    if fut_date > enc_date + timedelta(days=30):
                        future_codes.extend(future_enc.get("diagnoses", []))
                for note in encounter.get("notes", []):
                    sentences = [sent.text.strip() for sent in nlp(note["text"]).sents]
                    note["sentences"] = sentences
                    note["labels"] = [
                        int(any(code.lower() in sentence.lower() for code in future_codes))
                        for sentence in sentences
                    ]

    def __iter__(self):
        """Allows dataset to be iterable over patients."""
        return iter(self.patients.values())

