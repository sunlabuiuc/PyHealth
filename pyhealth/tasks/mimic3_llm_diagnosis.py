"""
Author: Stephen Moy
NetID: moy26
Description:
    This task aggregates MIMIC-III clinical notes by patient and applies a large language model
    (e.g., FLAN-T5) to classify whether the patient has a specified diagnosis. It integrates
    preprocessing, patient-level aggregation, and LLM-based classification into a single PyHealth
    Task class for reproducibility and open-source contribution.
"""
import pandas as pd
import torch
import math
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pyhealth.tasks import BaseTask

class MIMIC3LLMDiagnosisTask(BaseTask):
    """Aggregate MIMIC-III notes by patient and diagnose using an LLM."""

    def __init__(self, notes_path: str, diagnosis: str,
                 model_name: str = "google/flan-t5-large", device: str = "cuda"):
        """
        Initialize the MIMIC-III LLM diagnosis task.

        Args:
            notes_path (str): Path to the NOTEEVENTS.csv file from MIMIC-III.
            diagnosis (str): The diagnosis to classify (e.g., "heart failure").
            model_name (str): Hugging Face model name for the LLM.
            device (str): Device to run the model on ("cuda" or "cpu").

        Attributes:
            notes_df (pd.DataFrame): Raw notes loaded from CSV.
            patients_df (pd.DataFrame): Aggregated notes per patient.
            tokenizer (AutoTokenizer): Hugging Face tokenizer.
            model (AutoModelForSeq2SeqLM): Hugging Face seq2seq model.
        """
        super().__init__()
        self.notes_path = notes_path
        self.diagnosis = diagnosis
        self.device = device

        # Load data
        self.notes_df = pd.read_csv(notes_path, low_memory=False)
        self.patients_df = self.aggregate_by_patient()

        # Load LLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.model.eval()

    def aggregate_by_patient(self, text_col: str = "TEXT", patient_col: str = "SUBJECT_ID") -> pd.DataFrame:
        """
        Aggregate notes per patient into a single string.

        Args:
            text_col (str): Column name containing note text. Defaults to "TEXT".
            patient_col (str): Column name containing patient IDs. Defaults to "SUBJECT_ID".

        Returns:
            pd.DataFrame: DataFrame with columns:
                - SUBJECT_ID (int): Patient identifier.
                - PATIENT_NOTES (str): Concatenated notes for the patient.

        Example:
            >>> dataset = MIMIC3LLMDiagnosisTask("NOTEEVENTS.csv", "heart failure")
            >>> patients = dataset.aggregate_by_patient()
            >>> print(patients.head())
        """
        agg_df = (
            self.notes_df.groupby(patient_col)[text_col]
            .apply(lambda x: " \n\n ".join(x.dropna()))
            .reset_index()
            .rename(columns={text_col: "PATIENT_NOTES"})
        )
        return agg_df

    def classify_patient(self, patient_notes: str) -> dict:
        """
        Classify whether a patient has the specified diagnosis using the LLM.

        Args:
            patient_notes (str): Concatenated notes for a single patient.

        Returns:
            dict: Dictionary with keys:
                - "diagnosis" (str): The diagnosis being tested.
                - "verdict" (str): "YES" or "NO".
                - "confidence" (float): Confidence score in [0,1].

        Example:
            >>> task = MIMIC3LLMDiagnosisTask("NOTEEVENTS.csv", "heart failure")
            >>> result = task.classify_patient("Patient has CHF and hypertension.")
            >>> print(result)
            {'diagnosis': 'heart failure', 'verdict': 'YES', 'confidence': 0.92}
        """
        prompt = (
            f"Read the following clinical notes:\n\n{patient_notes}\n\n"
            f"Question: Does this patient have {self.diagnosis}?\n"
            f"Answer with YES or NO."
        )
        enc = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)

        def score_candidate(target: str):
            labels = self.tokenizer(target, return_tensors="pt").input_ids.to(self.device)
            out = self.model(**enc, labels=labels)
            return -float(out.loss.item())

        score_yes, score_no = score_candidate("YES"), score_candidate("NO")
        e_yes, e_no = math.exp(score_yes), math.exp(score_no)
        p_yes = e_yes / (e_yes + e_no + 1e-12)
        p_no = e_no / (e_yes + e_no + 1e-12)

        verdict = "YES" if p_yes >= p_no else "NO"
        confidence = max(p_yes, p_no)
        return {"diagnosis": self.diagnosis, "verdict": verdict, "confidence": confidence}

    def run(self, sample_size: int = 10) -> list:
        """
        Run diagnosis classification for a sample of patients.

        Args:
            sample_size (int): Number of patients to sample. Defaults to 10.

        Returns:
            list: List of dictionaries, each containing:
                - "patient_id" (int): Patient identifier.
                - "diagnosis" (str): Diagnosis tested.
                - "verdict" (str): "YES" or "NO".
                - "confidence" (float): Confidence score.

        Example:
            >>> task = MIMIC3LLMDiagnosisTask("NOTEEVENTS.csv", "heart failure")
            >>> results = task.run(sample_size=5)
            >>> print(results[0])
            {'patient_id': 123, 'diagnosis': 'heart failure', 'verdict': 'YES', 'confidence': 0.87}
        """
        sample_df = self.patients_df.sample(min(sample_size, len(self.patients_df)))
        results = []
        for _, row in sample_df.iterrows():
            res = self.classify_patient(row["PATIENT_NOTES"])
            res["patient_id"] = int(row["SUBJECT_ID"])
            results.append(res)
        return results
