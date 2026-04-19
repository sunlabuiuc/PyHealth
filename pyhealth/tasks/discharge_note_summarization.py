"""
PyHealth task for generating high quality patient summaries with LLMs while using the MIMIC4-Note dataset.

Dataset link:
    https://physionet.org/content/ann-pt-summ/1.0.1/

Dataset paper: (please cite if you use this dataset)
    Hegselmann, S., Shen, S.Z., Gierse, F., Agrawal, M., Sontag, D. and Jiang, X., 2024. 
    A data-centric approach to generate faithful and high quality patient summaries with large language models. 
    arXiv preprint arXiv:2402.15422.

Dataset paper link:
    https://arxiv.org/abs/2402.15422

Author:
    Vishal Vyas (vyas9@illinois.edu)
"""

import os
import sys
import re
import random
import string
import pickle
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from collections import Counter

import pandas as pd
import nltk
import swifter
from tqdm import tqdm

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask
from pyhealth.processors import TextProcessor

import logging


pd.options.mode.chained_assignment = None

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

#NOTE_ROOT = "./data/llm_data/"          # Path to MIMIC-4 notes directory
NOTE_ROOT = "/Users/vishalvyas/uiuc_cs_598/mimic-iv-note"
OUTPUT_DIR = "./outputdata/"            # Path for output CSV
OUTPUT_FILE = "mimic_processed_summaries.csv"

MAX_DOUBLE_NEWLINES = 5                 # Max allowed double newlines in a summary
NUM_WORDS_PER_DEIDENTIFIED = 10        # Allowed ___ density threshold
MIN_SUMMARY_LENGTH = 350               # Minimum character length for summaries
MIN_SENTENCES = 3                       # Minimum sentence count per summary


# ── PyHealth Task Definition ───────────────────────────────────────────────────

class DischargeNoteSummarization(BaseTask):

    """
    A PyHealth task class for generating faithful and high quality patient summaries with Large Language Models .

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The schema for the task input.
        output_schema (Dict[str, str]): The schema for the task output.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks import MIMIC4Dataset
        >>> dataset = MIMIC4Dataset(note_root=NOTE_ROOT,note_tables=["discharge"])
        >>> task = DataforLlmSummaries()
        >>> samples = dataset.set_task(task)
    """
     
    task_name: str = "DischargeNoteSummarization"
    input_schema: Dict[str, str] = {
        "subject_id": "text",
        "hadm_id": "text",
        "text": "text"
    }

    output_schema: Dict[str, str] = {
        "brief_hospital_course": "text",
        "summary": "text"
    }

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """
        Generates patient brief_hospital_course and summary samples for a single patient.

        Args:
            patient (Patient): A patient object containing at least one 'discharge' event.

        Returns:
            List[Dict]: A list containing a dictionary for each patient visit with:
                - "text": patient clinical notes text,
                - "brief_hospital_course": patient brief hospital course,
                - "summary": patient discharge summary text,
                - "subject_id": patient identifier,
                - "hadm_id": Hospital Admission Identifier,
                
        """
        samples = []
        subject_id = patient.patient_id

        for dis in patient.get_events("discharge"):
            textNote = dis.attr_dict["text"]
            hadm_id = dis.attr_dict["hadm_id"]

            # Extract Brief Hospital Course , remove new lines and remove whitespaces to create single paragraph
            start = textNote.find("Brief Hospital Course:")
            if start < 0:
                continue
            end = textNote.find("Medications on Admission:")
            if end == -1:
                end = textNote.find("Discharge Medications:")
            if end == -1:
                end = textNote.find("Discharge Disposition:")
            if end == 0 or start >= end:
                continue
            brief_hospital_course = textNote[start:end].replace("\n", " ")
            brief_hospital_course = " ".join(brief_hospital_course.split())

            # Extract Discharge Instructions (summary) and filter out samples less than MIN_SUMMARY_LENGTH
            start = textNote.find("Discharge Instructions:")
            end = textNote.find("Followup Instructions:")
            if start >= 0 and end >= 0:
                summary = textNote[start:end].replace("\n", " ")
                summary = " ".join(summary.split())
                
                summary = summary.strip()
                #Only add to samples if length of summary greater than specified MIN_SUMMARY_LENGTH
                if len(summary) >= MIN_SUMMARY_LENGTH:
                    samples.append({
                        "text": textNote,
                        "brief_hospital_course": brief_hospital_course,
                        "summary": summary,
                        "subject_id": subject_id,
                        "hadm_id": hadm_id,
                    })

        return samples