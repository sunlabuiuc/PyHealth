# ------------------------------------------------------------------------------
# Contribution Header
#
# Name:
#   Zilal Eiz Al Din && Payel Chakraborty
#
# NetID:
#   zelalae2 && payelc2
#
# Paper Title:
#   Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings
#
# Paper Link:
#   https://arxiv.org/pdf/2003.11515
#
# This contribution implements the HurtfulWordsPreprocessingTask, a task class
# within the PyHealth framework. The task processes clinical notes from the
# MIMIC-III database by applying de-identification masking (PHI replacement)
# and gendered term masking, supporting masked language model evaluation tasks.
# ------------------------------------------------------------------------------

import re
import logging
from dataclasses import dataclass, field
from typing import Dict, List

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


@dataclass
class HurtfulWordsPreprocessingTask(BaseTask):
    """Preprocessing task to apply PHI and gender masking to MIMIC-III notes.

    This task implements preprocessing steps from the Hurtful Words paper. It:
    - Replaces PHI tokens with placeholders (e.g., PHIDATEPHI, PHINAMEPHI)
    - Masks gendered terms with [MASK]
    - Cleans formatting issues

    Attributes:
        task_name (str): Name of the task.
        input_schema (dict): Schema of expected input (must include "text").
        output_schema (dict): Schema of output (produces "masked_text").
    """

    task_name: str = "hurtful_words_preprocessing"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"text": "text"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {
        "masked_text": "text"
    })

    def __call__(self, patient: Patient) -> List[Dict]:
        """Processes clinical notes from a single patient.

        Args:
            patient (Patient): A patient object from the PyHealth dataset.

        Returns:
            List[Dict]: A list of sample dictionaries. Each contains:
                - "text": the original note text
                - "masked_text": the note with PHI/gender terms masked
                - "timestamp": note timestamp
                - "patient_id": patient ID
        Example:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import HurtfulWordsPreprocessingTask
        >>> PATH_TO_CSVs='/content/drive/MyDrive/Project/HurtfulWords/Data/physionet.org/files/mimiciii/1.4'

        >>> # STEP 1: load data
        >>> mimic3_base = MIMIC3Dataset(root=f"{PATH_TO_CSVs}", tables=["noteevents"])

        >>> # STEP 2: set task
        >>> task = HurtfulWordsPreprocessingTask()
        >>> sample_dataset = mimic3_base.set_task(task)

        """
        samples = []
        for event in patient.get_events(event_type="noteevents"):
            text = event.attr_dict.get("text", "")
            if not text:
                continue
            masked_text = self._process_note(text)
            samples.append({
                "patient_id": patient.patient_id,
                "text": text,
                "masked_text": masked_text,
                "timestamp": event.timestamp,
            })
        return samples

    def _process_note(self, note: str) -> str:
        """Applies PHI replacement, gender masking, and formatting cleanup.

        Args:
            note (str): Raw clinical note.

        Returns:
            str: The processed note.
        """
        note = self._replace_phi(note)
        note = self._mask_gendered_terms(note)
        note = self._clean_note_format(note)
        return note

    def _replace_phi(self, text: str) -> str:
        """Replaces [** ... **] tags with PHI placeholders.

        Args:
            text (str): Clinical note text.

        Returns:
            str: Text with PHI replaced by placeholder tokens.
        """
        return re.sub(r"\[\*\*(.*?)\*\*\]", self._repl_phi, text)

    def _repl_phi(self, match: re.Match) -> str:
        """Returns a PHI placeholder string based on matched label.

        Args:
            match (re.Match): Regex match object for a PHI tag.

        Returns:
            str: Replacement placeholder token.
        """
        label = match.group(1).lower().strip()
        if self._is_date(label) or "holiday" in label:
            return "PHIDATEPHI"
        elif "hospital" in label:
            return "PHIHOSPITALPHI"
        elif any(word in label for word in [
            "location", "university", "state", "country"
        ]):
            return "PHILOCATIONPHI"
        elif any(word in label for word in [
            "name", "attending", "dictator", "contact"
        ]):
            return "PHINAMEPHI"
        elif "telephone" in label or "number" in label:
            return "PHINUMBERPHI"
        elif "age over 90" in label:
            return "PHIAGEPHI"
        else:
            return "PHIOTHERPHI"

    def _is_date(self, s: str) -> bool:
        """Checks if a string looks like a date.

        Args:
            s (str): The string to check.

        Returns:
            bool: True if the string resembles a date.
        """
        return bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}", s) or
                    "month" in s or "year" in s)

    def _mask_gendered_terms(self, text: str) -> str:
        """Replaces gendered terms with [MASK].

        Args:
            text (str): The input clinical note.

        Returns:
            str: Text with gendered words masked.
        """
        return re.sub(
            r"\b(M|F|male|gentleman|man|he|female|woman|she|his|her|him)\b",
            "[MASK]",
            text,
            flags=re.IGNORECASE
        )

    def _clean_note_format(self, note: str) -> str:
        """Cleans formatting artifacts from the note.

        Args:
            note (str): The raw note.

        Returns:
            str: Cleaned note text.
        """
        note = re.sub(r"\n", " ", note)
        note = re.sub(r"[0-9]+\.", "", note)
        note = re.sub(r"[-_=]{2,}", "", note)
        note = re.sub(r"\bdr\.", "doctor", note, flags=re.IGNORECASE)
        note = re.sub(r"\bm\.d\.", "md", note, flags=re.IGNORECASE)
        return note.strip()


if __name__ == "__main__":
    task = HurtfulWordsPreprocessingTask()
    print(task)
    print(type(task))



