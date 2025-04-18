
# ------------------------------------------------------------------------------
# Contribution Header
#
# Name: 
#   Zilal Eiz Al Din
#
# NetID: 
#   zelalae2@illinois.edu
#
# Paper Title: 
#   Hurtful Words: Quantifying Biases in Clinical Contextual Word Embeddings
#
# Paper Link:
#   https://arxiv.org/pdf/2003.11515
#
#   This contribution implements the HurtfulWordsDataset, a new dataset class 
#   within the PyHealth framework. The HurtfulWordsDataset is designed to 
#   preprocess clinical notes from the MIMIC-III database by applying 
#   de-identification masking (PHI replacement) and gendered term masking, 
#   to support masked language model evaluation tasks.
#
#   The dataset class reads from NOTEEVENTS tables, applies cleaning and 
#   masking logic aligned with the "Hurtful Words" paper, and prepares input 
#   samples compatible with bias evaluation tasks that require masked templates.
#
#   Main preprocessing steps include:
#   - Replacing protected health information (PHI) fields with placeholder tokens.
#   - Masking gendered words such as "he", "she", "male", "female", etc.
#   - Cleaning formatting artifacts (line breaks, numbering, underscores).
#
#   This implementation enables reproducibility of masked bias evaluation 
#   experiments on ClinicalBERT and related models, directly following the 
#   methodology described in the "Hurtful Words" paper.
# ------------------------------------------------------------------------------

import os
import re
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
from nltk.tokenize import sent_tokenize
from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)


class HurtfulWordsDataset(BaseDataset):
    """Dataset class for masked language modeling on de-identified clinical notes.

    This dataset parses NOTEEVENTS from MIMIC-III and applies preprocessing
    including PHI masking and gendered term masking.

    Attributes:
        root: Directory containing the MIMIC-III CSV files.
        tables: List of tables to load, must include "noteevents".
        dataset_name: Name of the dataset.
        config_path: Path to YAML config file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initializes HurtfulWordsDataset.

        Args:
            root: Directory containing CSV files.
            tables: List of tables to load (must include "noteevents").
            dataset_name: Optional name of the dataset (default: "hurtfulwords").
            config_path: Optional path to YAML config (default: mimic3.yaml).
        """
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "mimic3.yaml"

        assert "noteevents" in tables, "Must include 'noteevents' in tables."

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "hurtfulwords",
            config_path=config_path,
        )

    @property
    def collected_global_event_df(self) -> pl.DataFrame:
        """Returns preprocessed event DataFrame with masked text.

        Filtering and masking are applied:
        - Only specific note categories are kept.
        - PHI patterns are replaced.
        - Gendered terms are masked.

        Returns:
            A Polars DataFrame containing patient_id, event_type, timestamp,
            and a masked note text.

        Example:
            dataset.collected_global_event_df.select(["patient_id", "noteevents/masked_text"])
        """
        df = super().collected_global_event_df

        allowed_categories = [
            "Nursing", "Nursing/other", "Physician", "Discharge summary"
        ]
        df = df.filter(pl.col("noteevents/category").is_in(allowed_categories))

        if "noteevents/text" not in df.columns:
            raise ValueError("Missing expected column: noteevents/text")

        df = df.with_columns([
            pl.col("noteevents/text").map_elements(
                self._process_note,
                return_dtype=pl.Utf8
            ).alias("noteevents/masked_text")
        ])

        return df

    def _process_note(self, note: str) -> str:
        """Applies full preprocessing to a clinical note.

        This includes PHI masking, gendered word masking, and formatting cleanups.

        Args:
            note: The raw clinical note as a string.

        Returns:
            A cleaned and masked version of the note as a string.

        Example:
            masked_note = self._process_note("Patient is a 45-year-old male admitted...")
        """
        note = self._replace_phi(note)
        note = self._mask_gendered_terms(note)
        note = self._clean_note_format(note)
        return note

    def _replace_phi(self, text: str) -> str:
        """Replaces de-identified PHI tags (e.g., **NAME**) with placeholders.

        Args:
            text: Input text string containing PHI tags.

        Returns:
            String with PHI tags replaced with generic tokens like PHINAMEPHI.

        Example:
            result = self._replace_phi("Patient [**Name**] admitted to [**Hospital**]")
        """
        return re.sub(r'\[\*\*(.*?)\*\*\]', self._repl_phi, text)

    def _repl_phi(self, match: re.Match) -> str:
        """Replacement logic for PHI fields based on keyword detection.

        Args:
            match: A regex match object from _replace_phi().

        Returns:
            Replacement string based on PHI type (e.g., PHINAMEPHI).
        """
        label = match.group(1).lower().strip()

        if self._is_date(label) or "holiday" in label:
            return "PHIDATEPHI"
        elif "hospital" in label:
            return "PHIHOSPITALPHI"
        elif any(word in label for word in ["location", "university", "state", "country"]):
            return "PHILOCATIONPHI"
        elif any(word in label for word in ["name", "attending", "dictator", "contact"]):
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
            s: The input string.

        Returns:
            Boolean indicating whether the string resembles a date.

        Example:
            True if input is "2003-04-25" or contains "month", "year".
        """
        return bool(re.search(r"\d{4}-\d{1,2}-\d{1,2}", s) or "month" in s or "year" in s)

    def _mask_gendered_terms(self, text: str) -> str:
        """Masks gender-related words with [MASK].

        Args:
            text: Clinical note text after PHI replacement.

        Returns:
            Text with gendered words replaced.

        Example:
            "The patient is a male" -> "The patient is a [MASK]"
        """
        return re.sub(
            r"\b(male|gentleman|man|he|female|woman|she|his|her|him)\b",
            "[MASK]",
            text,
            flags=re.IGNORECASE
        )

    def _clean_note_format(self, note: str) -> str:
        """Final formatting cleanup on a clinical note.

        Args:
            note: Partially processed clinical note.

        Returns:
            Cleaned note text.

        Example:
            Removes line breaks, list numbers, underscores, etc.
        """
        note = re.sub(r'\n', ' ', note)
        note = re.sub(r'[0-9]+\.', '', note)
        note = re.sub(r'[-_=]{2,}', '', note)
        note = re.sub(r'\bdr\.', 'doctor', note, flags=re.IGNORECASE)
        note = re.sub(r'\bm\.d\.', 'md', note, flags=re.IGNORECASE)
        return note.strip()
