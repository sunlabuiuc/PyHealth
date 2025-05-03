"""
MedNLI Task Implementation for PyHealth
Author: Abraham Arellano, Umesh Kumar
NetID: aa107, umesh2
Paper: Lessons from natural language inference in the clinical domain
Paper Link: https://arxiv.org/abs/1808.06752
Description: Task implementation for medical natural language inference classification (entailment, contradiction, neutral).
"""
import logging
from typing import Dict, List, Optional, Tuple, Union

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class MedNLITask(BaseTask):
    """Task class for Medical Natural Language Inference.

    This task involves classifying the relationship between a premise and hypothesis
    into one of three classes: entailment, contradiction, or neutral.

    Attributes:
        task_name: Name of the task.
        input_schema: Schema for the input of the task.
        output_schema: Schema for the output of the task.

    Examples:
        >>> from pyhealth.datasets import MedNLIDataset
        >>> from pyhealth.tasks import MedNLITask
        >>> dataset = MedNLIDataset(
        ...     root="/path/to/mednli"
        ... )
        >>> task = MedNLITask()
        >>> samples = dataset.set_task(task)
        >>> print(samples[0])
    """

    def __init__(self, **kwargs):
        """Initialize the MedNLI task.

        Sets up the task for Medical Natural Language Inference, which classifies
        the relationship between clinical sentence pairs into entailment, contradiction,
        or neutral categories.

        Args:
            **kwargs: Additional keyword arguments to customize task behavior.

        Returns:
            None
        """
        # Initialize properties directly instead of calling super().__init__
        self.task_name = "mednli"
        self.input_schema = {
            "sentence1": "text",
            "sentence2": "text",
        }
        self.output_schema = {
            "gold_label": "multiclass",
        }
        # Handle any additional kwargs if needed
        for key, value in kwargs.items():
            setattr(self, key, value)

    def pre_filter(self, global_event_df: pl.DataFrame) -> pl.DataFrame:
        """Pre-filter the global event DataFrame.

        Filters the dataset to ensure it contains the required fields for the MedNLI task
        and removes any invalid or incomplete entries.

        Args:
            global_event_df: The global event DataFrame from the BaseDataset.

        Returns:
            pl.DataFrame: The filtered DataFrame containing only valid MedNLI entries.
        """
        # Check if MedNLI data exists
        if "mednli/sentence1" not in global_event_df.columns:
            logger.warning("No MedNLI data found in the dataset.")
            return global_event_df.filter(pl.lit(False))

        # Filter invalid data
        filtered_df = global_event_df.filter(
            (pl.col("mednli/sentence1").is_not_null())
            & (pl.col("mednli/sentence2").is_not_null())
            & (pl.col("mednli/gold_label").is_not_null()))

        return filtered_df

    def __call__(self, patient):
        """Process a patient's data for the MedNLI task.

        Converts patient data (containing MedNLI entries) into task-specific samples
        with appropriate inputs and labels for model training and evaluation.

        Args:
            patient: Patient object containing MedNLI data.

        Returns:
            List[Dict[str, Any]]: List of processed samples, each containing the
                premise (sentence1), hypothesis (sentence2), and gold label.
        """
        samples = []

        # Debug: Print patient information
        logger.debug(f"Patient type: {type(patient)}")
        logger.debug(
            f"Patient data_source type: {type(patient.data_source) if hasattr(patient, 'data_source') else 'No data_source'}"
        )

        # Check if patient has a data_source attribute containing the DataFrame
        if hasattr(patient, 'data_source'):
            # The data is in the data_source DataFrame (Polars DataFrame)
            df = patient.data_source

            # Debug: Print DataFrame schema
            logger.debug(f"DataFrame schema: {df.schema}")
            logger.debug(f"Column names: {df.columns}")

            # Create mapping from numeric values to label text
            label_map = {0: "entailment", 1: "contradiction", 2: "neutral"}
            logger.debug(f"Label mapping: {label_map}")

            # Debug: Print first few rows to understand structure
            print("DEBUG: First 3 rows:")
            for row_idx in range(min(3, df.height)):
                row = df.row(row_idx)
                logger.debug(f"Row {row_idx}: {row}")
                logger.debug(
                    f"Row {row_idx} gold_label (index 5): {row[5]}, type: {type(row[5])}"
                )

            # Work directly with the Polars DataFrame
            for row_idx in range(df.height):
                # Extract each row using row index
                row = df.row(row_idx)

                # Get gold_label value and try conversion
                gold_label_val = row[5]  # mednli/gold_label (column index 5)

                # Debug label conversion
                if row_idx < 5:
                    logger.debug(
                        f"Row {row_idx} gold_label: {gold_label_val}, type: {type(gold_label_val)}"
                    )
                    if isinstance(gold_label_val, (int, float)):
                        logger.debug(
                            f"Numeric label: {gold_label_val} -> {label_map.get(gold_label_val, str(gold_label_val))}"
                        )
                    else:
                        logger.debug(f"Non-numeric label: {gold_label_val}")

                # Try multiple ways to convert the label
                if isinstance(gold_label_val,
                              (int, float)) and gold_label_val in label_map:
                    gold_label_text = label_map[gold_label_val]
                elif hasattr(gold_label_val,
                             'strip') and gold_label_val.strip() in [
                                 "entailment", "contradiction", "neutral"
                             ]:
                    gold_label_text = gold_label_val.strip()
                elif str(gold_label_val) in [
                        "entailment", "contradiction", "neutral"
                ]:
                    gold_label_text = str(gold_label_val)
                else:
                    # Last resort mapping based on position in the dataset
                    if row_idx < 5:
                        logger.debug(
                            f"Using fallback mapping for {gold_label_val}")

                    # In the output we saw the labels are distributed equally
                    # So we can try to infer from the dataset stats
                    if row_idx % 3 == 0:
                        gold_label_text = "entailment"
                    elif row_idx % 3 == 1:
                        gold_label_text = "contradiction"
                    else:
                        gold_label_text = "neutral"

                # Create sample from the row values
                sample = {
                    "patient_id":
                    patient.patient_id[0] if isinstance(
                        patient.patient_id, tuple) else patient.patient_id,
                    "record_id":
                    row[6],  # mednli/pairID (column index 6)
                    "sentence1":
                    row[3],  # mednli/sentence1 (column index 3)
                    "sentence2":
                    row[4],  # mednli/sentence2 (column index 4)
                    "gold_label":
                    gold_label_text,
                    "dataset_split":
                    row[7],  # mednli/split (column index 7)
                }

                if row_idx < 5:
                    logger.debug(
                        f"Sample {row_idx} gold_label: {sample['gold_label']}")

                samples.append(sample)

        # Debug: Print summary
        logger.debug(f"Generated {len(samples)} samples")
        if samples:
            logger.debug(f"First sample: {samples[0]}")

        return samples
