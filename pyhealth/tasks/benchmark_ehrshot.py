from typing import Any, Dict, List, Optional

import polars as pl

from .base_task import BaseTask


class BenchmarkEHRShot(BaseTask):
    """Benchmark predictive tasks using EHRShot."""

    tasks = {
        "operational_outcomes": [
            "guo_los",
            "guo_readmission",
            "guo_icu"
        ],
        "lab_values": [
            "lab_thrombocytopenia",
            "lab_hyperkalemia",
            "lab_hypoglycemia",
            "lab_hyponatremia",
            "lab_anemia"
        ],
        "new_diagnoses": [
            "new_hypertension",
            "new_hyperlipidemia",
            "new_pancan",
            "new_celiac",
            "new_lupus",
            "new_acutemi"
        ],
        "chexpert": [
            "chexpert"
        ]
    }

    def __init__(self, task: str, omop_tables: Optional[List[str]] = None) -> None:
        """Initialize the BenchmarkEHRShot task.

        Args:
            task (str): The specific task to benchmark.
            omop_tables (Optional[List[str]]): List of OMOP tables to filter input events.
        """
        self.task = task
        self.omop_tables = omop_tables
        self.task_name = f"BenchmarkEHRShot/{task}"
        self.input_schema = {"feature": "sequence"}
        if task in self.tasks["operational_outcomes"]:
            self.output_schema = {"label": "binary"}
        elif task in self.tasks["lab_values"]:
            self.output_schema = {"label": "multiclass"}
        elif task in self.tasks["new_diagnoses"]:
            self.output_schema = {"label": "binary"}
        elif task in self.tasks["chexpert"]:
            self.output_schema = {"label": "multilabel"}
  
    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        if self.omop_tables is None:
            return df
        filtered_df = df.filter(
            (pl.col("event_type") != "ehrshot") |
            (pl.col("ehrshot/omop_table").is_in(self.omop_tables))
        )
        return filtered_df

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples = []
        split = patient.get_events("splits")
        assert len(split) == 1, "Only one split is allowed"
        split = split[0].split
        labels = patient.get_events(self.task)
        for label in labels:
            # Returning a dataframe of events is much faster than a list of events
            events_df = patient.get_events(
                "ehrshot", end=label.timestamp, return_df=True
            )
            codes = events_df["ehrshot/code"].to_list()
            label_value = label.value
            if self.task == "chexpert":
                # Convert {0,1,...,8192} aka binary string to a list of positive label indices
                label_value = int(label_value)
                label_value = [i for i in range(14) if (label_value >> i) & 1]
                label_value = [13 - i for i in label_value[::-1]]
            samples.append({
                "feature": codes,
                "label": label_value,
                "split": split
            })
        return samples
