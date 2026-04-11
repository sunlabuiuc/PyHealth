import logging
from pathlib import Path
from typing import List, Optional

import narwhals as pl

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class OMOPDataset(BaseDataset):
    """
    A dataset class for handling OMOP CDM (Common Data Model) data.

    The Observational Medical Outcomes Partnership (OMOP) Common Data
    Model (CDM) is an open community data standard, designed to
    standardize the structure and content of observational data.

    OMOP CDM provides a standardized way to represent observational health
    data, enabling consistent data analysis across different healthcare
    systems. The CDM includes standardized vocabularies and data structures
    for clinical events such as visits, conditions, procedures, drug
    exposures, measurements, and more.

    See: https://www.ohdsi.org/data-standardization/

    The default tables loaded are:
        - person: demographics and basic patient information
        - visit_occurrence: hospital/clinic visits
        - death: mortality information

    Additional tables can be specified via the `tables` parameter:
        - condition_occurrence: diagnoses (ICD codes)
        - procedure_occurrence: procedures (CPT, ICD codes)
        - drug_exposure: medication orders and administrations
        - measurement: laboratory tests and vital signs
        - observation: clinical observations
        - device_exposure: medical device usage

    The person_id field is used as the patient identifier across all tables.
    All clinical events are linked to visits via visit_occurrence_id.

    Args:
        root (str): The root directory where the OMOP dataset CSV files
            are stored.
        tables (List[str]): A list of additional tables to include beyond
            the default tables (person, visit_occurrence, death).
        dataset_name (Optional[str]): The name of the dataset. Defaults
            to "omop".
        config_path (Optional[str]): The path to the YAML configuration
            file defining table schemas. If not provided, uses the default
            OMOP config.
        dev (bool): Whether to enable dev mode (only use first 1000
            patients for faster testing). Default is False.
        refresh_cache (bool): Whether to refresh the cache; if true, the
            dataset will be processed from scratch. Default is False.

    Attributes:
        task: Optional[str], name of the task (e.g., "mortality
            prediction"). Default is None.
        samples: Optional[List[Dict]], a list of samples, each sample is
            a dict with patient_id, visit_id, and other task-specific
            attributes as key. Default is None.
        patient_to_index: Optional[Dict[str, List[int]]], a dict mapping
            patient_id to a list of sample indices. Default is None.
        visit_to_index: Optional[Dict[str, List[int]]], a dict mapping
            visit_id to a list of sample indices. Default is None.

    Examples:
        >>> from pyhealth.datasets import OMOPDataset
        >>> from pyhealth.tasks import MortalityPredictionOMOP
        >>>
        >>> # Load OMOP dataset with clinical tables
        >>> dataset = OMOPDataset(
        ...     root="/path/to/omop/data",
        ...     tables=["condition_occurrence", "procedure_occurrence",
        ...             "drug_exposure"],
        ...     dev=False,
        ... )
        >>> dataset.stat()
        >>> dataset.info()
        >>>
        >>> # Access patient data
        >>> patient = dataset.get_patient("123")
        >>> print(f"Patient has {len(patient.data_source)} events")
        >>>
        >>> # Get events by type
        >>> visits = patient.get_events(event_type="visit_occurrence")
        >>> conditions = patient.get_events(
        ...     event_type="condition_occurrence"
        ... )
        >>>
        >>> # Filter events by visit
        >>> visit = visits[0]
        >>> visit_conditions = patient.get_events(
        ...     event_type="condition_occurrence",
        ...     filters=[("visit_occurrence_id", "==",
        ...               visit.visit_occurrence_id)]
        ... )
        >>>
        >>> # Create task-specific samples
        >>> mortality_task = MortalityPredictionOMOP()
        >>> sample_dataset = dataset.set_task(task=mortality_task)
        >>> print(f"Generated {len(sample_dataset)} samples")
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Initializes the OMOPDataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset.
                Defaults to "omop".
            config_path (Optional[str]): The path to the configuration
                file. If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default OMOP config")
            config_path = Path(__file__).parent / "configs" / "omop.yaml"
        default_tables = ["person", "visit_occurrence", "death"]
        tables = default_tables + tables
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "omop",
            config_path=config_path,
            **kwargs,
        )
        return

    def preprocess_person(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Preprocesses the person table by constructing birth_datetime.

        Concatenates year_of_birth, month_of_birth, and day_of_birth
        into a single birth_datetime field. Missing month/day default
        to 01.

        Args:
            df (pl.LazyFrame): The input dataframe containing person data.

        Returns:
            pl.LazyFrame: The processed dataframe with birth_datetime.
        """
        df = df.with_columns(
            [
                (
                    pl.col("year_of_birth").cast(pl.String)
                    + "-"
                    + (
                        pl.when(pl.col("month_of_birth").is_null())
                        .then(pl.lit("01"))
                        .otherwise(pl.col("month_of_birth").cast(pl.String).str.zfill(2))
                    ).cast(pl.String)
                    + "-"
                    + (
                        pl.when(pl.col("day_of_birth").is_null())
                        .then(pl.lit("01"))
                        .otherwise(pl.col("day_of_birth").cast(pl.String).str.zfill(2))
                    ).cast(pl.String)
                    + " 00:00:00"
                ).alias("birth_datetime")
            ]
        )
        return df
