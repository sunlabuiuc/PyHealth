import logging
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class eICUDataset(BaseDataset):
    """
    A dataset class for handling eICU data.

    The eICU dataset is a large dataset of de-identified health records of ICU
    patients. The dataset is available at https://eicu-crd.mit.edu/.

    The basic information is stored in the following tables:
        - patient: defines a patient (uniquepid), a hospital admission
            (patienthealthsystemstayid), and an ICU stay (patientunitstayid)
            in the database.
        - hospital: contains information about a hospital (e.g., region).

    Note that in eICU, a patient can have multiple hospital admissions and each
    hospital admission can have multiple ICU stays. The data in eICU is centered
    around the ICU stay and all timestamps are relative to the ICU admission time.

    We further support the following tables:
        - diagnosis: contains ICD diagnoses (ICD9CM and ICD10CM code)
            and diagnosis information for patients
        - treatment: contains treatment information for patients.
        - medication: contains medication related order entries for patients.
        - lab: contains laboratory measurements for patients
        - physicalexam: contains all physical exams conducted for patients.
        - admissiondx: table contains the primary diagnosis for admission to
            the ICU per the APACHE scoring criteria.

    Attributes:
        root (str): The root directory where the dataset is stored.
        tables (List[str]): A list of tables to be included in the dataset.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> dataset = eICUDataset(
        ...     root="/path/to/eicu-crd/2.0",
        ...     tables=["diagnosis", "medication", "treatment"],
        ... )
        >>> dataset.stats()
        >>> patient = dataset.get_patient("patient_id")
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the eICUDataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            tables (List[str]): A list of additional tables to include.
            dataset_name (Optional[str]): The name of the dataset. Defaults to "eicu".
            config_path (Optional[str]): The path to the configuration file. 
                If not provided, a default config is used.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "eicu.yaml"
        
        # Default table is patient which contains basic patient/stay info
        default_tables = ["patient"]
        tables = default_tables + tables
        
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "eicu",
            config_path=config_path,
            **kwargs
        )
        return
