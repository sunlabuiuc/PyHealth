"""SEER (Surveillance, Epidemiology, and End Results) dataset for PyHealth.

The SEER program of the National Cancer Institute (NCI) collects cancer
incidence and survival data from population-based cancer registries covering
approximately 48% of the US population.

Data access
-----------
SEER data requires a free research data agreement:
  https://seer.cancer.gov/data/access.html

Once approved, download a cohort as a CSV export from SEER*Stat.

The pre-processed file ``seer_clinical.csv`` expected by this class should
have the columns listed in ``configs/seer.yaml``.
:func:`SEERDataset.prepare_metadata` converts a standard SEER*Stat CSV export
to this format when the processed file is absent.

Common SEER*Stat export columns (variable labels vary by release; see the
SEER*Stat dictionary for your download):
  - ``Patient ID``
  - ``Age recode with single ages and 85+``  (or ``Age at Diagnosis``)
  - ``Sex``
  - ``Race recode (W, B, AI, API)``
  - ``Primary Site``
  - ``Histologic Type ICD-O-3``
  - ``Derived AJCC Stage Group, 7th ed (2010-2015)``
  - ``Grade``
  - ``CS tumor size (2004-2015)``
  - ``Regional nodes examined (1988+)``
  - ``Regional nodes positive (1988+)``
  - ``Survival months``
  - ``Vital status recode (study cutoff used)``  (Alive / Dead)
  - ``Year of diagnosis``

Citation:
  National Cancer Institute, DCCPS, Surveillance Research Program,
  SEER*Stat software (www.seer.cancer.gov/seerstat).
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class SEERDataset(BaseDataset):
    """SEER cancer incidence / survival dataset.

    Each row represents a single tumour record.  Patients with multiple
    primaries will have multiple rows (distinguished by ``SEQUENCE_NUMBER``).
    The dataset is loaded with ``PATIENT_ID`` as the patient identifier.

    Args:
        root: Directory containing ``seer_clinical.csv`` (or a raw SEER*Stat
            export named ``seer_raw.csv`` / ``seer.csv``).
        tables: Additional tables beyond the default ``["seer"]``.
        dataset_name: Optional name; defaults to ``"seer"``.
        config_path: Optional YAML path; defaults to the bundled
            ``configs/seer.yaml``.
        **kwargs: Passed through to :class:`~pyhealth.datasets.BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import SEERDataset
        >>> dataset = SEERDataset(root="/path/to/seer")
        >>> dataset.stats()
        >>> samples = dataset.set_task(task)
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = Path(__file__).parent / "configs" / "seer.yaml"

        processed_csv = os.path.join(root, "seer_clinical.csv")
        if not os.path.exists(processed_csv):
            logger.info(
                "seer_clinical.csv not found — attempting to prepare from raw data."
            )
            self.prepare_metadata(root)

        default_tables = ["seer"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "seer",
            config_path=str(config_path),
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Convert a raw SEER*Stat CSV export to the standardised format.

        Looks for ``seer_raw.csv``, ``seer.csv``, or ``*.csv`` (first match)
        inside ``root`` and writes ``seer_clinical.csv``.

        The mapping below covers the most common SEER*Stat export variable
        names.  If your export uses different labels, rename the columns in
        the output CSV or pass a custom ``config_path`` to :class:`SEERDataset`.

        Args:
            root: Directory to search and write output.
        """
        raw_candidates = ["seer_raw.csv", "seer.csv"]
        raw_file: Optional[str] = None
        for fname in raw_candidates:
            candidate = os.path.join(root, fname)
            if os.path.exists(candidate):
                raw_file = candidate
                break

        # Fall back: any CSV in root
        if raw_file is None:
            for fname in os.listdir(root):
                if fname.endswith(".csv") and fname != "seer_clinical.csv":
                    raw_file = os.path.join(root, fname)
                    logger.info(f"Using fallback raw file: {raw_file}")
                    break

        output_path = os.path.join(root, "seer_clinical.csv")

        if raw_file is None:
            logger.warning(
                f"No raw SEER file found in {root}. "
                "Please export a cohort from SEER*Stat as a CSV and save it "
                "as 'seer_raw.csv' in the root directory.  "
                "See https://seer.cancer.gov/seerstat for instructions."
            )
            pd.DataFrame(
                columns=[
                    "PATIENT_ID",
                    "AGE_AT_DIAGNOSIS",
                    "SEX",
                    "RACE",
                    "PRIMARY_SITE",
                    "HISTOLOGY",
                    "STAGE",
                    "GRADE",
                    "TUMOR_SIZE_MM",
                    "REGIONAL_NODES_EXAMINED",
                    "REGIONAL_NODES_POSITIVE",
                    "SURVIVAL_MONTHS",
                    "VITAL_STATUS",
                    "YEAR_OF_DIAGNOSIS",
                    "SEQUENCE_NUMBER",
                    "LATERALITY",
                    "SURGERY",
                    "RADIATION",
                    "CHEMOTHERAPY",
                ]
            ).to_csv(output_path, index=False)
            return

        logger.info(f"Processing SEER raw file: {raw_file}")
        df = pd.read_csv(raw_file, low_memory=False)

        # --- column name normalisation ---
        # SEER*Stat uses verbose labels; map common variants to short names.
        rename: dict = {}
        for col in df.columns:
            col_upper = col.upper().strip()
            if "PATIENT ID" in col_upper or col_upper == "PATIENT_ID":
                rename[col] = "PATIENT_ID"
            elif "AGE" in col_upper and "DIAGNOSIS" in col_upper:
                rename[col] = "AGE_AT_DIAGNOSIS"
            elif col_upper in ("SEX", "GENDER"):
                rename[col] = "SEX"
            elif "RACE" in col_upper and "RECODE" not in col_upper:
                rename[col] = "RACE"
            elif "RACE RECODE" in col_upper:
                rename[col] = "RACE"
            elif "PRIMARY SITE" in col_upper or "PRIMARY_SITE" in col_upper:
                rename[col] = "PRIMARY_SITE"
            elif "HISTOLOGIC TYPE" in col_upper or "HISTOLOGY" in col_upper:
                rename[col] = "HISTOLOGY"
            elif "STAGE" in col_upper and "AJCC" not in col_upper:
                rename[col] = "STAGE"
            elif "DERIVED AJCC STAGE" in col_upper:
                rename[col] = "STAGE"
            elif col_upper == "GRADE":
                rename[col] = "GRADE"
            elif "CS TUMOR SIZE" in col_upper or "TUMOR SIZE" in col_upper:
                rename[col] = "TUMOR_SIZE_MM"
            elif "REGIONAL NODES EXAMINED" in col_upper:
                rename[col] = "REGIONAL_NODES_EXAMINED"
            elif "REGIONAL NODES POSITIVE" in col_upper:
                rename[col] = "REGIONAL_NODES_POSITIVE"
            elif "SURVIVAL MONTHS" in col_upper:
                rename[col] = "SURVIVAL_MONTHS"
            elif "VITAL STATUS" in col_upper:
                rename[col] = "VITAL_STATUS"
            elif "YEAR OF DIAGNOSIS" in col_upper:
                rename[col] = "YEAR_OF_DIAGNOSIS"
            elif "SEQUENCE NUMBER" in col_upper:
                rename[col] = "SEQUENCE_NUMBER"
            elif "LATERALITY" in col_upper:
                rename[col] = "LATERALITY"
            elif "SURGERY" in col_upper:
                rename[col] = "SURGERY"
            elif "RADIATION" in col_upper:
                rename[col] = "RADIATION"
            elif "CHEMO" in col_upper:
                rename[col] = "CHEMOTHERAPY"

        df = df.rename(columns=rename)

        # SEER vital status: "Alive" → 0, "Dead" → 1
        if "VITAL_STATUS" in df.columns:
            df["VITAL_STATUS"] = (
                df["VITAL_STATUS"]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"ALIVE": 0, "DEAD": 1, "0": 0, "1": 1})
                .fillna(df["VITAL_STATUS"])
            )

        if "PATIENT_ID" not in df.columns:
            df["PATIENT_ID"] = df.index.astype(str)

        # Select output columns (use those present)
        desired_cols = [
            "PATIENT_ID",
            "AGE_AT_DIAGNOSIS",
            "SEX",
            "RACE",
            "PRIMARY_SITE",
            "HISTOLOGY",
            "STAGE",
            "GRADE",
            "TUMOR_SIZE_MM",
            "REGIONAL_NODES_EXAMINED",
            "REGIONAL_NODES_POSITIVE",
            "SURVIVAL_MONTHS",
            "VITAL_STATUS",
            "YEAR_OF_DIAGNOSIS",
            "SEQUENCE_NUMBER",
            "LATERALITY",
            "SURGERY",
            "RADIATION",
            "CHEMOTHERAPY",
        ]
        present_cols = [c for c in desired_cols if c in df.columns]
        df_out = df[present_cols]

        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} SEER records to {output_path}")
