"""Load tumor records from a formatted SEER*Stat Case Listing export.

Contributors:
    Parimala Palani (ppalani2), Kushaal Manchella (kushaal2),
    and Nick King (nrking2).

Paper:
    Reproducible Survival Prediction with SEER Cancer Data.
    https://proceedings.mlr.press/v85/hegselmann18a.html
"""

import csv
import re
from pathlib import Path

import dask.dataframe as dd
import pandas as pd

from .base_dataset import BaseDataset

_COLUMNS = {
    "patient_key": "Patient ID",
    "record_number": "Record number recode",
    "diagnosis_year": "Year of diagnosis",
    "site_recode": "Site recode ICD-O-3/WHO 2008",
    "primary_site": "Primary Site",
    "sex": "Sex",
    "age_group": "Age recode with <1 year olds and 90+",
    "histology": "Histologic Type ICD-O-3",
    "grade": "Grade Recode (thru 2017)",
    "stage": "Derived AJCC Stage Group, 6th ed (2004-2015)",
    "laterality": "Laterality",
    "marital_status": "Marital status at diagnosis",
    "tumor_size": "CS tumor size (2004-2015)",
    "extension": "CS extension (2004-2015)",
    "lymph_nodes": "CS lymph nodes (2004-2015)",
    "metastases": "CS mets at dx (2004-2015)",
    "survival_months": "Survival months",
    "cause_of_death": "SEER cause-specific death classification",
}

_DEATH_CLASSES = {
    "0": "alive_or_other_cause",
    "Alive or dead of other cause": "alive_or_other_cause",
    "1": "cancer_death",
    "Dead (attributable to this cancer dx)": "cancer_death",
    "8": "unknown",
    "Dead (missing/unknown COD)": "unknown",
    "9": "not_applicable",
    "N/A not seq 0-59": "not_applicable",
    "": "unknown",
}

_FEATURE_FORMATS = {
    "age_group": re.compile(r"[0-9]{1,2}(?:-[0-9]{1,2}|\+)?(?: years?)?|<1 year"),
    "histology": re.compile(r"[0-9]{4}"),
    "stage": re.compile(
        r"(?:0(?:a|is)?|I{1,3}|IV)(?:[A-C]| NOS)?|OCCULT", re.IGNORECASE
    ),
}
_MISSING_FEATURE_VALUES = {"", "unknown", "unk stage", "na", "n/a", "blank(s)"}


class SEERDataset(BaseDataset):
    """Represents each tumor as one event grouped by its SEER patient key.

    Args:
        root: Directory containing the formatted ``seer_export.txt`` Case
            Listing. The file must be comma-delimited and include headers.
        cache_dir: Location for processed records. Use approved local storage
            for the source and its cache.
        num_workers: Number of worker processes for PyHealth processing.

    The dataset preserves outcome data for task construction and leaves
    eligibility, labels, and feature encoding to the task.

    Examples:
        >>> dataset = SEERDataset("local_data", cache_dir="cache")  # doctest: +SKIP
        >>> dataset.get_patient("00001").get_events("tumor")  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str | Path,
        *,
        cache_dir: str | Path,
        num_workers: int = 1,
    ) -> None:
        super().__init__(
            root=str(Path(root).expanduser().resolve()),
            tables=["tumor"],
            config_path=Path(__file__).parent / "configs" / "seer.yaml",
            cache_dir=cache_dir,
            num_workers=num_workers,
        )

    def _scan_table(self, source_path: str) -> dd.DataFrame:
        """Read and validate the SEER export before PyHealth builds its cache.

        Args:
            source_path: Path to the local Case Listing export.

        Returns:
            A Dask dataframe with one row per tumor and normalized fields.

        Raises:
            ValueError: The export has missing columns, malformed rows,
                incomplete or duplicate tumor keys, or invalid field values.
        """
        path = Path(source_path)
        with path.open("rb") as stream:
            encoding = "utf-8-sig" if stream.read(3) == b"\xef\xbb\xbf" else "cp1252"

        records = []
        keys = set()
        with path.open(encoding=encoding, newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = [name.strip() for name in next(reader, [])]
            missing = sorted(set(_COLUMNS.values()) - set(header))
            if missing:
                raise ValueError(f"Missing SEER export columns: {', '.join(missing)}")
            if len(header) != len(set(header)):
                raise ValueError("Duplicate column names in SEER export")
            indexes = {
                field: header.index(source) for field, source in _COLUMNS.items()
            }
            try:
                for row in reader:
                    line = reader.line_num
                    if len(row) != len(header):
                        raise ValueError(f"Wrong number of fields on SEER line {line}")
                    values = {name: row[i].strip() for name, i in indexes.items()}
                    key = (values["patient_key"], values["record_number"])
                    if not all(key):
                        raise ValueError(f"Incomplete tumor key on SEER line {line}")
                    if key in keys:
                        raise ValueError(f"Duplicate tumor key on SEER line {line}")
                    keys.add(key)
                    for field, pattern in _FEATURE_FORMATS.items():
                        value = values[field]
                        if (
                            value.lower() not in _MISSING_FEATURE_VALUES
                            and not pattern.fullmatch(value)
                        ):
                            raise ValueError(f"Invalid SEER {field} on line {line}")
                    months = values["survival_months"]
                    year = values["diagnosis_year"]
                    cause = values["cause_of_death"]
                    if len(year) != 4 or not (year.isascii() and year.isdecimal()):
                        raise ValueError(f"Invalid diagnosis year on SEER line {line}")
                    if months in {"", "9999", "Unknown"}:
                        survival_months = None
                    elif months.isascii() and months.isdecimal():
                        survival_months = int(months)
                    else:
                        raise ValueError(f"Invalid survival months on SEER line {line}")
                    if cause not in _DEATH_CLASSES:
                        raise ValueError(
                            f"Unrecognized cause of death on SEER line {line}"
                        )
                    records.append(
                        {
                            **values,
                            "survival_months": survival_months,
                            "cause_of_death": _DEATH_CLASSES[cause],
                            "diagnosis_year": int(year),
                        }
                    )
            except csv.Error as error:
                raise ValueError(
                    f"Malformed CSV on SEER line {reader.line_num}"
                ) from error
        if not records:
            raise ValueError("The SEER export has no tumor records")
        frame = pd.DataFrame.from_records(records)
        frame["survival_months"] = pd.array(
            [record["survival_months"] for record in records], dtype="Int64"
        )
        return dd.from_pandas(frame, chunksize=50_000)
