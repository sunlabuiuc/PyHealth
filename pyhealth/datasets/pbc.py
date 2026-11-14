"""Local YAML-backed PBC cohort for baseline competing-risk prediction.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import csv
from pathlib import Path
from typing import Any

import pandas as pd

from ..tasks.competing_risks_pbc import PBC_CATEGORIES, PBC_NUMERIC
from .base_dataset import BaseDataset
from .configs.config import load_yaml_config


class PBCDataset(BaseDataset):
    """Load the author's longitudinal pbc2.csv without learned preprocessing.

    The source contains 312 randomized PBC participants, with death and liver
    transplantation as competing first events. Supply a local CSV and review
    its data-use terms; this class neither downloads nor redistributes records.
    R's 418-person pbc dataset has a different schema and event mapping.

    Args:
        root: Directory containing pbc2.csv (or the configured local CSV).
        tables: None or ["pbc"].
        config_path: Optional YAML with the same attribute and identifier schema.
        **kwargs: BaseDataset options, including dataset_name and a controlled
            cache_dir. An omitted/None dataset_name defaults to "pbc".

    Raises:
        FileNotFoundError: The local CSV or configuration does not exist.
        ValueError: Schema/columns are incompatible or a patient ID is missing.

    Examples:
        >>> from pyhealth.tasks import CompetingRisksPBC
        >>> cohort = PBCDataset(root="/local/pbc", cache_dir="/local/cache")
        ... # doctest: +SKIP
        >>> samples = cohort.set_task(CompetingRisksPBC())  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        tables = ["pbc"] if tables is None else tables
        if tables != ["pbc"]:
            raise ValueError("Only the pbc table is supported")
        config_path = config_path or str(Path(__file__).parent / "configs" / "pbc.yaml")
        config = load_yaml_config(config_path)
        required = {*PBC_CATEGORIES, *PBC_NUMERIC, "year", "years", "status"}
        table = config.tables.get("pbc")
        if (
            table is None
            or table.patient_id != "id"
            or table.timestamp is not None
            or table.join
            or not required.issubset(table.attributes)
        ):
            raise ValueError("PBC configuration must preserve the baseline schema")
        relative = Path(table.file_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("PBC source must be a local file inside root")
        path = Path(root) / relative
        with path.open(newline="", encoding="utf-8-sig") as stream:
            columns = next(csv.reader(stream), [])
        missing = {"id", *table.attributes}.difference(columns)
        if missing or len(columns) != len(set(columns)):
            raise ValueError(f"Missing or duplicate PBC columns: {sorted(missing)}")
        identifiers = pd.read_csv(path, usecols=["id"], dtype=str)["id"]
        if (identifiers.isna() | identifiers.str.strip().eq("")).any():
            raise ValueError("Every PBC row must have a nonempty patient ID")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=kwargs.pop("dataset_name", None) or "pbc",
            config_path=config_path,
            **kwargs,
        )
