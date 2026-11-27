# Contributor: Arun Vishnu (anv5@illinois.edu)
# Paper: Evaluating Model Performance in Medical Datasets Over Time
# https://proceedings.mlr.press/v209/zhou23a.html
# Dataset: CDC COVID-19 case-surveillance public-use records.

"""CDC COVID-19 case-surveillance dataset.

The public-use source contains deidentified case records, not patient histories.
"""

import csv
from pathlib import Path
from typing import Any

from .base_dataset import BaseDataset
from .configs.config import load_yaml_config


class CDCCOVIDDataset(BaseDataset):
    """Load CDC COVID-19 Case Surveillance Public Use Data with Geography.

    The CDC source is a 19-column, deidentified, case-level CSV. It has no
    person identifier and must not be interpreted as longitudinal patient
    data. The bundled configuration therefore sets ``patient_id: null``;
    :class:`BaseDataset` assigns one identifier per physical CSV row.

    Download the CSV for Socrata dataset ``n8mc-b4w4`` and place it at
    ``root/n8mc-b4w4.csv``. A bounded export with the same columns is also
    supported. This loader never downloads data itself.

    Args:
        root: Directory containing ``n8mc-b4w4.csv``.
        tables: None or ``["cases"]``.
        config_path: Optional compatible YAML configuration.
        **kwargs: Options forwarded to :class:`BaseDataset`, such as
            ``cache_dir``, ``dev`` and ``num_workers``.

    Raises:
        FileNotFoundError: The configuration or CSV does not exist.
        ValueError: The table selection or source schema is incompatible.

    Examples:
        >>> from pyhealth.datasets import CDCCOVIDDataset
        >>> dataset = CDCCOVIDDataset(
        ...     root="/data/cdc-covid", cache_dir="/data/cache"
        ... )  # doctest: +SKIP
        >>> dataset.unique_patient_ids[:2]  # one ID per case row
        ... # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        config_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        tables = ["cases"] if tables is None else tables
        if tables != ["cases"]:
            raise ValueError("Only the cases table is supported")
        config_path = config_path or str(
            Path(__file__).parent / "configs" / "cdc_covid.yaml"
        )
        config = load_yaml_config(config_path)
        table = config.tables.get("cases")
        if (
            table is None
            or table.patient_id is not None
            or table.timestamp != "case_month"
            or table.timestamp_format != "%Y-%m"
            or table.join
        ):
            raise ValueError(
                "CDC COVID configuration must preserve case-level row identity "
                "and the case_month timestamp"
            )

        relative = Path(table.file_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("CDC COVID source must be a local file inside root")
        path = Path(root) / relative
        with path.open(newline="", encoding="utf-8-sig") as stream:
            columns = next(csv.reader(stream), [])
        expected = ["case_month", *table.attributes]
        if len(columns) != len(set(columns)) or set(columns) != set(expected):
            missing = sorted(set(expected).difference(columns))
            extra = sorted(set(columns).difference(expected))
            raise ValueError(
                f"CDC COVID CSV schema mismatch; missing={missing}, extra={extra}"
            )

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=kwargs.pop("dataset_name", None) or "cdc_covid",
            config_path=config_path,
            **kwargs,
        )
