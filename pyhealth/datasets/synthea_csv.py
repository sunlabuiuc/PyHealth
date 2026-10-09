"""Synthea CSV output exposed through the standard PyHealth dataset API."""

from pathlib import Path

import narwhals as nw

from .base_dataset import BaseDataset

DEFAULT_TABLES = [
    "patients",
    "encounters",
    "conditions",
    "medications",
    "observations",
    "procedures",
    "immunizations",
    "careplans",
    "allergies",
    "devices",
    "imaging_studies",
    "supplies",
    "payer_transitions",
]


class SyntheaCSVDataset(BaseDataset):
    """Loads Synthea CSV files through the PyHealth dataset API.

    The dataset only reads files; it does not run Synthea. Produce the CSV
    directory with :class:`~pyhealth.models.Synthea` first, or point ``root``
    at any existing Synthea CSV export.

    Each row becomes a standard PyHealth event (``patient_id``,
    ``event_type``, ``timestamp`` plus the table's attributes), but Synthea's
    codes are SNOMED-CT, RxNorm, LOINC, and CVX rather than ICD and NDC, so
    MIMIC tasks and ICD code mappings do not apply directly.

    Examples:
        >>> from pyhealth.datasets import SyntheaCSVDataset
        >>> from pyhealth.models import Synthea
        >>> csv_dir = Synthea("./synthea-output").generate(population=10, seed=42)
        >>> dataset = SyntheaCSVDataset(csv_dir, tables=["conditions"])
        >>> patient = dataset.get_patient(dataset.unique_patient_ids[0])
        >>> event = patient.get_events(event_type="conditions")[0]
        >>> event.code, event.description  # SNOMED-CT code and its text
    """

    def __init__(
        self,
        root: str | Path,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        """Initializes a Synthea CSV dataset.

        Args:
            root (str or Path): Directory containing the Synthea CSV files.
            tables (list[str], optional): CSV tables to load. The supported
                defaults are used when omitted or empty.
            dataset_name (str, optional): Dataset name used by PyHealth.
            config_path (str, optional): Dataset schema configuration. The
                bundled Synthea CSV schema is used when omitted.
            **kwargs: Additional arguments passed to :class:`BaseDataset`.
        """
        self._explicit_tables = bool(tables)
        selected = list(dict.fromkeys(DEFAULT_TABLES if not tables else tables))
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "synthea_csv.yaml")
        super().__init__(
            root=str(root),
            tables=selected,
            dataset_name=dataset_name or "synthea",
            config_path=config_path,
            **kwargs,
        )

    def load_data(self):
        """Loads the selected tables, skipping default tables not emitted.

        Synthea omits CSV files for record types with no rows. A missing
        default table is skipped; a missing table requested explicitly is
        treated as an error.

        Returns:
            dd.DataFrame: Concatenated lazy Dask DataFrame for the selected
            tables.

        Raises:
            RuntimeError: If an explicitly selected table is missing.
        """
        root = Path(self.root)
        missing = [
            table for table in self.tables if not (root / f"{table}.csv").is_file()
        ]
        if missing and self._explicit_tables:
            raise RuntimeError(f"Synthea CSV tables not found: {', '.join(missing)}")
        if missing:
            self.tables = [table for table in self.tables if table not in missing]
        return super().load_data()

    def preprocess_procedures(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Normalizes legacy Synthea procedure timestamps.

        Older exports use ``date`` where the current schema uses ``start``.
        Existing ``start`` columns are preserved.

        Args:
            frame (nw.LazyFrame): Procedure table before schema normalization.

        Returns:
            nw.LazyFrame: Procedure table containing a ``start`` column.
        """
        if "start" not in frame.columns:
            frame = frame.with_columns(nw.col("date").alias("start"))
        return frame
