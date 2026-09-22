"""Synthea CSV output exposed through the standard PyHealth dataset API."""

from pathlib import Path

import narwhals as nw

from .base_dataset import BaseDataset
from .synthea_generator import SyntheaGenerator

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
    """Loads generated Synthea CSV files through the PyHealth dataset API.

    Generation remains lazy: constructing the dataset does not run Synthea.
    The first data load asks the associated generator to create any missing CSV
    files before delegating parsing and normalization to :class:`BaseDataset`.

    Examples:
        >>> from pyhealth.datasets import SyntheaCSVDataset, SyntheaGenerator
        >>> generator = SyntheaGenerator("./synthea-output", population=10)
        >>> dataset = SyntheaCSVDataset(generator, tables=["patients"])
        >>> events = dataset.load_data()
    """

    def __init__(
        self,
        generator: SyntheaGenerator,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        """Initializes a Synthea CSV dataset.

        Args:
            generator (SyntheaGenerator): Generator that owns the CSV output.
            tables (list[str], optional): CSV tables to load. The supported
                defaults are used when omitted or empty.
            dataset_name (str, optional): Dataset name used by PyHealth.
            config_path (str, optional): Dataset schema configuration. The
                bundled Synthea CSV schema is used when omitted.
            **kwargs: Additional arguments passed to :class:`BaseDataset`.
        """
        self.generator = generator
        self._explicit_tables = bool(tables)
        selected = list(dict.fromkeys(DEFAULT_TABLES if not tables else tables))
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "synthea_csv.yaml")
        super().__init__(
            root=str(generator.output_path()),
            tables=selected,
            dataset_name=dataset_name or "synthea",
            config_path=config_path,
            **kwargs,
        )

    def load_data(self):
        """Generates missing CSV files and loads the selected tables.

        Default tables that Synthea did not emit are skipped. A missing table
        requested explicitly is treated as an error.

        Returns:
            dd.DataFrame: Concatenated lazy Dask DataFrame for the selected
            tables.

        Raises:
            RuntimeError: If Synthea fails or an explicitly selected table is
                not generated.
        """
        self.generator.ensure_generated()
        root = self.generator.resolved_output_path()
        self.root = str(root)
        missing = [
            table for table in self.tables if not (root / f"{table}.csv").is_file()
        ]
        if missing and self._explicit_tables:
            raise RuntimeError(f"Synthea did not generate tables: {', '.join(missing)}")
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
