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


class SyntheaDataset(BaseDataset):
    """Loads CSV output from a :class:`SyntheaGenerator`."""

    def __init__(
        self,
        generator: SyntheaGenerator,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if not generator.csv_enabled:
            raise ValueError("SyntheaDataset requires CSV output to be enabled")
        self.generator = generator
        self._explicit_tables = bool(tables)
        selected = list(dict.fromkeys(DEFAULT_TABLES if not tables else tables))
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "synthea_csv.yaml")
        super().__init__(
            root=str(generator.output_path("csv")),
            tables=selected,
            dataset_name=dataset_name or "synthea",
            config_path=config_path,
            **kwargs,
        )

    def load_data(self):
        """Generates missing CSV output before loading selected tables."""
        self.generator.ensure_generated()
        root = self.generator.resolved_output_path("csv")
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
        """Normalizes legacy Synthea procedure timestamps."""
        if "start" not in frame.columns:
            frame = frame.with_columns(nw.col("date").alias("start"))
        return frame


SyntheaCSVDataset = SyntheaDataset
