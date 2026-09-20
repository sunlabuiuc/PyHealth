"""Synthea Bulk FHIR R4 output exposed through the PyHealth FHIR pipeline."""

from __future__ import annotations

import os

from ..synthea_generator import SyntheaGenerator
from .base import FHIRDataset


class SyntheaFHIRDataset(FHIRDataset):
    """Loads Bulk FHIR NDJSON output from a :class:`SyntheaGenerator`."""

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "configs", "synthea_fhir.yaml"
    )
    DATASET_NAME = "synthea_fhir"

    def __init__(self, generator: SyntheaGenerator, **kwargs) -> None:
        if not generator.fhir_enabled:
            raise ValueError(
                "SyntheaFHIRDataset requires Bulk FHIR output to be enabled"
            )
        self.generator = generator
        super().__init__(root=str(generator.output_path("fhir")), **kwargs)

    def _ensure_prepared_tables(self) -> None:
        self.generator.ensure_generated()
        super()._ensure_prepared_tables()
