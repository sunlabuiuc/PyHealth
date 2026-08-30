"""MIMIC-IV-on-FHIR (R4) dataset.

A thin :class:`~pyhealth.datasets.fhir.base.FHIRDataset` wrapper that points at
the bundled YAML for the PhysioNet MIMIC-IV on FHIR export. The whole ingest
contract (resource projection + downstream table schema + glob hints) lives in
the YAML; this class only names its default path.

Use this YAML as the worked example when authoring a config for a different
FHIR export — copy ``pyhealth/datasets/fhir/configs/mimic4fhir.yaml`` and
adapt the ``resource_specs:`` and ``tables:`` blocks.

Authors:
    John Wu and Evan Febrianto
"""

from __future__ import annotations

import os

from .base import FHIRDataset


class MIMIC4FHIR(FHIRDataset):
    """MIMIC-IV-on-FHIR (R4) dataset.

    Streams the PhysioNet MIMIC-IV on FHIR NDJSON.GZ export into flattened
    Patient/Encounter/Condition/Observation/MedicationRequest/Procedure tables,
    then runs the standard :class:`~pyhealth.datasets.BaseDataset` pipeline.

    The bundled config at ``pyhealth/datasets/fhir/configs/mimic4fhir.yaml``
    matches both the PhysioNet 2.1.0 demo and the full release. Override
    ``config_path=`` to point at a customised copy.

    Examples:
        >>> ds = MIMIC4FHIR(root="/data/mimic-iv-fhir", max_patients=500)
        >>> sample_ds = ds.set_task(task, num_workers=4)
    """

    DEFAULT_CONFIG_PATH = os.path.join(
        os.path.dirname(__file__), "configs", "mimic4fhir.yaml"
    )
    DATASET_NAME = "mimic4fhir"
