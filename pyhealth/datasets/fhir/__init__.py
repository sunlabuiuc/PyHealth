"""FHIR datasets: a generic engine + per-source subclasses.

- :class:`~pyhealth.datasets.fhir.base.FHIRDataset` — generic, config-driven base.
- :class:`~pyhealth.datasets.fhir.mimic4.MIMIC4FHIR` — MIMIC-IV-on-FHIR (R4).
- :mod:`~pyhealth.datasets.fhir.utils` — the stateless flattening engine
  (``Col``, ``ResourceSpec``, ``flatten_resource``, …).

Authors:
    John Wu and Evan Febrianto
"""

from .base import FHIRDataset
from .mimic4 import MIMIC4FHIR
from .utils import Col, ResourceSpec

__all__ = ["FHIRDataset", "MIMIC4FHIR", "Col", "ResourceSpec"]
