# Author: Chinmaykatpatal
# Description: In-hospital mortality task for MAESTRO on MIMIC-IV EHR data.

"""In-hospital mortality from the first 48 ICU hours, split into MAESTRO modalities.

Uses only the EHR tables, so it runs on the open MIMIC-IV demo dataset (no
credentialing and no chest X-rays). It reuses the MedFuse variable definitions,
binning and stay exclusions of :mod:`pyhealth.tasks.ehr_cxr_mimic4`, so its
inputs match the EHR side of :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4`.

Each MAESTRO modality is a group of clinically related variables, with the
value and the measured-in-this-bin mask of each variable.
"""

from datetime import timedelta
from typing import Any, ClassVar

import polars as pl

from .base_task import BaseTask
from .ehr_cxr_mimic4 import (
    _EVENT_TABLES,
    _MEASURED_NAMES,
    _eligible_stays,
    _measurements,
    _pre_filter,
    _time_series,
)

# Modality name -> MedFuse variables in that modality. Together they cover the
# 16 measured variables used by the MedFuse EHR inputs.
MODALITY_GROUPS: dict[str, list[str]] = {
    "vitals": ["hr", "sbp", "dbp", "map", "resp", "temp", "o2sat", "crr"],
    "neuro": ["gcs_eye", "gcs_motor", "gcs_verbal"],
    "labs": ["glu", "ph", "fio2"],
    "anthropometrics": ["height", "weight"],
}


def _modality_columns(modality: str) -> list[int]:
    """Column indices of a modality in the ``[bins, 2 * 17]`` EHR array.

    The first 17 columns are the values (16 measured variables and the GCS
    total); the next 17 are their masks.
    """
    n = len(_MEASURED_NAMES) + 1
    values = [_MEASURED_NAMES.index(name) for name in MODALITY_GROUPS[modality]]
    return values + [n + index for index in values]


class MAESTROMortalityMIMIC4(BaseTask):
    """In-hospital mortality from the first 48 ICU hours, for MAESTRO.

    One sample per ICU stay of at least 48 hours in a hospital admission with
    a known outcome. Stay exclusions are those of
    :class:`~pyhealth.tasks.EHRCXRMortalityMIMIC4`. Stays without any
    measurement in the window are dropped.

    Each modality in :data:`MODALITY_GROUPS` is a ``[24, k]`` tensor: the
    variable values and masks in 2-hour bins over the window, where ``k`` is
    twice the number of variables in the group. Values still missing before a
    variable's first measurement are imputed with zero, as in the MedFuse
    inputs.

    ``mortality``: ``hospital_expire_flag`` of the stay's admission.

    Requires the ``patients``, ``admissions``, ``icustays``, ``chartevents``
    and ``labevents`` tables. No CXR table is needed.

    Examples:
        >>> from pyhealth.tasks import MAESTROMortalityMIMIC4
        >>> task = MAESTROMortalityMIMIC4()
        >>> sorted(task.input_schema)
        ['anthropometrics', 'labs', 'neuro', 'vitals']
        >>> task.output_schema
        {'mortality': 'binary'}
    """

    task_name: str = "MAESTROMortalityMIMIC4"
    output_schema: ClassVar[dict[str, Any]] = {"mortality": "binary"}

    OBSERVATION_HOURS: ClassVar[int] = 48

    def __init__(self, bin_hours: int = 2) -> None:
        self.input_schema = {
            name: (
                "tensor",
                {"normalization": "zscore", "impute_strategy": "zero"},
            )
            for name in MODALITY_GROUPS
        }
        super().__init__()
        self.bin_hours = bin_hours

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Drops measurement rows whose item ID is not used by the task."""
        return _pre_filter(
            df,
            self.task_name,
            [*_EVENT_TABLES, "admissions", "icustays", "patients"],
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds the samples of one patient.

        Args:
            patient: The patient.

        Returns:
            One sample per eligible stay, with ``patient_id``, ``visit_id``,
            ``stay_id``, one tensor per modality, and ``mortality``.
        """
        stays = _eligible_stays(patient)
        if stays.height == 0:
            return []
        measurements = _measurements(patient)
        expired = {
            int(event.hadm_id): int(event.hospital_expire_flag)
            for event in patient.get_events(event_type="admissions")
            if event.hospital_expire_flag is not None
        }

        window = timedelta(hours=self.OBSERVATION_HOURS)
        samples = []
        for stay_id, hadm_id, intime, outtime in stays.iter_rows():
            if outtime - intime < window or hadm_id not in expired:
                continue
            end = intime + window
            ehr = _time_series(measurements, stay_id, intime, end, self.bin_hours)
            if ehr is None:
                continue
            sample: dict[str, Any] = {
                "patient_id": patient.patient_id,
                "visit_id": str(hadm_id),
                "stay_id": stay_id,
                "mortality": expired[hadm_id],
            }
            for modality in MODALITY_GROUPS:
                sample[modality] = ehr[:, _modality_columns(modality)]
            samples.append(sample)
        return samples
