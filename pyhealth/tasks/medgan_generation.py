import polars as pl
from typing import Dict, List

from pyhealth.tasks.base_task import BaseTask


class MedGANGenerationMIMIC3(BaseTask):
    """MedGAN generation task for MIMIC-III.

    Aggregates all ICD-9 diagnosis codes across all admissions into a
    single flat list per patient, matching the ``multi_hot`` input schema
    expected by :class:`~pyhealth.models.MedGAN`.

    Args:
        None

    Examples:
        >>> task = MedGANGenerationMIMIC3()
        >>> task.task_name
        'MedGANGenerationMIMIC3'
    """

    task_name = "MedGANGenerationMIMIC3"
    input_schema = {"visits": "multi_hot"}
    output_schema = {}
    _icd_col = "diagnoses_icd/icd9_code"

    def __call__(self, patient) -> List[Dict]:
        admissions = list(patient.get_events(event_type="admissions"))
        codes = []
        for adm in admissions:
            visit_codes = (
                patient.get_events(
                    event_type="diagnoses_icd",
                    filters=[("hadm_id", "==", adm.hadm_id)],
                    return_df=True,
                )
                .select(pl.col(self._icd_col))
                .to_series()
                .drop_nulls()
                .to_list()
            )
            codes.extend(visit_codes)
        if not codes:
            return []
        return [{"patient_id": patient.patient_id, "visits": codes}]


class MedGANGenerationMIMIC4(MedGANGenerationMIMIC3):
    """MedGAN generation task for MIMIC-IV.

    Identical to :class:`MedGANGenerationMIMIC3` but uses the MIMIC-IV
    ICD column name ``diagnoses_icd/icd_code``.

    Examples:
        >>> task = MedGANGenerationMIMIC4()
        >>> task.task_name
        'MedGANGenerationMIMIC4'
    """

    task_name = "MedGANGenerationMIMIC4"
    _icd_col = "diagnoses_icd/icd_code"


medgan_generation_mimic3_fn = MedGANGenerationMIMIC3()
medgan_generation_mimic4_fn = MedGANGenerationMIMIC4()
