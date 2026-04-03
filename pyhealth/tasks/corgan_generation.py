from typing import Dict, List

import polars as pl

from pyhealth.tasks.base_task import BaseTask


class CorGANGenerationMIMIC3(BaseTask):
    """Task function for CorGAN synthetic EHR generation using MIMIC-III.

    Extracts ICD-9 diagnosis codes from MIMIC-III admission records into a
    flat list of codes suitable for training the CorGAN model.

    CorGAN is a bag-of-codes model: it collapses all visit codes for a patient
    into a single binary vector, so visit structure is irrelevant. All codes
    from all admissions are pooled into one flat list per patient.
    Patients with no codes are excluded.

    Attributes:
        task_name (str): Unique task identifier.
        input_schema (dict): Schema descriptor — ``"visits"`` field uses
            ``"multi_hot"`` encoding (flat list of code strings).
        output_schema (dict): Empty — generative task, no conditioning label.
        _icd_col (str): Polars column path for ICD codes in MIMIC-III.

    Examples:
        >>> fn = CorGANGenerationMIMIC3()
        >>> fn.task_name
        'CorGANGenerationMIMIC3'
    """

    task_name = "CorGANGenerationMIMIC3"
    input_schema = {"visits": "multi_hot"}
    output_schema = {}
    _icd_col = "diagnoses_icd/icd9_code"

    def __call__(self, patient) -> List[Dict]:
        """Extract flat code list for a single patient.

        All ICD codes from all admissions are pooled into a single flat list.
        Visit temporal structure is discarded because CorGAN operates on a
        single multi-hot binary vector per patient.

        Args:
            patient: A PyHealth patient object with admission and diagnosis
                event data.

        Returns:
            list of dict: A single-element list containing the patient record,
                or an empty list if the patient has no diagnosis codes. Each
                dict has:
            ``"patient_id"`` (str): the patient identifier.
            ``"visits"`` (list of str): flat list of all ICD codes across all
                admissions.
        """
        admissions = list(patient.get_events(event_type="admissions"))
        all_codes = []
        for adm in admissions:
            codes = (
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
            all_codes.extend(codes)
        if not all_codes:
            return []
        return [{"patient_id": patient.patient_id, "visits": all_codes}]


class CorGANGenerationMIMIC4(CorGANGenerationMIMIC3):
    """Task function for CorGAN synthetic EHR generation using MIMIC-IV.

    Inherits all logic from :class:`CorGANGenerationMIMIC3`. Overrides only
    the task name and the ICD code column to match the MIMIC-IV schema, where
    the column is ``icd_code`` (unversioned) rather than ``icd9_code``.

    Attributes:
        task_name (str): Unique task identifier.
        _icd_col (str): Polars column path for ICD codes in MIMIC-IV.

    Examples:
        >>> fn = CorGANGenerationMIMIC4()
        >>> fn.task_name
        'CorGANGenerationMIMIC4'
    """

    task_name = "CorGANGenerationMIMIC4"
    _icd_col = "diagnoses_icd/icd_code"


corgan_generation_mimic3_fn = CorGANGenerationMIMIC3()
corgan_generation_mimic4_fn = CorGANGenerationMIMIC4()
