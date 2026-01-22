from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
base_dataset = MIMIC4Dataset(
    ehr_root="/srv/local/data/physionet.org/files/mimiciv/2.2/",
    ehr_tables=["patients", "admissions", "diagnoses_icd", "procedures_icd", "prescriptions"],
)
sample_dataset = base_dataset.set_task(LengthOfStayPredictionMIMIC4())
