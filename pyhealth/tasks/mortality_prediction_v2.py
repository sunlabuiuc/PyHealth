from dataclasses import dataclass, field
from typing import Dict
from datetime import timedelta
from pyhealth.tasks.task_template import TaskTemplate


@dataclass(frozen=True)
class Mortality30DaysMIMIC4(TaskTemplate):
    task_name: str = "Mortality30Days"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"diagnoses": "sequence", "procedures": "sequence"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"mortality": "label"})

    def __call__(self, patient):
        death_datetime = patient.attr_dict["death_datetime"]
        diagnoses = patient.get_events_by_type("diagnoses_icd")
        procedures = patient.get_events_by_type("procedures_icd")
        mortality = 0
        if death_datetime is not None:
            mortality = 1
            # remove events 30 days before death
            diagnoses = [
                diag
                for diag in diagnoses
                if diag.timestamp <= death_datetime - timedelta(days=30)
            ]
            procedures = [
                proc
                for proc in procedures
                if proc.timestamp <= death_datetime - timedelta(days=30)
            ]
        diagnoses = [diag.attr_dict["code"] for diag in diagnoses]
        procedures = [proc.attr_dict["code"] for proc in procedures]

        if len(diagnoses) * len(procedures) == 0:
            return []

        samples = [
            {
                "patient_id": patient.patient_id,
                "diagnoses": diagnoses,
                "procedures": procedures,
                "mortality": mortality,
            }
        ]
        return samples



if __name__ == "__main__":
    from pyhealth.datasets import MIMIC4Dataset

    dataset = MIMIC4Dataset(
        root="/srv/local/data/physionet.org/files/mimiciv/2.0/hosp",
        tables=["procedures_icd"],
        dev=True,
    )
    task = Mortality30DaysMIMIC4()
    samples = dataset.set_task(task)
    print(samples[0])
    print(len(samples))
