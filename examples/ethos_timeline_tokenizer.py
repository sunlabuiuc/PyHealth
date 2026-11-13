"""Tokenize MEDS patient timelines the way ETHOS does.

Smoke test on synthetic data: this writes a five-patient MEDS directory to a
temporary folder, so it needs no credentials and no download. The numbers mean
nothing clinically and nothing here reproduces a result from the paper.

The steps are the same on a real MEDS export: point ``MEDSDataset`` at it,
keep the task below, and fit on the training patients only.

Usage:
    python examples/ethos_timeline_tokenizer.py
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq

from pyhealth.datasets import MEDSDataset, PatientSplit, get_dataloader
from pyhealth.tasks.base_task import BaseTask


class EthosTimeline(BaseTask):
    """One sample per patient: the whole MEDS timeline.

    The label only makes the task complete; nothing is trained here.
    """

    task_name = "EthosTimeline"
    input_schema: ClassVar[dict[str, str]] = {"timeline": "ethos"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __call__(self, patient):
        events = patient.get_events(event_type="meds", return_df=True)
        if events.height == 0:
            return []
        return [
            {
                "patient_id": patient.patient_id,
                "timeline": {
                    "time": events["timestamp"].to_list(),
                    "code": events["meds/code"].to_list(),
                    "numeric_value": events["meds/numeric_value"].to_list(),
                },
                "label": int(patient.patient_id) % 2,
            }
        ]


def write_synthetic_meds(root: Path) -> None:
    """Writes five patients in the MEDS layout."""
    start = datetime.fromisoformat("2024-01-01T08:00:00")  # MEDS times are naive
    rows = []
    for patient in range(5):
        hemoglobin = 10.0 + patient
        rows += [
            (patient, None, "GENDER//F", None),
            (patient, start, "HOSPITAL_ADMISSION//EMERGENCY", None),
            (patient, start, "LAB//HEMOGLOBIN", hemoglobin),
            (patient, start, "LAB//LACTATE", 1.0 + patient / 2),
            (patient, start + timedelta(minutes=3), "MEDICATION//ASPIRIN", None),
            (patient, start + timedelta(hours=6), "LAB//HEMOGLOBIN", hemoglobin + 1),
            (patient, start + timedelta(days=2), "HOSPITAL_DISCHARGE//HOME", None),
            (patient, start + timedelta(days=400), "HOSPITAL_ADMISSION//ELECTIVE", None),
        ]
    subject_id, time, code, numeric_value = map(list, zip(*rows))
    (root / "data" / "train").mkdir(parents=True)
    (root / "metadata").mkdir()
    pq.write_table(
        pa.table(
            {
                "subject_id": pa.array(subject_id, type=pa.int64()),
                "time": pa.array(time, type=pa.timestamp("us")),
                "code": pa.array(code, type=pa.string()),
                "numeric_value": pa.array(numeric_value, type=pa.float32()),
            }
        ),
        root / "data" / "train" / "0.parquet",
    )
    patients = sorted(set(subject_id))
    pq.write_table(
        pa.table(
            {
                "subject_id": pa.array(patients, type=pa.int64()),
                "split": pa.array(["train"] * len(patients), type=pa.string()),
            }
        ),
        root / "metadata" / "subject_splits.parquet",
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "meds"
        write_synthetic_meds(root)

        dataset = MEDSDataset(root=str(root), cache_dir=str(Path(tmp) / "cache"))
        # The split makes fit() see the training patients only, so the code
        # vocabulary and quantile breaks do not depend on test patients.
        train, test = dataset.set_task(
            EthosTimeline(),
            split=PatientSplit(ratios=(0.6, 0.4), seed=0),
            num_workers=1,
        )
        print("train and test samples:", len(train), len(test))
        processor = train.input_processors["timeline"]
        print(processor)
        print("quantile breaks:", processor.quantiles)

        time, tokens = test[0]["timeline"]
        print("hours  token")
        for hours, token in zip(time.tolist(), processor.decode(tokens)):
            print(f"{hours:7.2f}  {token}")

        batch = next(iter(get_dataloader(train, batch_size=5)))
        print("batched tokens:", batch["timeline"][1].shape)

        path = str(Path(tmp) / "ethos_processor.json")
        processor.save(path)
        print("saved fitted state:", Path(path).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
