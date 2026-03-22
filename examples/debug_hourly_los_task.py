 from pyhealth.datasets import eICUDataset
from pyhealth.tasks.hourly_los import hourly_los_prediction_fn

root = "/home/medukonis/Documents/Illinois/Spring_2026/CS598_Deep_Learning_For_Healthcare/Project/Datasets/eicu-collaborative-research-database-2.0"

dataset = eICUDataset(
    root=root,
    tables=["patient", "lab", "vitalPeriodic", "vitalAperiodic", "nurseCharting"],
    dev=True,
)

task_dataset = dataset.set_task(
    lambda patient: hourly_los_prediction_fn(
        patient=patient,
        time_series_tables=["lab", "vitalPeriodic"],
        time_series_features={
            "lab": ["glucose", "sodium", "potassium"],
            "vitalPeriodic": ["heartrate", "respiration", "temperature"],
        },
        static_features=[],
        min_history_hours=5,
        max_hours=48,
    )
)

print(task_dataset[0])
print("Number of samples:", len(task_dataset))
