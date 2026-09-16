from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import ReadmissionPredictionMIMIC3
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.metrics import fairness_metrics_fn
from pyhealth.models import Transformer
from pyhealth.trainer import Trainer
from pyhealth.metrics.fairness_utils.utils import sensitive_attributes_from_patient_ids

# This script must be run under an `if __name__ == "__main__":` guard: the
# dataloaders below use multiprocessing workers, and without the guard each
# worker re-imports this module as if it were the main script, re-running
# everything at the top level (including reconstructing MIMIC3Dataset) --
# causing a runaway process-spawning hang instead of ever completing.
if __name__ == "__main__":
    # STEP 1: load data
    base_dataset = MIMIC3Dataset(
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
    )
    base_dataset.stats()

    # STEP 2: set task
    sample_dataset = base_dataset.set_task(ReadmissionPredictionMIMIC3(exclude_minors=False)) # Must include minors to get any readmission samples on the synthetic dataset

    train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 3: define model
    model = Transformer(dataset=sample_dataset)

    # STEP 4: define trainer
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=3,
        monitor="pr_auc",
    )

    # STEP 5: inference, return patient_ids
    y_true, y_prob, loss, patient_ids = trainer.inference(test_dataloader, return_patient_ids=True)

    # STEP 6: get sensitive attribute array from patient_ids
    sensitive_attribute_array = sensitive_attributes_from_patient_ids(base_dataset, patient_ids,
                                                                      'gender', 'F')

    # STEP 7: use pyhealth.metrics to evaluate fairness
    fairness_metrics = fairness_metrics_fn(y_true, y_prob, sensitive_attribute_array,
                                           favorable_outcome=0)
    print(fairness_metrics)