"""Length-of-stay prediction on MIMIC-IV with an RNN.

Loads diagnoses, procedures and prescriptions with ``MIMIC4EHRDataset``,
applies ``LengthOfStayPredictionMIMIC4`` (10-class LOS category per
admission), and trains an ``RNN`` for 50 epochs, reporting accuracy and F1.

Data: MIMIC-IV 2.2 from PhysioNet (https://physionet.org/content/mimiciv/,
credentialed). Set ``MIMIC4_ROOT`` below to the folder that contains
``hosp/``. ``dev=True`` limits the run to a subset of patients; set it to
``False`` for the full dataset.

Run: ``python examples/length_of_stay/length_of_stay_mimic4_rnn.py``.
Without credentials, set ``MIMIC4_ROOT`` to ``test-resources/core/mimic4demo``
and ``dev=False`` (very few samples; for checking that the code runs).
"""

from pyhealth.datasets import MIMIC4EHRDataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import LengthOfStayPredictionMIMIC4
from pyhealth.trainer import Trainer

MIMIC4_ROOT = "/path/to/mimiciv/2.2"

if __name__ == "__main__":
    # STEP 1: load data
    base_dataset = MIMIC4EHRDataset(
        root=MIMIC4_ROOT,
        tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: set task
    sample_dataset = base_dataset.set_task(LengthOfStayPredictionMIMIC4())

    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 3: define model
    model = RNN(
        dataset=sample_dataset,
    )

    # STEP 4: define trainer
    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_weighted", "f1_macro", "f1_micro"],
    )
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=50,
        monitor="accuracy",
    )

    # STEP 5: evaluate
    results = trainer.evaluate(test_dataloader)
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
