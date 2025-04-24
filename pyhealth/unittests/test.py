import pyhealth.datasets.mimic3 as mimic3
import pyhealth.datasets.mimic4 as mimic4
from pyhealth.tasks.mortality_prediction import test_mortality_prediction_mimic4
import pyhealth.tasks.medical_coding as coding
import time
import os
import sys

# Add the parent directory to sys.path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def time_function(func, name):
    start_time = time.time()
    func()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"{name} execution time: {execution_time:.2f} seconds")

def train_medical_coding():
    from pyhealth.datasets import MIMIC4Dataset, MIMIC3Dataset
    root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III"
    dataset = MIMIC3Dataset(
        root=root,
        dataset_name="mimic3",
        tables=[
            "DIAGNOSES_ICD",
            "PROCEDURES_ICD",
            "NOTEEVENTS"
        ],
        dev=True,
    )

    dataset.stat()

    from pyhealth.tasks.medical_coding import MIMIC3ICD9Coding
    mimic3_coding = MIMIC3ICD9Coding()
    samples = dataset.set_task(mimic3_coding)
    # Print sample information
    print(f"Total samples generated: {len(samples)}")
    if len(samples) > 0:
        print("First sample:")
        print(f"  - Text length: {len(samples[0]['text'])} characters")
        print(f"  - Number of ICD codes: {len(samples[0]['icd_codes'])}")
        if len(samples[0]['icd_codes']) > 0:
            print(f"  - Sample ICD codes: {samples[0]['icd_codes'][:5] if len(samples[0]['icd_codes']) > 5 else samples[0]['icd_codes']}")


    from pyhealth.models import TransformersModel

    model = TransformersModel(
        model_name="emilyalsentzer/Bio_ClinicalBERT",
        dataset=samples,
        feature_keys=["text"],
        label_key="icd_codes",
        mode="multilabel",
    )


    from pyhealth.datasets import split_by_sample

    train_dataset, val_dataset, test_dataset = split_by_sample(
        dataset=samples,
        ratios=[0.7, 0.1, 0.2]
    )


    from pyhealth.datasets import get_dataloader

    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    print(model)

    from pyhealth.trainer import Trainer

    # Specify the metrics we want to track
    trainer = Trainer(
        model=model,
        metrics=["f1_micro", "f1_macro", "f1_weighted"]
    )

    # Evaluate before training
    print("Metrics before training:")
    eval_results = trainer.evaluate(test_dataloader)
    print(eval_results)

    # Train with monitoring f1_micro
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="f1_micro",  # Monitor f1_micro specifically
        optimizer_params={"lr": 5e-5}  # Using learning rate of 5e-5
    )

    # Evaluate after training
    print("Metrics after training:")
    print(trainer.evaluate(test_dataloader))

if __name__ == "__main__":
    print("Starting MIMIC-III processing...")
    # time_function(mimic3.main, "MIMIC-III")
    test_mortality_prediction_mimic4()
    print("\nStarting MIMIC-IV processing...")
    time_function(mimic4.test_mimic4_dataset, "MIMIC-IV")
    print("\nStart Medical Coding Test")
    time_function(coding.main, "Medical Coding")
    time_function(train_medical_coding, "Train Medical Coding")