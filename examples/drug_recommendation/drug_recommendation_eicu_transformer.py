"""
Drug Recommendation on eICU with Transformer

This example demonstrates how to use the modernized eICUDataset with the
DrugRecommendationEICU task class for drug recommendation using a Transformer model.

Features:
- Uses the new BaseDataset-based eICUDataset with YAML configuration
- Uses the new DrugRecommendationEICU BaseTask class
- Demonstrates the standardized PyHealth workflow
"""

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.tasks import DrugRecommendationEICU
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load dataset
    # Replace with your eICU dataset path
    base_dataset = eICUDataset(
        root="/srv/local/data/physionet.org/files/eicu-crd/2.0",
        tables=["diagnosis", "medication", "physicalexam"],
        cache_dir="/shared/eng/pyhealth/eicu",
    )
    base_dataset.stats()

    # STEP 2: Set task using the new DrugRecommendationEICU class
    task = DrugRecommendationEICU()
    sample_dataset = base_dataset.set_task(task)

    # STEP 3: Split and create dataloaders
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.8, 0.1, 0.1]
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=32, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=32, shuffle=False)

    # STEP 4: Define model
    model = Transformer(
        dataset=sample_dataset,
        feature_keys=["conditions", "procedures", "drugs_hist"],
        label_key="drugs",
        mode="multilabel",
    )

    # STEP 5: Train
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=50,
        monitor="pr_auc_samples",
    )

    # STEP 6: Evaluate
    print(trainer.evaluate(test_dataloader))
