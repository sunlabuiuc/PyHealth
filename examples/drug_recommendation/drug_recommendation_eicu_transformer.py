"""Drug recommendation on eICU with a Transformer.

Loads the eICU-CRD demo (diagnosis, medication and physical exam tables),
applies ``DrugRecommendationEICU`` (multilabel: the drugs given during each
ICU stay, from the stay history), and trains/evaluates a ``Transformer`` for
one epoch.

Data: by default ``root`` points at a public copy of the eICU-CRD demo
(downloaded on the fly, no credentials). For the full eICU-CRD
(https://physionet.org/content/eicu-crd/, credentialed) set ``root`` to your
local folder and set ``dev=False``.

Run: ``python examples/drug_recommendation/drug_recommendation_eicu_transformer.py``
(CPU). Offline: set ``root`` to ``test-resources/core/eicudemo``.
"""

import tempfile

from pyhealth.datasets import eICUDataset
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.models import Transformer
from pyhealth.tasks import DrugRecommendationEICU
from pyhealth.trainer import Trainer


if __name__ == "__main__":
    # STEP 1: Load dataset
    base_dataset = eICUDataset(
        root="https://storage.googleapis.com/pyhealth/eicu-demo/",
        tables=["diagnosis", "medication", "physicalexam"],
        cache_dir=tempfile.TemporaryDirectory().name,
        dev=True,
    )
    base_dataset.stats()

    # STEP 2: Set task
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
    )

    # STEP 5: Train
    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=1,
        monitor="pr_auc_samples",
    )

    # STEP 6: Evaluate
    print(trainer.evaluate(test_dataloader))
