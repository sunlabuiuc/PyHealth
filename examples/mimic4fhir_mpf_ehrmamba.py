"""EHRMambaCEHR on the MIMIC-IV on FHIR demo (MPF clinical prediction).

Barebones path: FHIR NDJSON -> ``MIMIC4FHIR`` dataset ->
``MPFClinicalPredictionTask`` -> ``EHRMambaCEHR`` -> ``Trainer`` -> evaluate.

Data: the MIMIC-IV on FHIR demo, open access, no credentials
(https://physionet.org/content/mimic-iv-fhir-demo/2.1.0/). Download it and set
``DEMO_ROOT`` to its ``fhir`` folder (the ``*.ndjson.gz`` files), and
``CACHE_DIR`` to a writable folder; a second run reuses the flattened-table
cache there. The full credentialed MIMIC-IV on FHIR export works the same way.

Run: ``python examples/mimic4fhir_mpf_ehrmamba.py``. Two epochs of a small
model; a few minutes on a laptop CPU for the demo.
"""

from __future__ import annotations

from pathlib import Path

from pyhealth.datasets import MIMIC4FHIR, get_dataloader, split_by_patient
from pyhealth.models import EHRMambaCEHR
from pyhealth.tasks.mpf_clinical_prediction import MPFClinicalPredictionTask
from pyhealth.trainer import Trainer

# Set these: the demo's fhir/ folder and a writable cache folder.
DEMO_ROOT = Path("/path/to/mimic-iv-fhir-demo/2.1.0/fhir")
CACHE_DIR = Path("/path/to/pyhealth_cache/fhir-demo")


def main() -> None:
    dataset = MIMIC4FHIR(root=str(DEMO_ROOT), cache_dir=str(CACHE_DIR))
    sample_dataset = dataset.set_task(MPFClinicalPredictionTask(), num_workers=1)

    train_ds, val_ds, test_ds = split_by_patient(sample_dataset, [0.7, 0.1, 0.2])
    train_loader = get_dataloader(train_ds, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=8, shuffle=False)

    vocab_size = sample_dataset.input_processors["concept_ids"].vocab.vocab_size
    model = EHRMambaCEHR(
        dataset=sample_dataset,
        vocab_size=vocab_size,
        embedding_dim=32,
        num_layers=2,
        dropout=0.1,
    )

    trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"])
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=2,
        monitor="roc_auc",
    )
    print(trainer.evaluate(test_loader))


if __name__ == "__main__":
    main()
