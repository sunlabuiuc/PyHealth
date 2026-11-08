"""Fit processors on training patients only with set_task(task, split=...).

Splitting after set_task() means the code vocabularies (and any statistics a
processor learns) were fitted on every patient, test patients included.
Passing a PatientSplit to set_task fits them on the training patients only and
returns train / val / test datasets ready for a model.

Uses the small MIMIC-III demo bundled in ``test-resources/``; no download.

Usage (from the repository root):
    python examples/split_aware_set_task_mimic3demo.py
"""

import tempfile
from pathlib import Path

from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks import MortalityPredictionMIMIC3
from pyhealth.trainer import Trainer

DEMO_ROOT = (
    Path(__file__).resolve().parent.parent / "test-resources" / "core" / "mimic3demo"
)


def main():
    with tempfile.TemporaryDirectory() as cache:
        dataset = MIMIC3Dataset(
            root=str(DEMO_ROOT),
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            cache_dir=cache,
        )
        train, val, test = dataset.set_task(
            MortalityPredictionMIMIC3(),
            split=PatientSplit(ratios=(0.6, 0.2, 0.2), seed=0),
        )
        # Ratios split patients, not samples: patients have different numbers
        # of admissions, so sample counts need not follow the ratios.
        print("patients (train, val, test):",
              *(len(part.patient_to_index) for part in (train, val, test)))
        print("samples  (train, val, test):", len(train), len(val), len(test))
        print("processors fitted on:", train.fit_split)

        vocab = train.input_processors["conditions"].code_vocab
        test_codes = {c for i in range(len(test)) for c in test[i]["conditions"].tolist()}
        unknown = vocab["<unk>"]
        print(f"training vocabulary: {len(vocab)} codes; "
              f"test samples use <unk>: {unknown in test_codes}")

        model = RNN(dataset=train)
        trainer = Trainer(model=model, metrics=["roc_auc", "pr_auc"], enable_logging=False)
        trainer.train(
            get_dataloader(train, batch_size=16, shuffle=True),
            val_dataloader=get_dataloader(val, batch_size=16),
            epochs=1,
        )
        print("test:", trainer.evaluate(get_dataloader(test, batch_size=16)))


if __name__ == "__main__":
    main()
