"""
MedLingo jargon expansion with :class:`~pyhealth.models.TransformersModel`.

**Paper:** Jia, Sontag & Agrawal — *Diagnosing our datasets* (CHIL 2025),
https://arxiv.org/abs/2505.15024. Public CSV: ``questions.csv`` (columns
``word1``, ``word2``, ``question``, ``answer``) from the MedLingo export in
Flora-jia-jfr/diagnosing_our_datasets — place that file under the directory you
pass as ``root`` below.

**Ablation (two task configs):**

- ``MedLingoJargonExpansionTask(shot_mode="one_shot")`` — ``prompt`` is the
  released ``question`` string (matches the distributed MedLingo item).
- ``MedLingoJargonExpansionTask(shot_mode="zero_shot")`` — ``prompt`` is rebuilt
  from ``word1`` and ``word2`` only; the CSV ``question`` field is not used, so
  any one-shot / ICL demo in that column is stripped by construction.

**Limitation vs the paper:** this PyHealth task uses **multiclass classification
on the string ``answer``** (via ``TransformersModel`` + Hugging Face encoders).
The paper evaluates **open-ended** generations with an LLM judge; this script
does not reproduce that protocol.

**Smoke run (no Hugging Face download):** by default this script only builds the
dataset, runs ``set_task`` for both shot modes, and prints sample counts. To
also run one forward pass with a **tiny** BERT (small one-time download unless
cached), set environment variable ``PYHEALTH_MEDLINGO_RUN_MODEL=1``::

    PYHEALTH_MEDLINGO_RUN_MODEL=1 python examples/medlingo_medlingo_jargon_expansion_transformersmodel.py

Optional: ``PYHEALTH_MEDLINGO_MODEL=<hf_model_id>`` overrides the tiny default
(``hf-internal-testing/tiny-random-bert``).

Run from the repository root after ``pip install -e .``, or set
``PYTHONPATH`` to the repo root so ``import pyhealth`` resolves.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.WARNING)
for _name in ("pyhealth", "pyhealth.datasets", "pyhealth.datasets.base_dataset"):
    logging.getLogger(_name).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _write_synthetic_questions_csv(path: Path) -> None:
    """Tiny stand-in for ``datasets/MedLingo/questions.csv`` (no secrets)."""
    rows = [
        {
            "word1": "MI",
            "word2": "STEMI",
            "question": "ICL_STUB What is MI vs STEMI in one sentence?",
            "answer": "types of heart attack",
        },
        {
            "word1": "HTN",
            "word2": "BP",
            "question": "ICL_STUB Define HTN.",
            "answer": "high blood pressure",
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    from pyhealth.datasets import MedLingoDataset, get_dataloader
    from pyhealth.tasks import MedLingoJargonExpansionTask

    tmp = Path(tempfile.mkdtemp(prefix="pyhealth_medlingo_"))
    root = tmp / "root"
    root.mkdir()
    cache = tmp / "cache"
    _write_synthetic_questions_csv(root / "questions.csv")

    base = MedLingoDataset(root=str(root), cache_dir=cache, num_workers=1)
    logger.info("Patients: %s", len(base.unique_patient_ids))

    for shot in ("one_shot", "zero_shot"):
        task = MedLingoJargonExpansionTask(shot_mode=shot)
        samples = base.set_task(task=task, num_workers=1)
        logger.info("shot_mode=%s -> %s samples", shot, len(samples))
        if len(samples):
            s0 = samples[0]
            logger.info("First keys: %s", sorted(s0.keys()))

    if os.environ.get("PYHEALTH_MEDLINGO_RUN_MODEL") != "1":
        logger.info(
            "Skipping TransformersModel forward (no download). "
            "Set PYHEALTH_MEDLINGO_RUN_MODEL=1 to run a tiny HF model on one batch."
        )
        return

    from pyhealth.models import TransformersModel

    model_name = os.environ.get(
        "PYHEALTH_MEDLINGO_MODEL", "hf-internal-testing/tiny-random-bert"
    )
    task = MedLingoJargonExpansionTask(shot_mode="one_shot")
    samples = base.set_task(task=task, num_workers=1)
    loader = get_dataloader(samples, batch_size=2, shuffle=False)
    model = TransformersModel(dataset=samples, model_name=model_name)
    model.eval()
    batch = next(iter(loader))
    import torch

    with torch.no_grad():
        out = model(**batch)
    logger.info("Forward ok; loss=%s", out.get("loss"))


if __name__ == "__main__":
    main()
