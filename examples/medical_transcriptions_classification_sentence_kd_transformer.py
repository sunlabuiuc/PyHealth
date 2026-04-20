"""Ablation study for ``SentenceKDTransformer`` on ``MedicalTranscriptionsDataset``.

Paper: Kim et al., "Integrating ChatGPT into Secure Hospital Networks:
A Case Study on Improving Radiology Report Analysis", CHIL 2024.
https://proceedings.mlr.press/v248/kim24a.html

This script exercises the PyHealth ``SentenceKDTransformer`` model on an
existing PyHealth dataset — ``MedicalTranscriptionsDataset`` — and runs four
ablations:

1. ``lambda`` (contrastive weight) sweep. The paper only toggles ``lambda``
   between ``0`` and ``1`` (Table 4); we extend the sweep.
2. Temperature ``tau`` sweep. Never varied in the paper.
3. Standard model hyperparameters (learning rate, dropout, batch size) —
   explicitly required by the DL4H Option-2 rubric.
4. Backbone comparison (``bert-base-uncased`` vs ``Bio_ClinicalBERT`` vs
   ``RadBERT``). The paper compares backbones only in the fixed Table 2
   setup, never in interaction with the contrastive loss.

Two run modes are supported:

- ``--quick`` (default): synthetic multiclass text samples, a tiny BERT
  backbone (``prajjwal1/bert-tiny``), and a handful of epochs so the whole
  ablation finishes in a minute on CPU / T4. This is what Colab Pro and CI
  should run, and it is also what the test suite exercises.
- ``--data_root /path/to/mtsamples``: real MedicalTranscriptionsDataset.
  Because the mtsamples CSV is not part of PyHealth demos, the script falls
  back to synthetic mode if ``--data_root`` is not a readable directory.

Example:

    # fast smoke run
    python examples/medical_transcriptions_classification_sentence_kd_transformer.py \
        --quick --epochs 2 --batch_size 8

    # full ablation on real data, writes ablations.json next to the script
    python examples/medical_transcriptions_classification_sentence_kd_transformer.py \
        --data_root /path/to/mtsamples --epochs 3

Note on the teacher step
------------------------
Kim et al. use ChatGPT as a cloud-based teacher to generate ternary sentence
labels (``normal``/``abnormal``/``uncertain``) for MIMIC-CXR reports, and the
student BERT is distilled against those pseudo-labels. This example does *not*
replicate that step for three reasons:

1. The ``mtsamples`` corpus ships with ground-truth medical-specialty labels,
   so no LLM teacher is needed to train the student on this task — the
   distillation-from-noisy-teacher framing from the paper is orthogonal here.
2. A credentialed LLM teacher step would require an OpenAI API key with paid
   budget, which adds reviewer friction and blocks "Run All" reproducibility
   for PyHealth users without the credentials.
3. For the paper's real deployment setting (credentialed MIMIC-CXR text),
   institutional DUAs generally prohibit sending report text to third-party
   cloud APIs without a BAA. A credential-free default is the safer choice
   for a general-purpose PyHealth example.

``SentenceKDTransformer`` itself is teacher-agnostic: any label source (human
annotation, a rule-based labeler, or a locally-hosted LLM such as
``Llama-3-8B``) slots in unchanged. The ablations below exercise the loss
function and the backbone choice, which are the parts of the paper this
contribution faithfully reproduces.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from pyhealth.datasets import (
    create_sample_dataset,
    get_dataloader,
    split_by_sample,
)
from pyhealth.models import SentenceKDTransformer
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
_SPECIALTIES = [
    "Cardiovascular",
    "Orthopedic",
    "Neurology",
    "Gastroenterology",
    "Radiology",
]

_TEMPLATES = {
    "Cardiovascular": [
        "Patient reports chest pain radiating to the left arm.",
        "ECG shows ST elevation consistent with infarction.",
        "History of hypertension and atrial fibrillation.",
        "Echocardiogram demonstrates reduced ejection fraction.",
    ],
    "Orthopedic": [
        "Patient complains of right knee pain after fall.",
        "MRI reveals a meniscal tear and mild joint effusion.",
        "Planned arthroscopic repair of rotator cuff.",
        "Lumbar disc herniation at L4-L5 with sciatica.",
    ],
    "Neurology": [
        "Patient presents with recurrent migraine with aura.",
        "EEG shows focal epileptiform activity in the left temporal lobe.",
        "Clinical picture consistent with multiple sclerosis.",
        "Tremor worsens with intention; cerebellar exam abnormal.",
    ],
    "Gastroenterology": [
        "Colonoscopy reveals a sigmoid polyp, biopsy taken.",
        "Patient reports epigastric pain after meals.",
        "Abdominal ultrasound shows cholelithiasis without obstruction.",
        "Endoscopy demonstrates grade B reflux esophagitis.",
    ],
    "Radiology": [
        "Chest radiograph shows a small left pleural effusion.",
        "CT abdomen pelvis unremarkable for acute pathology.",
        "MRI brain demonstrates a 6 mm white matter lesion.",
        "Mammogram shows a BIRADS 3 lesion in the right breast.",
    ],
}


def make_synthetic_samples(n_per_class: int = 40, seed: int = 0) -> List[Dict[str, Any]]:
    """Build a balanced synthetic mtsamples-style dataset.

    Args:
        n_per_class: Samples per specialty.
        seed: RNG seed.

    Returns:
        List of dicts with keys ``patient_id``, ``transcription``,
        ``medical_specialty``.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    pid = 0
    for specialty in _SPECIALTIES:
        bank = _TEMPLATES[specialty]
        for _ in range(n_per_class):
            body_lines = [rng.choice(bank) for _ in range(rng.randint(2, 4))]
            text = " ".join(body_lines)
            samples.append(
                {
                    "patient_id": f"p{pid:04d}",
                    "transcription": text,
                    "medical_specialty": specialty,
                }
            )
            pid += 1
    rng.shuffle(samples)
    return samples


def build_dataset(quick: bool, data_root: Optional[str]):
    """Build either a synthetic or a real MedicalTranscriptions dataset.

    Args:
        quick: If ``True``, always use synthetic samples.
        data_root: Optional path to the raw mtsamples directory. If the path
            does not exist or fails to load, the script falls back to
            synthetic samples with a warning.

    Returns:
        A fitted ``SampleDataset`` ready for ``get_dataloader``.
    """
    if quick or not data_root or not Path(data_root).is_dir():
        if data_root:
            print(f"[warn] data_root={data_root!r} unreadable; using synthetic data")
        samples = make_synthetic_samples(n_per_class=40)
        return create_sample_dataset(
            samples=samples,
            input_schema={"transcription": "text"},
            output_schema={"medical_specialty": "multiclass"},
            dataset_name="medical_transcriptions_synth",
        )

    from pyhealth.datasets import MedicalTranscriptionsDataset

    raw = MedicalTranscriptionsDataset(root=data_root)
    return raw.set_task()


# ---------------------------------------------------------------------------
# Experiment plumbing
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """One cell in the ablation grid."""

    name: str
    model_name: str = "prajjwal1/bert-tiny"
    lam: float = 1.0
    temperature: float = 0.07
    dropout: float = 0.1
    lr: float = 3e-5
    batch_size: int = 16
    epochs: int = 2
    max_length: int = 128


@dataclass
class RunResult:
    """Test-split metrics for a single ablation cell."""

    name: str
    accuracy: float
    macro_f1: float
    loss: float
    config: Dict[str, Any] = field(default_factory=dict)


def _train_and_eval(
    cfg: RunConfig, dataset, seed: int = 0
) -> RunResult:
    """Train a :class:`SentenceKDTransformer` and return test metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    model = SentenceKDTransformer(
        dataset=dataset,
        model_name=cfg.model_name,
        dropout=cfg.dropout,
        lam=cfg.lam,
        temperature=cfg.temperature,
        max_length=cfg.max_length,
    )

    train_ds, val_ds, test_ds = split_by_sample(
        dataset, ratios=[0.6, 0.2, 0.2], seed=seed
    )
    train_loader = get_dataloader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    trainer = Trainer(
        model=model,
        metrics=["accuracy", "f1_macro"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=cfg.epochs,
        optimizer_params={"lr": cfg.lr},
        monitor="accuracy",
        monitor_criterion="max",
    )
    scores = trainer.evaluate(test_loader)

    return RunResult(
        name=cfg.name,
        accuracy=float(scores.get("accuracy", float("nan"))),
        macro_f1=float(scores.get("f1_macro", float("nan"))),
        loss=float(scores.get("loss", float("nan"))),
        config=asdict(cfg),
    )


def _print_table(title: str, rows: List[RunResult]) -> None:
    print()
    print(f"## {title}")
    print()
    print("| name | accuracy | macro-f1 | loss |")
    print("|------|---------:|---------:|-----:|")
    for r in rows:
        print(f"| {r.name} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {r.loss:.3f} |")


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------
def run_lambda_sweep(
    dataset, *, backbone: str, epochs: int, batch_size: int
) -> List[RunResult]:
    """Ablation 1: contrastive-weight ``lambda`` sweep.

    The paper only toggles ``lambda ∈ {0, 1}`` (Table 4). We sweep a wider
    range, including overshoots, to locate the phase transition where the
    contrastive term starts dominating cross-entropy.
    """
    grid = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    results: List[RunResult] = []
    for lam in grid:
        cfg = RunConfig(
            name=f"lam={lam}",
            model_name=backbone,
            lam=lam,
            epochs=epochs,
            batch_size=batch_size,
        )
        results.append(_train_and_eval(cfg, dataset))
    return results


def run_temperature_sweep(
    dataset, *, backbone: str, epochs: int, batch_size: int
) -> List[RunResult]:
    """Ablation 2: contrastive temperature ``tau`` sweep.

    The paper uses a single ``tau`` (0.07, the Khosla et al. default). We
    sweep a range spanning the "peaked" and "uniform" regimes of the
    contrastive softmax to quantify how sensitive the student is to this
    hyperparameter when ``lambda`` is held at 1.
    """
    grid = [0.05, 0.1, 0.2, 0.5, 1.0]
    results: List[RunResult] = []
    for tau in grid:
        cfg = RunConfig(
            name=f"tau={tau}",
            model_name=backbone,
            temperature=tau,
            epochs=epochs,
            batch_size=batch_size,
        )
        results.append(_train_and_eval(cfg, dataset))
    return results


def run_hyperparameter_ablation(
    dataset, *, backbone: str, epochs: int
) -> List[RunResult]:
    """Ablation 3: standard model hyperparameters (lr, dropout, batch size).

    Required by the DL4H Option-2 Model-contribution rubric.
    """
    grid: List[RunConfig] = []
    for lr in (1e-5, 3e-5, 1e-4):
        for dropout in (0.1, 0.3):
            for bs in (16, 32):
                grid.append(
                    RunConfig(
                        name=f"lr={lr}_do={dropout}_bs={bs}",
                        model_name=backbone,
                        lr=lr,
                        dropout=dropout,
                        batch_size=bs,
                        epochs=epochs,
                    )
                )
    return [_train_and_eval(cfg, dataset) for cfg in grid]


def run_backbone_comparison(
    dataset, *, quick: bool, epochs: int, batch_size: int
) -> List[RunResult]:
    """Ablation 4: backbone comparison with the contrastive term switched on.

    The paper evaluates backbones only in the fixed Table 2 setup; we pair
    each backbone with ``lambda=1`` to highlight where domain pretraining
    helps the supervised-contrastive signal.
    """
    if quick:
        backbones = ["prajjwal1/bert-tiny"]
    else:
        backbones = [
            "bert-base-uncased",
            "emilyalsentzer/Bio_ClinicalBERT",
            "StanfordAIMI/RadBERT",
        ]
    results: List[RunResult] = []
    for bb in backbones:
        cfg = RunConfig(
            name=f"backbone={bb}",
            model_name=bb,
            lam=1.0,
            epochs=epochs,
            batch_size=batch_size,
        )
        results.append(_train_and_eval(cfg, dataset))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use synthetic data + prajjwal1/bert-tiny for a CPU-friendly smoke run.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to the mtsamples raw directory. Falls back to synthetic if unreadable.",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--output",
        type=str,
        default="ablations.json",
        help="Where to write the structured results.",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        default="lambda,temperature,hyperparam,backbone",
        help="Comma-separated subset of: lambda, temperature, hyperparam, backbone.",
    )
    args = parser.parse_args(argv)

    dataset = build_dataset(quick=args.quick, data_root=args.data_root)
    backbone = "prajjwal1/bert-tiny" if args.quick else "emilyalsentzer/Bio_ClinicalBERT"
    selected = {name.strip() for name in args.ablations.split(",") if name.strip()}

    all_results: Dict[str, List[Dict[str, Any]]] = {}

    if "lambda" in selected:
        rows = run_lambda_sweep(
            dataset, backbone=backbone, epochs=args.epochs, batch_size=args.batch_size
        )
        _print_table("Ablation 1: lambda sweep (novel)", rows)
        all_results["lambda"] = [asdict(r) for r in rows]

    if "temperature" in selected:
        rows = run_temperature_sweep(
            dataset, backbone=backbone, epochs=args.epochs, batch_size=args.batch_size
        )
        _print_table("Ablation 2: temperature sweep (novel)", rows)
        all_results["temperature"] = [asdict(r) for r in rows]

    if "hyperparam" in selected:
        rows = run_hyperparameter_ablation(
            dataset, backbone=backbone, epochs=args.epochs
        )
        _print_table("Ablation 3: lr / dropout / batch size (rubric)", rows)
        all_results["hyperparam"] = [asdict(r) for r in rows]

    if "backbone" in selected:
        rows = run_backbone_comparison(
            dataset, quick=args.quick, epochs=args.epochs, batch_size=args.batch_size
        )
        _print_table("Ablation 4: backbone x contrastive (novel)", rows)
        all_results["backbone"] = [asdict(r) for r in rows]

    out_path = Path(args.output)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved structured ablation results to {out_path.resolve()}")


if __name__ == "__main__":
    main()
