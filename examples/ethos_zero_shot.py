"""Train a tiny ETHOS and estimate risks by sampling future timelines.

Smoke test on synthetic data: this writes a small MEDS directory to a temporary
folder, so it needs no credentials and no download. The numbers mean nothing
clinically, and nothing here reproduces a result from the paper. Sixty made-up
patients and a tiny model cannot, whatever the AUROCs below happen to be.

For reference, Renc et al. (arXiv:2502.06124, Table S1) report these test
AUROCs for the full model, with 100 rollouts per sample, on all of MIMIC-IV 2.2:
ICU admission 0.932 and prolonged stay 0.853. Those runs also use ETHOS's MIMIC
preprocessing (ICD and ATC mapping, the top-200 lab filter, blood pressure and
BMI rewrites), which PyHealth does not have, so vocabularies will not match the
paper's on MIMIC-IV.

To use a real MEDS dataset, such as MIMIC-IV after MEDS extraction (credentialed
access required), pass ``--root``, and raise the sizes to the paper's:

    python examples/ethos_zero_shot.py --root /path/to/meds \\
        --n-layer 6 --n-head 12 --n-embd 768 --n-positions 2048 \\
        --dropout 0.3 --lr 6e-4 --epochs 20 --rollouts 100

Usage:
    python examples/ethos_zero_shot.py
"""

import argparse
import random
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import torch
from sklearn.metrics import roc_auc_score

import pyhealth.datasets.configs as meds_configs
from pyhealth.datasets import MEDSDataset, PatientSplit, get_dataloader
from pyhealth.models import ETHOS
from pyhealth.tasks import (
    EthosOutcomeMEDS,
    EthosTimelineMEDS,
    HospitalReadmissionMEDS,
    ICUAdmissionMEDS,
    ProlongedLengthOfStayMEDS,
)
from pyhealth.trainer import Trainer

ADMIT = "HOSPITAL_ADMISSION//EW EMER.//ER"
SICK_LAB = "LAB//50813//mmol/L"
ROUTINE_LAB = "LAB//51221//%"


def write_synthetic_meds(root: Path, n_subjects: int = 60, seed: int = 0) -> None:
    """Writes synthetic stays where age and past labs foreshadow the outcome.

    Args:
        root: Output directory.
        n_subjects: Number of synthetic subjects.
        seed: Random seed.
    """
    rng = random.Random(seed)
    rows = []
    for subject in range(1, n_subjects + 1):
        now = datetime.fromisoformat("2024-01-01T08:00:00")  # MEDS times are naive
        now += timedelta(days=rng.randint(0, 60))
        birth = datetime.fromisoformat(f"{rng.randint(1940, 2000)}-01-01")
        frail = birth.year < 1970  # Stays of older patients go worse.
        rows.append((subject, None, f"GENDER//{rng.choice('FM')}", None, None))
        rows.append((subject, birth, "MEDS_BIRTH", None, None))
        for stay in range(rng.randint(1, 3)):
            hadm = subject * 100 + stay
            sick = rng.random() < (0.9 if frail else 0.1)
            length = timedelta(days=rng.uniform(11, 16) if sick else rng.uniform(1, 6))
            rows.append((subject, now, ADMIT, hadm, None))
            for hours in (1, 2, 6, 24):
                value = rng.gauss(8.0 if sick else 1.5, 0.5)
                rows.append(
                    (subject, now + timedelta(hours=hours), SICK_LAB, hadm, value)
                )
                value = rng.gauss(30.0, 3.0)
                rows.append(
                    (subject, now + timedelta(hours=hours), ROUTINE_LAB, hadm, value)
                )
            if sick:
                icu = now + timedelta(hours=30)
                rows.append((subject, icu, "ICU_ADMISSION//MICU", hadm, None))
                rows.append(
                    (subject, icu + timedelta(hours=40), "ICU_DISCHARGE//MICU", hadm, None)
                )
            discharge = now + length
            rows.append((subject, discharge, "HOSPITAL_DISCHARGE//HOME", hadm, None))
            now = discharge + timedelta(days=rng.choice([10, 20, 90]))

    (root / "data").mkdir(parents=True)
    pl.DataFrame(
        {
            "subject_id": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "time": pl.Series([r[1] for r in rows], dtype=pl.Datetime("us")),
            "code": pl.Series([r[2] for r in rows], dtype=pl.String),
            "hadm_id": pl.Series([r[3] for r in rows], dtype=pl.Int64),
            "numeric_value": pl.Series([r[4] for r in rows], dtype=pl.Float32),
        }
    ).write_parquet(root / "data" / "0.parquet")


def evaluate(dataset, model, test_patients, task, args) -> None:
    """Prints zero-shot risks for one outcome on the test patients.

    Args:
        dataset: The MEDS dataset.
        model: The trained model.
        test_patients: Patient ids held out of training.
        task: An :class:`EthosOutcomeMEDS`.
        args: Command-line arguments.
    """
    samples = dataset.set_task(
        task, num_workers=1, input_processors=model.dataset.input_processors
    )
    held_out = [
        index
        for patient, indices in samples.patient_to_index.items()
        if patient in test_patients
        for index in indices
    ]
    label_key = next(iter(task.output_schema))
    risk = model.estimate_risk(
        get_dataloader(samples.subset(held_out), batch_size=8),
        task.rule,
        num_rollouts=args.rollouts,
        max_new_tokens=args.max_new_tokens,
        label_key=label_key,
        seed=0,
    )
    scored = ~risk["y_prob"].isnan()
    y_true, y_prob = risk["y_true"][scored], risk["y_prob"][scored]
    kept = risk["num_kept"].float().mean().item()
    print(
        f"{task.task_name}: samples={len(held_out)} scored={int(scored.sum())} "
        f"mean rollouts kept={kept:.1f}/{args.rollouts} "
        f"true rate={y_true.mean().item():.3f} mean risk={y_prob.mean().item():.3f}"
    )
    if len(set(y_true.tolist())) == 2:
        auroc = roc_auc_score(y_true.numpy(), y_prob.numpy())
        print(f"  AUROC on these few samples (smoke test only): {auroc:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=None, help="MEDS root; omit for synthetic.")
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-embd", type=int, default=32)
    parser.add_argument("--n-positions", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    config = Path(meds_configs.__file__).parent / "meds_with_hadm.yaml"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if args.root is None:
            print("Using synthetic MEDS data: a smoke test, not a result.")
            root = tmp_path / "synthetic_meds"
            write_synthetic_meds(root)
        else:
            root = Path(args.root)

        dataset = MEDSDataset(
            root=str(root),
            config_path=str(config),
            cache_dir=tmp_path / "cache",
            num_workers=1,
        )
        # The split fits the vocabulary and quantile breaks on training patients.
        train, test = dataset.set_task(
            EthosTimelineMEDS(),
            split=PatientSplit(ratios=(0.7, 0.3), seed=0),
            num_workers=1,
        )
        print("training and held-out patients:", len(train), len(test))

        model = ETHOS(
            train,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_positions=args.n_positions,
            dropout=args.dropout,
        )
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(
            train_dataloader=get_dataloader(train, batch_size=8, shuffle=True),
            epochs=args.epochs,
            optimizer_class=torch.optim.AdamW,
            optimizer_params={"lr": args.lr, "betas": (0.9, 0.95)},
            weight_decay=0.1,
            max_grad_norm=1.0,
        )
        held_out = trainer.evaluate(get_dataloader(test, batch_size=8))
        print("loss on held-out timelines:", held_out["loss"])

        checkpoint = str(tmp_path / "ethos.ckpt")
        trainer.save_ckpt(checkpoint)
        trainer.load_ckpt(checkpoint)
        print("checkpoint reloaded")

        test_patients = set(test.patient_to_index)
        tasks = [
            EthosOutcomeMEDS(ICUAdmissionMEDS, death_as_positive=False),
            EthosOutcomeMEDS(ProlongedLengthOfStayMEDS),
            EthosOutcomeMEDS(HospitalReadmissionMEDS),
        ]
        for task in tasks:
            evaluate(dataset, model, test_patients, task, args)


if __name__ == "__main__":
    main()
