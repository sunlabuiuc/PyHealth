"""SleepKD on Sleep-EDF-20 and ISRUC-III with DeepSleepNet teachers and students.

Reproduces the DeepSleepNet experiments of SleepKD (Liang et al., IJCAI 2023,
Table 3): a DeepSleepNet teacher (4 convolutions per branch, 512 LSTM units)
is distilled into a student (1, 128) through a teacher assistant (2, 256).

Each ``--run`` trains one model and writes ``<run>.pt`` (its weights) and
``<run>.json`` (test scores) to ``--out``. Runs that distill load the models
they need from there, so train them in this order::

    teacher                       DeepSleepNet (4, 512), cross-entropy
    student                       DeepSleepNet (1, 128) from scratch
    kd         teacher -> student      KD: soft + hard labels
    fitnets    teacher -> student      FitNets: epoch-feature hint + KD
    sleepkd_noTA teacher -> student    SleepKD loss, no assistant
    takd_ta    teacher -> TA           TAKD, first step (KD)
    takd       takd_ta -> student      TAKD, second step (KD)
    sleepkd_ta teacher -> TA           SleepKD, first step
    sleepkd    sleepkd_ta -> student   SleepKD, second step

The models are the SleepKD paper's variant of DeepSleepNet (``--layout
sleepkd``, one LSTM layer), whose sizes match its Table 5; ``--layout
supratak`` uses the original DeepSleepNet (two LSTM layers) instead.

Settings follow the paper: 20-epoch sequences of one EEG channel at 100 Hz
(Fpz-Cz on Sleep-EDF-20, F3-A2 resampled from 200 Hz on ISRUC-III), AASM
stages, Adam with learning rate 1e-5, batch size 20, 200 epochs on
Sleep-EDF-20 and 300 on ISRUC-III, and loss weights alpha=1.0, beta=0.1,
gamma=delta=1.0. The 8:1:1 split is by subject (the paper does not say),
with ``--seed``. ``--fold k`` takes fold k of a subject-wise
cross-validation instead: the sorted subjects are cut into ``--folds``
groups, group k is the test set and the next group the validation set.
Give each split its own ``--out``, since the distillation runs load the
models trained on the same split. The checkpoint kept is the one with the
best validation macro-F1.

Data: Sleep-EDF Database Expanded v1.0.0, open access
(https://physionet.org/content/sleep-edfx/1.0.0/). ``--root`` is the folder
with ``SC-subjects.xls`` and ``sleep-cassette/``; Sleep-EDF-20 is its 39
nights of subjects 0-19 (SC40xx-SC41xx), and only those should be present.
ISRUC-Sleep subgroup III, open access (https://sleeptight.isr.uc.pt/), with
``--dataset isruc``: ``--root`` is the extracted ``ISRUC-Sleep-III`` folder
of its 10 subjects, see :class:`~pyhealth.datasets.ISRUCDataset`.

Run: ``python examples/eeg/sleep_staging/sleepkd_deepsleepnet.py
--root /path/to/sleep-edfx --run teacher``. On an Apple M4 Pro (``--device
mps``), each run takes 8 to 16 minutes.

Results: over seeds 0-2, accuracies are in the range of the paper's Table 3,
but macro-F1 is 2 to 7 points higher than the paper's for every method; we
have not found why. Each test set is only 2 subjects, and the spread between
splits is larger than the gaps between methods.

On ISRUC-III over the 10 subject-wise folds (``--fold 0`` to ``9``), the mean
(std) of accuracy and macro-F1 next to the paper's Table 3::

    teacher        67.7 (5.3)  62.9 (4.5)   paper 83.97
    student        66.5 (5.2)  61.5 (3.8)
    kd             64.9 (7.3)  60.0 (5.3)   paper 80.22  74.54
    fitnets        65.1 (6.3)  60.0 (5.8)   paper 81.11  75.05
    sleepkd_noTA   65.2 (6.3)  60.1 (5.3)
    takd_ta        68.4 (5.0)  62.0 (4.2)
    takd           66.1 (5.8)  60.6 (4.9)   paper 81.59  76.46
    sleepkd_ta     67.0 (6.2)  61.4 (5.5)
    sleepkd        65.0 (5.6)  59.0 (5.6)   paper 83.29  77.29

Every method is about 15 points of accuracy below the paper and no distilled
student beats the student trained alone on average. Each fold tests on one
subject so the spread between folds is again larger than the gaps between
methods. On an RTX 5050 laptop GPU each ISRUC run takes 3 to 20 minutes.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import (
    ISRUCDataset,
    SleepEDFDataset,
    get_dataloader,
    split_by_patient,
    split_by_subject,
)
from pyhealth.models import DeepSleepNet, SleepKD
from pyhealth.tasks import SleepStagingSequenceISRUC, SleepStagingSequenceSleepEDF
from pyhealth.trainer import Trainer

SIZES = {"teacher": (4, 512), "assistant": (2, 256), "student": (1, 128)}
KD = {"alpha": 0.0, "beta": 0.0, "gamma": 1.0, "delta": 1.0}
FITNETS = {"alpha": 1.0, "beta": 0.0, "gamma": 1.0, "delta": 1.0}
SLEEPKD = {"alpha": 1.0, "beta": 0.1, "gamma": 1.0, "delta": 1.0}
# run: (size, run of the model to distill from or None, loss weights)
RUNS = {
    "teacher": ("teacher", None, None),
    "student": ("student", None, None),
    "kd": ("student", "teacher", KD),
    "fitnets": ("student", "teacher", FITNETS),
    "sleepkd_noTA": ("student", "teacher", SLEEPKD),
    "takd_ta": ("assistant", "teacher", KD),
    "takd": ("student", "takd_ta", KD),
    "sleepkd_ta": ("assistant", "teacher", SLEEPKD),
    "sleepkd": ("student", "sleepkd_ta", SLEEPKD),
}
METRICS = ["accuracy", "f1_macro", "cohen_kappa"]


def deepsleepnet(dataset, size, layout):
    n_conv, hidden_dim = SIZES[size]
    return DeepSleepNet(
        dataset=dataset,
        n_conv=n_conv,
        hidden_dim=hidden_dim,
        n_rnn_layers=1 if layout == "sleepkd" else 2,
        layout=layout,
    )


def subject_fold(samples, fold, folds):
    """Train, validation and test parts of fold ``fold`` of ``folds``."""
    subjects = sorted(samples.patient_to_index, key=lambda s: (len(s), s))
    groups = np.array_split(np.array(subjects, dtype=object), folds)
    test, val = fold, (fold + 1) % folds
    train = [s for i, g in enumerate(groups) if i not in (test, val) for s in g]
    return split_by_subject(samples, [train, list(groups[val]), list(groups[test])])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--dataset", default="sleepedf", choices=["sleepedf", "isruc"])
    parser.add_argument("--out", default="sleepkd_runs")
    parser.add_argument("--layout", default="sleepkd", choices=["sleepkd", "supratak"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, help="200 on Sleep-EDF, 300 on ISRUC")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, help="subject-wise fold, 0 to --folds - 1")
    parser.add_argument("--folds", type=int, default=10)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.epochs is None:
        args.epochs = 300 if args.dataset == "isruc" else 200

    if args.dataset == "isruc":
        dataset = ISRUCDataset(root=args.root)
        samples = dataset.set_task(SleepStagingSequenceISRUC(seq_len=20))
    else:
        dataset = SleepEDFDataset(root=args.root, subset="cassette")
        samples = dataset.set_task(SleepStagingSequenceSleepEDF(seq_len=20))
    if args.fold is None:
        train, val, test = split_by_patient(samples, [0.8, 0.1, 0.1], seed=args.seed)
    else:
        train, val, test = subject_fold(samples, args.fold, args.folds)
    print(f"windows: train {len(train)}, val {len(val)}, test {len(test)}")

    torch.manual_seed(args.seed)
    size, source, weights = RUNS[args.run]
    model = deepsleepnet(samples, size, args.layout)
    trained = model
    if source is not None:
        teacher = deepsleepnet(samples, RUNS[source][0], args.layout)
        state = torch.load(out / f"{source}.pt", map_location="cpu", weights_only=True)
        teacher.load_state_dict(state)
        trained = SleepKD(dataset=samples, teacher=teacher, student=model, **weights)

    trainer = Trainer(
        model=trained,
        metrics=METRICS,
        device=args.device,
        output_path=str(out / "logs"),
        exp_name=args.run,
    )
    trainer.train(
        train_dataloader=get_dataloader(train, batch_size=20, shuffle=True),
        val_dataloader=get_dataloader(val, batch_size=20, shuffle=False),
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="f1_macro",
    )
    scores = trainer.evaluate(get_dataloader(test, batch_size=20, shuffle=False))
    print(f"{args.run} test: {scores}")

    torch.save(model.state_dict(), out / f"{args.run}.pt")
    result = {
        "run": args.run,
        "dataset": args.dataset,
        "layout": args.layout,
        "seed": args.seed,
        "fold": args.fold,
        "epochs": args.epochs,
        **scores,
    }
    (out / f"{args.run}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
