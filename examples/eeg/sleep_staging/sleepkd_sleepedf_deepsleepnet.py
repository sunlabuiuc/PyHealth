"""SleepKD on Sleep-EDF-20 with DeepSleepNet teachers and students.

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

Settings follow the paper: 20-epoch sequences of Fpz-Cz at 100 Hz, AASM
stages, Adam with learning rate 1e-5, batch size 20, 200 epochs, and loss
weights alpha=1.0, beta=0.1, gamma=delta=1.0. The 8:1:1 split is by subject
(the paper does not say), with ``--seed``. The checkpoint kept is the one
with the best validation macro-F1.

Data: Sleep-EDF Database Expanded v1.0.0, open access
(https://physionet.org/content/sleep-edfx/1.0.0/). ``--root`` is the folder
with ``SC-subjects.xls`` and ``sleep-cassette/``; Sleep-EDF-20 is its 39
nights of subjects 0-19 (SC40xx-SC41xx), and only those should be present.

Run: ``python examples/eeg/sleep_staging/sleepkd_sleepedf_deepsleepnet.py
--root /path/to/sleep-edfx --run teacher``. On an Apple M4 Pro (``--device
mps``), the teacher takes about 30 minutes and a student a few minutes.
"""

import argparse
import json
from pathlib import Path

import torch

from pyhealth.datasets import SleepEDFDataset, get_dataloader, split_by_patient
from pyhealth.models import DeepSleepNet, SleepKD
from pyhealth.tasks import SleepStagingSequenceSleepEDF
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


def deepsleepnet(dataset, size):
    n_conv, hidden_dim = SIZES[size]
    return DeepSleepNet(dataset=dataset, n_conv=n_conv, hidden_dim=hidden_dim)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", required=True)
    parser.add_argument("--run", required=True, choices=RUNS)
    parser.add_argument("--out", default="sleepkd_runs")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dataset = SleepEDFDataset(root=args.root, subset="cassette")
    samples = dataset.set_task(SleepStagingSequenceSleepEDF(seq_len=20))
    train, val, test = split_by_patient(samples, [0.8, 0.1, 0.1], seed=args.seed)
    print(f"windows: train {len(train)}, val {len(val)}, test {len(test)}")

    torch.manual_seed(args.seed)
    size, source, weights = RUNS[args.run]
    model = deepsleepnet(samples, size)
    trained = model
    if source is not None:
        teacher = deepsleepnet(samples, RUNS[source][0])
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
    result = {"run": args.run, "seed": args.seed, "epochs": args.epochs, **scores}
    (out / f"{args.run}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
