"""Fit MAS-4 on Mayo PBC and report enrollment concordance.

Run with `python examples/pbcseq_liver_survival_mas.py --root /path/to/pbcseq-directory`.
Pass `--log-labs` to evaluate log-transformed labs on the same patient splits.
"""

import argparse

import pandas as pd
import torch

from pyhealth.datasets import PatientSplit, PBCSeqDataset, get_dataloader
from pyhealth.metrics import concordance_index
from pyhealth.models import MAS
from pyhealth.tasks import LongitudinalLiverSurvival
from pyhealth.trainer import Trainer


def enrollment_concordance(model, dataset):
    """Score each patient using first-interval risk and terminal follow-up."""
    loader = get_dataloader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))
    with torch.inference_mode():
        risk = model.linear(batch["features"].to(model.device)).flatten().cpu().numpy()
    intervals = pd.DataFrame(
        batch["survival_interval"].numpy(), columns=["start", "stop", "event"]
    )
    intervals["patient_id"] = batch["patient_id"]
    intervals["risk"] = risk
    intervals = intervals.sort_values("start")
    first = intervals.drop_duplicates("patient_id").set_index("patient_id")
    last = intervals.drop_duplicates("patient_id", keep="last").set_index("patient_id")
    return concordance_index(
        duration=last["stop"], event=last["event"], risk=first.loc[last.index, "risk"]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", required=True, help="Directory containing pbcseq.csv.")
    parser.add_argument("--cache-dir", help="Directory for the processed dataset cache.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--log-labs", action="store_true")
    args = parser.parse_args()

    dataset = PBCSeqDataset(root=args.root, cache_dir=args.cache_dir)
    task = LongitudinalLiverSurvival(
        mode="counting_process", feature_set="mas4", log_transform=args.log_labs
    )
    results = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        train, val, test = dataset.set_task(
            task, split=PatientSplit(ratios=(0.7, 0.15, 0.15), seed=seed)
        )
        train_loader = get_dataloader(train, batch_size=len(train), shuffle=False)
        model = MAS(train)
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(
            train_dataloader=train_loader,
            epochs=args.epochs,
            optimizer_params={"lr": args.lr},
        )
        coefficients = model.linear.weight.detach().flatten().tolist()
        results.append(
            {
                "seed": seed,
                "val_c_index": enrollment_concordance(model, val),
                "test_c_index": enrollment_concordance(model, test),
                **dict(zip(task.feature_names, coefficients)),
            }
        )

    results = pd.DataFrame(results).set_index("seed")
    print(results.round(4).to_string())
    print("\nMean across patient splits")
    print(results.mean().round(4).to_string())


if __name__ == "__main__":
    main()
