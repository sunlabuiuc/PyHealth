"""Train RNN_V2 on MIMIC-III decompensation, ablating the label horizon.

Authors: Sarah Husted, Allyson Moran (Georgia Tech CSE/OAN 6250 project).

Task: Yeche et al., "Dynamic Survival Analysis for Early Event Prediction",
CHIL 2024, https://proceedings.mlr.press/v248/yeche24a.html (decompensation
cohort of Harutyunyan et al. 2019). This is the per-step classifier baseline:
each hour predicts whether death follows within ``horizon`` hours.

The task needs MIMIC-III columns that PyHealth's default config does not load
(ICU ward columns, ``hadm_id`` on lab events), so this script defaults to the
packaged ``pyhealth/datasets/configs/mimic3_icu.yaml``.

Run from the repository root, for example on the demo data::

    python examples/mimic3_decompensation_rnn_v2.py --root /path/to/mimic-iii-demo \
        --horizons 12 24 48 --epochs 5

The demo has 100 patients, so scores are noisy; use it to check the pipeline.
Time-step metrics pool every labeled hour across stays.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from pyhealth.datasets import MIMIC3Dataset, PatientSplit, get_dataloader
from pyhealth.models import RNN_V2
from pyhealth.tasks import DecompensationMIMIC3
from pyhealth.trainer import Trainer
from pyhealth.utils import set_seed

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "pyhealth" / "datasets" / "configs" / "mimic3_icu.yaml"
)
METRICS = ["roc_auc", "pr_auc", "accuracy", "f1"]


def run_horizon(base_dataset, horizon, args, output_dir):
    """Fits processors on training patients, trains, and evaluates once."""
    set_seed(args.seed)
    task = DecompensationMIMIC3(horizon=horizon)
    train_data, val_data, test_data = base_dataset.set_task(
        task,
        split=PatientSplit(ratios=tuple(args.ratios), seed=args.seed),
        num_workers=args.num_workers,
    )
    sizes = {
        "train": len(train_data),
        "validation": len(val_data),
        "test": len(test_data),
    }
    print(f"\nhorizon {horizon:g} h: {sizes}", flush=True)
    if not all(sizes.values()):
        raise ValueError(f"Empty split {sizes}; use more patients.")
    train_loader = get_dataloader(
        train_data, batch_size=args.batch_size, shuffle=True, seed=args.seed
    )
    val_loader = get_dataloader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = get_dataloader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # One unidirectional GRU over the hourly series, so an hour only sees the
    # past. "concat" feeds the missing-value mask to the model.
    model = RNN_V2(
        dataset=train_data,
        embedding_dim={"dynamic": [{"values": args.embedding_dim}]},
        hidden_dim=args.hidden_dim,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        bidirectional=False,
        mask="concat",
    )
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device=args.device,
        output_path=str(output_dir),
        exp_name=f"horizon_{horizon:g}",
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
        max_grad_norm=5.0,
    )
    scores = {k: float(v) for k, v in trainer.evaluate(test_loader).items()}
    print(f"horizon {horizon:g} h test metrics: {scores}", flush=True)
    return {"samples": sizes, "test_metrics": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MIMIC-III directory")
    parser.add_argument(
        "--config-path", default=str(DEFAULT_CONFIG), help="Dataset config yaml"
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output/decomp"))
    parser.add_argument("--horizons", type=float, nargs="+", default=[24.0])
    parser.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--rnn-type", choices=["GRU", "LSTM", "RNN"], default="GRU")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", default=None, help="E.g. cpu or cuda:0")
    parser.add_argument("--dev", action="store_true", help="Use a few patients")
    args = parser.parse_args()
    args.device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    output_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    base_dataset = MIMIC3Dataset(
        root=args.root,
        tables=["chartevents", "labevents"],
        config_path=args.config_path,
        dev=args.dev,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )
    results = {}
    for horizon in args.horizons:
        results[f"{horizon:g}"] = run_horizon(base_dataset, horizon, args, output_dir)
        (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nLogs, checkpoints and test metrics: {output_dir}")


if __name__ == "__main__":
    main()
