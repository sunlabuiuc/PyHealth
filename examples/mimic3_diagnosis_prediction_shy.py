"""
Diagnosis Prediction with SHy on MIMIC-III.

Ablation study with different configs:
1. Number of temporal phenotypes (K=1, 3, 5)
2. Number of HGNN layers (0, 1, 2)
3. Loss components (w/ and w/o each auxiliary loss)
4. Gumbel-Softmax temperature (0.5, 1.0, 2.0) — novel extension

Paper: Leisheng Yu, Yanxiao Cai, Minxing Zhang, and Xia Hu.
    Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction.
    Proceedings of Machine Learning Research (CHIL), 2025.

Results (MIMIC-III dev=True, 1000 patients, 50 epochs, lr=1e-3):

    config               jaccard      f1  pr_auc  roc_auc
    -------------------------------------------------------
    K=1                   0.0339  0.0652  0.1732   0.7240
    K=3                   0.0401  0.0762  0.1294   0.6905
    K=5                   0.0402  0.0766  0.1533   0.7126
    hgnn=0                0.0436  0.0827  0.1517   0.7067
    hgnn=1                0.0413  0.0787  0.1398   0.6997
    hgnn=2                0.0400  0.0759  0.1352   0.7142
    no auxiliary loss     0.0426  0.0808  0.1671   0.7134
    no fidelity           0.0420  0.0799  0.1422   0.6990
    no distinct           0.0390  0.0743  0.1459   0.6905
    no alpha              0.0408  0.0776  0.1429   0.6917
    full (all loss)       0.0347  0.0666  0.1389   0.6881
    temp=0.5              0.0408  0.0778  0.1265   0.7095
    temp=1.0              0.0397  0.0757  0.1354   0.6961
    temp=2.0              0.0411  0.0780  0.1431   0.6948
"""

import random
import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.datasets.splitter import split_by_patient
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models import SHy
from pyhealth.tasks import DiagnosisPredictionMIMIC3
from pyhealth.trainer import Trainer

# seed
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def run_one(sample_dataset, train_loader, val_loader, test_loader, name, **kw):
    """train + eval a single SHy config, return test metrics"""
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    model = SHy(dataset=sample_dataset, **kw)

    trainer = Trainer(
        model=model,
        metrics=["jaccard_samples", "f1_samples", "pr_auc_samples", "roc_auc_samples"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        optimizer_params={"lr": 1e-3},
        monitor="pr_auc_samples",
        monitor_criterion="max",
    )

    res = trainer.evaluate(test_loader)
    print(f"=> {res}")
    return res


if __name__ == "__main__":

    # -- load mimic-iii --
    # adjust path to local one
    base_dataset = MIMIC3Dataset(
        root="/path/to/mimic-iii/1.4",
        tables=["DIAGNOSES_ICD"],
        dev=True,
    )
    base_dataset.stats()

    # -- set up task + splits --
    task = DiagnosisPredictionMIMIC3()
    samples = base_dataset.set_task(task)
    print(f"got {len(samples)} samples total")

    train_ds, val_ds, test_ds = split_by_patient(samples, [0.8, 0.1, 0.1], seed=SEED)
    print(f"split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

    # default hyperparams (from paper)
    defaults = dict(
        embedding_dim=32,
        hgnn_dim=64,
        hgnn_layers=2,
        num_tp=5,
        hidden_dim=64,
        num_heads=8,
        dropout=0.1,
    )

    results = {}

    # -- ablation 1: vary K (number of phenotypes) --
    for k in [1, 3, 5]:
        cfg = {**defaults, "num_tp": k}
        results[f"K={k}"] = run_one(
            samples,
            train_loader,
            val_loader,
            test_loader,
            name=f"K={k}",
            **cfg,
        )

    # -- ablation 2: vary hgnn layers --
    for n in [0, 1, 2]:
        cfg = {**defaults, "hgnn_layers": n}
        results[f"hgnn={n}"] = run_one(
            samples,
            train_loader,
            val_loader,
            test_loader,
            name=f"HGNN layers={n}",
            **cfg,
        )

    # -- ablation 3: loss components --
    loss_setups = {
        "no auxiliary loss": dict(fidelity_weight=0, distinct_weight=0, alpha_weight=0),
        "no fidelity": dict(fidelity_weight=0, distinct_weight=0.01, alpha_weight=0.01),
        "no distinct": dict(fidelity_weight=0.1, distinct_weight=0, alpha_weight=0.01),
        "no alpha": dict(fidelity_weight=0.1, distinct_weight=0.01, alpha_weight=0),
        "full (all loss)": dict(
            fidelity_weight=0.1, distinct_weight=0.01, alpha_weight=0.01
        ),
    }
    for tag, loss_kw in loss_setups.items():
        cfg = {**defaults, **loss_kw}
        results[tag] = run_one(
            samples,
            train_loader,
            val_loader,
            test_loader,
            name=tag,
            **cfg,
        )

    # -- ablation 4 (extension): gumbel-softmax temperature --
    # lower temp = more discrete selections, higher = more exploration
    for temp in [0.5, 1.0, 2.0]:
        cfg = {**defaults, "temperature": temp}
        results[f"temp={temp}"] = run_one(
            samples,
            train_loader,
            val_loader,
            test_loader,
            name=f"temperature={temp}",
            **cfg,
        )

    # -- print summary table --
    print(f"\n{'='*66}")
    print("ABLATION RESULTS")
    print(f"{'='*66}")
    print(f"{'config':<20} {'jaccard':>10} {'f1':>10} {'pr_auc':>10} {'roc_auc':>10}")
    print("-" * 76)
    for tag, r in results.items():
        j = r.get("jaccard_samples", 0)
        f = r.get("f1_samples", 0)
        p = r.get("pr_auc_samples", 0)
        a = r.get("roc_auc_samples", 0)
        print(f"{tag:<20} {j:>10.4f} {f:>10.4f} {p:>10.4f} {a:>10.4f}")
    print("=" * 76)
