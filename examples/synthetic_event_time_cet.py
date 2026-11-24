"""CET model ablation for synthetic and optional PyHealth SUPPORT2 data.

Paper: https://proceedings.mlr.press/v126/engelhard20a.html

Run from the PyHealth repository root:
    python examples/synthetic_event_time_cet.py --epochs 8 --output cet_ablation.json
    python examples/synthetic_event_time_cet.py --source support2 \
        --data /path/to/support2.csv --epochs 8

The synthetic benchmark preserves true eventual occurrence for evaluation and
simulates independent right censoring. The optional SUPPORT2 branch uses the
existing PyHealth Support2Dataset through survival_mdn_data.py. It reports
held-out censored likelihood, *not* eventual-occurrence AUC, because the true
latent eventual-occurrence label is unavailable in real observational data.

Ablations compare binary Concrete sample count and temperature, holding data,
seeds and network widths constant. All datasets are locally generated or read
from a user-supplied file; no protected data are downloaded.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from pyhealth.datasets import create_sample_dataset
from pyhealth.models import CET


@dataclass
class Split:
    """Dense data and hidden evaluation labels for a disjoint subject split."""

    x: np.ndarray
    t: np.ndarray
    s: np.ndarray
    c_true: np.ndarray | None

    def dataset(self):
        """Return a PyHealth SampleDataset with tensor-based event-time labels."""
        examples = [
            {
                "patient_id": f"synthetic_{i}",
                "features": x.tolist(),
                "duration": t.tolist(),
                "event": e.tolist(),
            }
            for i, (x, t, e) in enumerate(zip(self.x, self.t, self.s))
        ]
        return create_sample_dataset(
            samples=examples,
            input_schema={"features": "tensor"},
            output_schema={"duration": "tensor", "event": "tensor"},
            dataset_name="cet_ablation",
            in_memory=True,
        )


@dataclass
class Bundle:
    """Train/validation/test cohorts with no subject overlap."""

    train: Split
    valid: Split
    test: Split


def make_synthetic(n: int, seed: int) -> Bundle:
    """Generate a multi-event toy survival benchmark with latent cure status.

    Args:
        n: Number of synthetic independent subjects.
        seed: Random generator seed.
    Returns:
        Three disjoint datasets with hidden true occurrence for scoring.
    """
    if n < 60:
        raise ValueError("Use at least 60 synthetic subjects")
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6)).astype(np.float32)
    logits = np.stack((0.8 * x[:, 0] - 0.6 * x[:, 1],
                       -0.2 + 0.7 * x[:, 2] + 0.3 * x[:, 3]), axis=1)
    p = 1 / (1 + np.exp(-logits))
    c = (rng.uniform(size=p.shape) < p).astype(np.float32)
    log_time = np.stack((0.6 * x[:, 1] - 0.2 * x[:, 4],
                         0.5 + 0.5 * x[:, 3] + 0.1 * x[:, 5]), axis=1)
    event_time = np.exp(log_time + 0.4 * rng.standard_normal(p.shape))
    cutoff = rng.uniform(0.2, 2.5, size=p.shape)
    s = (c == 1) & (event_time <= cutoff)
    t = np.where(s, event_time, cutoff).astype(np.float32)
    idx = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)

    def subset(indices: np.ndarray) -> Split:
        return Split(x[indices], t[indices], s[indices].astype(np.float32),
                     c[indices])

    return Bundle(subset(idx[:a]), subset(idx[a:b]), subset(idx[b:]))


def make_support2(csv_path: str, seed: int) -> Bundle:
    """Load existing PyHealth SUPPORT2 through its established example helper.

    Note:
        SUPPORT2 has right-censored mortality, not the MIMIC laboratory event
        benchmark. Therefore this demonstrates general usability, not exact
        replication of the original CET paper's outcome definitions.
    """
    # This helper belongs to PyHealth's existing SurvivalMDN example and
    # invokes Support2Dataset before creating train-only-scaled arrays.
    from survival_mdn_data import read_support2, split_frame

    # The public SUPPORT2 CSV does not necessarily have an explicit patient ID.
    # The existing SurvivalMDN helper expects one, so synthesize a temporary
    # row ID without modifying the source CSV or the helper maintained by
    # another PyHealth contribution.
    source_path = Path(csv_path)
    raw = pd.read_csv(source_path)
    if not {"wblc", "crea"}.issubset(raw.columns):
        raise ValueError("SUPPORT2 input must contain 'wblc' and 'crea'")
    missing_lab_count = int(raw[["wblc", "crea"]].isna().any(axis=1).sum())

    if "sno" not in raw.columns and "patient_id" not in raw.columns:
        raw.insert(0, "sno", np.arange(1, len(raw) + 1))
        with TemporaryDirectory(prefix="cet_support2_input_") as tmp:
            staged_path = Path(tmp) / "support2.csv"
            raw.to_csv(staged_path, index=False)
            encoded = read_support2(staged_path)
    else:
        encoded = read_support2(source_path)

    print(
        "SUPPORT2 cohort audit: "
        f"raw_rows={len(raw)}, "
        f"rows_missing_wblc_or_crea={missing_lab_count}, "
        f"encoded_rows={len(encoded)}"
    )
    if len(encoded) != len(raw) - missing_lab_count:
        print(
            "NOTE: The parsed cohort differs from the simple wblc/crea "
            "missingness filter; inspect the dataset parser for other exclusions."
        )
    source = split_frame(encoded, None, split=1, seed=seed)

    def convert(part) -> Split:
        x = np.asarray(part.x, dtype=np.float32)
        t = np.asarray(part.duration, dtype=np.float32).reshape(-1, 1)
        s = np.asarray(part.event, dtype=np.float32).reshape(-1, 1)
        return Split(x, np.maximum(t, 1e-4), s, None)

    return Bundle(convert(source.train), convert(source.valid), convert(source.test))


def tensors(part: Split, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Convert one split to model tensors."""
    return tuple(torch.as_tensor(a, dtype=torch.float32, device=device)
                 for a in (part.x, part.t, part.s))


@torch.no_grad()
def mean_nll(model: CET, data: Split, seed: int, device: torch.device) -> float:
    """Compute right-censored MC loss with a fixed evaluation seed."""
    model.eval()
    x, t, s = tensors(data, device)
    losses = []
    for idx, start in enumerate(range(0, len(x), 256)):
        # Fork RNG state to avoid changing training randomness.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + idx)
            out = model(features=x[start:start + 256],
                        duration=t[start:start + 256], event=s[start:start + 256])
        losses.append((out["loss"].item(), len(x[start:start + 256])))
    return float(sum(v * n for v, n in losses) / sum(n for _, n in losses))


@torch.no_grad()
def occurrence_metrics(model: CET, part: Split,
                       device: torch.device) -> dict | None:
    """Score true eventual-event status without using it for training.

    The synthetic benchmark retains c_true for evaluation. In a real
    right-censored dataset these true latent outcomes are unavailable.
    """
    if part.c_true is None:
        return None
    model.eval()
    x = torch.as_tensor(part.x, dtype=torch.float32, device=device)
    probs = torch.sigmoid(model.occurrence_logits(x)).cpu().numpy()
    aucs, auprcs = [], []
    for j in range(probs.shape[1]):
        labels = part.c_true[:, j]
        aucs.append(float(roc_auc_score(labels, probs[:, j]))
                    if len(np.unique(labels)) == 2 else None)
        auprcs.append(float(average_precision_score(labels, probs[:, j]))
                      if np.any(labels == 1) else None)
    valid = [score for score in aucs if score is not None]
    return {
        "auc": aucs,
        "auprc": auprcs,
        "mean_auc": float(np.mean(valid)) if valid else None,
        "true_positives": part.c_true.sum(axis=0).astype(int).tolist(),
        "observed_positives": part.s.sum(axis=0).astype(int).tolist(),
    }


def run_arm(bundle: Bundle, samples: int, temperature: float,
            args: argparse.Namespace, device: torch.device) -> dict:
    """Fit one matched arm and return held-out metrics."""
    torch.manual_seed(args.seed)
    model = CET(
        bundle.train.dataset(),
        hidden_dim=args.hidden_dim,
        n_samples=samples,
        temperature=temperature,
        dropout=0.2,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    x, t, s = tensors(bundle.train, device)
    rng = np.random.default_rng(args.seed + 10)
    best, best_epoch = float("inf"), -1
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        for indices in np.array_split(rng.permutation(len(x)),
                                      max(1, int(np.ceil(len(x) / args.batch_size)))):
            ids = torch.as_tensor(indices, device=device)
            optim.zero_grad(set_to_none=True)
            loss = model(features=x[ids], duration=t[ids], event=s[ids])["loss"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite CET training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optim.step()
        val = mean_nll(model, bundle.valid, args.seed + 100, device)
        if val < best:
            best, best_epoch = val, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("No valid CET checkpoint")
    model.load_state_dict(best_state)
    row = {
        "n_samples": samples,
        "temperature": temperature,
        "best_epoch": best_epoch,
        "valid_nll": best,
        "test_nll": mean_nll(model, bundle.test, args.seed + 200, device),
    }
    # Compare performance for the same fitted checkpoint on each split.
    for split_name, part in (("train", bundle.train),
                             ("valid", bundle.valid),
                             ("test", bundle.test)):
        metrics = occurrence_metrics(model, part, device)
        if metrics is None:
            continue  # No hidden eventual outcomes available for SUPPORT2.
        row[f"{split_name}_eventual_auc"] = metrics["auc"]
        row[f"{split_name}_eventual_auprc"] = metrics["auprc"]
        row[f"{split_name}_mean_eventual_auc"] = metrics["mean_auc"]
        row[f"{split_name}_true_positives"] = metrics["true_positives"]
        row[f"{split_name}_observed_positives"] = metrics["observed_positives"]
    # Preserve the previous output keys for downstream summaries.
    if "test_eventual_auc" in row:
        row["eventual_auc"] = row["test_eventual_auc"]
        row["eventual_auprc"] = row["test_eventual_auprc"]
        row["mean_eventual_auc"] = row["test_mean_eventual_auc"]
    return row


def main() -> None:
    """Run paired model ablations and optionally write machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("synthetic", "support2"),
                        default="synthetic")
    parser.add_argument("--data", help="Path to local SUPPORT2 CSV")
    parser.add_argument("--n", type=int, default=360)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    if args.source == "support2" and not args.data:
        parser.error("--data is required for --source support2")
    bundle = (make_synthetic(args.n, args.seed) if args.source == "synthetic"
              else make_support2(args.data, args.seed))
    device = torch.device(args.device)
    configs = [(8, 0.3), (32, 0.3), (32, 1.0)]
    rows = [run_arm(bundle, count, temp, args, device)
            for count, temp in configs]
    report = {"source": args.source, "seed": args.seed,
              "n_train": len(bundle.train.x), "n_valid": len(bundle.valid.x),
              "n_test": len(bundle.test.x), "ablation": rows}
    print(json.dumps(report, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
