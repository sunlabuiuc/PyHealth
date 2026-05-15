"""Synthetic DSA example — runs in < 30 seconds on CPU.

This script is self-contained: it imports only the two files that will be
part of the PyHealth PR and uses plain PyTorch Dataset/DataLoader so it
runs on Python 3.11 without requiring the full PyHealth installation
(which needs Python ≥ 3.12 and litdata).

When run inside a complete PyHealth environment (Python ≥ 3.12,
``pip install -e .``), swap the bootstrap block for::

    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import DynamicSurvivalModel
    from pyhealth.tasks import DecompensationDSA
    from pyhealth.tasks.decompensation_dsa import make_synthetic_dsa_samples

Demonstrates:
  1. Generating a synthetic DSA dataset (no MIMIC required).
  2. Training DynamicSurvivalModel for 3 epochs.
  3. Quick ablation over hidden_dim and prediction horizon.
  4. Printing a results table.

Usage (from the pyhealth/ repo root)::

    python examples/synthetic_decompensation_dsa_model.py
"""

from __future__ import annotations

import importlib.util as _ilu
import pathlib as _pl
import sys as _sys
import types as _types
import time
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Bootstrap: load our two PR files directly without triggering the pyhealth
# package __init__ (which requires litdata / Python 3.12).
# ---------------------------------------------------------------------------
_REPO = _pl.Path(__file__).resolve().parents[1]   # pyhealth/ repo root
_PKG  = _REPO / "pyhealth"


def _stub(name: str, attrs: Dict[str, Any] = {}) -> _types.ModuleType:
    m = _sys.modules.get(name) or _types.ModuleType(name)
    m.__path__    = [str(_PKG)]
    m.__package__ = name
    for k, v in attrs.items():
        setattr(m, k, v)
    _sys.modules[name] = m
    return m


def _file(dotted: str, path: _pl.Path) -> _types.ModuleType:
    parts = dotted.split(".")
    for i in range(1, len(parts)):
        ns = ".".join(parts[:i])
        if ns not in _sys.modules:
            _stub(ns)
    spec = _ilu.spec_from_file_location(dotted, path)
    mod  = _ilu.module_from_spec(spec)
    _sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# Minimal BaseModel stub (the PR file imports this at module level)
class _BaseModel(nn.Module):
    def __init__(self, dataset: Any, **_: Any) -> None:
        super().__init__()
        self.dataset      = dataset
        self.feature_keys = list(getattr(dataset, "input_schema",  {}).keys())
        self.label_keys   = list(getattr(dataset, "output_schema", {}).keys())
        self._dummy_param = nn.Parameter(torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self._dummy_param.device


# Minimal BaseTask stub
class _BaseTask:
    task_name:     str  = ""
    input_schema:  dict = {}
    output_schema: dict = {}
    def __init__(self, code_mapping=None): pass
    def __call__(self, patient): raise NotImplementedError


_stub("pyhealth.models",           {"BaseModel": _BaseModel})
_stub("pyhealth.datasets",         {"SampleDataset": object})
_stub("pyhealth.tasks.base_task",  {"BaseTask": _BaseTask})

# Now load the two PR files
_task_mod  = _file("pyhealth.tasks.decompensation_dsa",
                   _PKG / "tasks" / "decompensation_dsa.py")
_model_mod = _file("pyhealth.models.dynamic_survival_model",
                   _PKG / "models" / "dynamic_survival_model.py")

make_synthetic_dsa_samples = _task_mod.make_synthetic_dsa_samples
DecompensationDSA          = _task_mod.DecompensationDSA
DynamicSurvivalModel       = _model_mod.DynamicSurvivalModel


# ---------------------------------------------------------------------------
# Minimal PyTorch Dataset wrapping the synthetic sample dicts
# ---------------------------------------------------------------------------

class _DSADataset(Dataset):
    """Thin wrapper so torch DataLoader can consume the sample dicts."""

    def __init__(self, samples: List[Dict[str, Any]]) -> None:
        self._s = samples

    def __len__(self) -> int:
        return len(self._s)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self._s[idx]
        return {
            "timeseries": torch.tensor(s["timeseries"], dtype=torch.float32),
            "label":      torch.tensor(s["label"],      dtype=torch.float32),
        }


class _FakeDataset:
    """Minimal stub so DynamicSurvivalModel's BaseModel.__init__ has metadata."""
    input_schema  = {"timeseries": "tensor"}
    output_schema = {"label": "binary"}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_PATIENTS  = 200
N_FEATURES  = 8
MAX_SEQ_LEN = 100
BATCH_SIZE  = 32
LR          = 1e-3
DEVICE      = "cpu"


# ---------------------------------------------------------------------------
# Helper: train one configuration
# ---------------------------------------------------------------------------

def train_model(
    hidden_dim: int = 128,
    horizon:    int = 24,
    epochs:     int = 3,
    seed:       int = 42,
) -> Dict[str, float]:
    """Train DynamicSurvivalModel on synthetic data and return metrics.

    Args:
        hidden_dim: GRU hidden state size.
        horizon: Prediction horizon in time steps.
        epochs: Number of training epochs.
        seed: Random seed.

    Returns:
        Dict with ``"final_loss"`` and ``"time_s"``.
    """
    samples = make_synthetic_dsa_samples(
        n_patients  = N_PATIENTS,
        n_features  = N_FEATURES,
        horizon     = horizon,
        max_seq_len = MAX_SEQ_LEN,
        seed        = seed,
    )
    loader = DataLoader(_DSADataset(samples), batch_size=BATCH_SIZE, shuffle=True)

    model = DynamicSurvivalModel(
        dataset       = _FakeDataset(),
        input_dim     = N_FEATURES,
        hidden_dim    = hidden_dim,
        embedding_dim = min(hidden_dim, 64),
        num_layers    = 1,
        horizon       = horizon,
        l1_reg        = 0.01,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    avg = 0.0
    t0  = time.time()

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            out  = model(**batch)
            loss = out["loss"]
            loss.backward()
            optimizer.step()
            total += loss.item()
            n     += 1
        avg = total / max(n, 1)
        print(f"  epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return {"final_loss": avg, "time_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 58)
    print("Dynamic Survival Analysis — Synthetic Example")
    print("=" * 58)

    # 1. Single run
    print("\n[1] Training  hidden_dim=128  horizon=24  epochs=3")
    r = train_model(hidden_dim=128, horizon=24, epochs=3)
    print(f"    final loss: {r['final_loss']:.4f}  |  {r['time_s']:.1f}s")

    # 2. Ablation: hidden_dim
    print("\n[2] Ablation: hidden_dim  (horizon=24, 2 epochs each)")
    print(f"    {'hidden_dim':>12}  {'loss':>8}  {'time(s)':>8}")
    print("    " + "-" * 32)
    for hdim in [64, 128, 256]:
        r = train_model(hidden_dim=hdim, horizon=24, epochs=2, seed=0)
        print(f"    {hdim:>12}  {r['final_loss']:>8.4f}  {r['time_s']:>8.1f}")

    # 3. Ablation: horizon
    print("\n[3] Ablation: horizon  (hidden_dim=128, 2 epochs each)")
    print(f"    {'horizon':>10}  {'loss':>8}  {'time(s)':>8}")
    print("    " + "-" * 30)
    for h in [6, 12, 24]:
        r = train_model(hidden_dim=128, horizon=h, epochs=2, seed=1)
        print(f"    {h:>10}  {r['final_loss']:>8.4f}  {r['time_s']:>8.1f}")

    print("\n" + "=" * 58)
    print("Done.")
    print("=" * 58)


if __name__ == "__main__":
    main()
