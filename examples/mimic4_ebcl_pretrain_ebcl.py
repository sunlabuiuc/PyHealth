"""
MIMIC-IV–style EBCL pretraining demo with **synthetic** paired sequences (no PhysioNet).

This script is the course ``examples/{dataset}_{task_name}_{model}.py`` entry for the
:class:`~pyhealth.models.EBCL` model. It mirrors the *intent* of Event-Based
Contrastive Learning (Oufattole et al., MLHC 2024): contrastive alignment of
pre- and post–index event code sequences. Full MIMIC-IV runs require credentialed
data; here we use small synthetic samples so the script runs everywhere.

**Paper:** `Event-Based Contrastive Learning for Medical Time Series
<https://arxiv.org/abs/2312.10308>`__

Ablation study (same synthetic data and seed for each row)
----------------------------------------------------------
We sweep hyperparameters and report the **mean contrastive loss** over one epoch
of minibatches (pure contrastive, ``supervised_weight=0``). Lower is not always
“better” on synthetic noise; the point is to show how each knob moves the
objective and to give a reproducible table for the PR.

+------------------+--------------+-------------+----------+----------+---------------------------+
| temperature      | hidden_dim   | emb_dim     | dropout  | batch    | mean contrastive loss     |
+==================+==============+=============+==========+==========+===========================+
| 0.05             | 32           | 32          | 0.0      | 4        | (printed at runtime)      |
| 0.10             | 32           | 32          | 0.0      | 4        |                             |
| 0.20             | 32           | 32          | 0.0      | 4        |                             |
| 0.10             | 16           | 32          | 0.0      | 4        |                             |
| 0.10             | 64           | 32          | 0.0      | 4        |                             |
| 0.10             | 32           | 32          | 0.5      | 4        |                             |
+------------------+--------------+-------------+----------+----------+---------------------------+

**Shuffled-post baseline:** For the last row we permute ``conditions_post`` within
the batch before ``forward``. This breaks correct pre/post pairing and should
**increase** contrastive loss relative to the matched setting (event-based
structure vs. random pairing).

**Findings (fill in after you run):** See console table; typically shuffled-post
loss >> matched loss for batch size > 1.

Run::

    cd /path/to/PyHealth
    python examples/mimic4_ebcl_pretrain_ebcl.py
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import EBCL


def build_synthetic_samples(
    n: int = 8,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Create paired pre/post code lists (same pool as tests; 2–8 patients)."""
    rng = random.Random(seed)
    pool = ["E11.9", "I50.9", "N18.3", "J44.1", "Z79.4", "N17.9", "I21.9"]
    samples = []
    for i in range(n):
        pre = [pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]]
        post = [pool[rng.randrange(len(pool))], pool[rng.randrange(len(pool))]]
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "conditions_pre": pre,
                "conditions_post": post,
                "label": i % 2,
            }
        )
    return samples


def _shuffle_batch_dim_post(batch: Dict[str, Any], post_key: str) -> Dict[str, Any]:
    """Break pre/post pairing by permuting the post feature along batch axis."""
    b = dict(batch)
    if post_key not in b:
        return b
    feat = b[post_key]
    if isinstance(feat, torch.Tensor):
        n = feat.size(0)
        perm = torch.randperm(n)
        b[post_key] = feat[perm]
        return b
    if isinstance(feat, tuple) and len(feat) >= 1:
        n = feat[0].size(0)
        perm = torch.randperm(n)
        parts = list(feat)
        parts[0] = parts[0][perm]
        for i in range(1, len(parts)):
            if isinstance(parts[i], torch.Tensor) and parts[i].size(0) == n:
                parts[i] = parts[i][perm]
        b[post_key] = tuple(parts)
        return b
    return b


def mean_contrastive_loss(
    dataset,
    *,
    embedding_dim: int = 32,
    hidden_dim: int = 32,
    projection_dim: int = 16,
    temperature: float = 0.1,
    dropout: float = 0.0,
    batch_size: int = 4,
    shuffle_post: bool = False,
) -> float:
    """One pass over the dataset; optional within-batch post shuffle (ablation)."""
    torch.manual_seed(0)
    model = EBCL(
        dataset,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        projection_dim=projection_dim,
        temperature=temperature,
        supervised_weight=0.0,
        dropout=dropout,
    )
    model.eval()
    loader = get_dataloader(dataset, batch_size=batch_size, shuffle=False)
    losses = []
    for batch in loader:
        b = _shuffle_batch_dim_post(batch, "conditions_post") if shuffle_post else batch
        with torch.no_grad():
            out = model(**b)
        losses.append(float(out["loss"]))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    samples = build_synthetic_samples(n=8, seed=42)
    base_ds = create_sample_dataset(
        samples=samples,
        input_schema={
            "conditions_pre": "sequence",
            "conditions_post": "sequence",
        },
        output_schema={"label": "binary"},
        dataset_name="mimic4_ebcl_synth",
    )

    # --- Ablation grid (hyperparameters) ---
    configs: List[Tuple[str, Dict[str, Any]]] = [
        ("t=0.05", {"temperature": 0.05}),
        ("t=0.10", {"temperature": 0.10}),
        ("t=0.20", {"temperature": 0.20}),
        ("hidden=16", {"hidden_dim": 16}),
        ("hidden=64", {"hidden_dim": 64}),
        ("dropout=0.5", {"dropout": 0.5}),
    ]

    print("EBCL synthetic ablation (mean contrastive loss)\n")
    rows = []
    defaults = {
        "embedding_dim": 32,
        "hidden_dim": 32,
        "projection_dim": 16,
        "temperature": 0.10,
        "dropout": 0.0,
        "batch_size": 4,
    }
    for name, overrides in configs:
        kw = {**defaults, **overrides}
        loss = mean_contrastive_loss(base_ds, **kw)
        rows.append((name, loss))
        print(f"  {name:16s}  loss={loss:.4f}")

    torch.manual_seed(123)
    matched = mean_contrastive_loss(
        base_ds,
        **defaults,
        shuffle_post=False,
    )
    torch.manual_seed(123)
    shuffled = mean_contrastive_loss(
        base_ds,
        **defaults,
        shuffle_post=True,
    )
    print("\n  Baseline (matched pre/post):     {:.4f}".format(matched))
    print("  Ablation (shuffled post codes):  {:.4f}".format(shuffled))
    if shuffled > matched + 1e-6:
        print(
            "  => Shuffled-post loss is higher: "
            "mismatched pre/post pairs hurt as expected."
        )
    elif shuffled < matched - 1e-6:
        print(
            "  => Shuffled loss lower (random batch effects on tiny data)."
        )
    else:
        print(
            "  => Loss tie: collated batch may use tensor layout where "
            "shuffle matched baseline seed; increase n or batch_size."
        )


if __name__ == "__main__":
    main()
