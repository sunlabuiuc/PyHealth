"""How a trained model decides: importance, effect shapes and directions, in clinical terms.

:func:`attributions` measures, for each sample, how much each input column
moved the model's prediction (on the log-odds scale):

- tree models (``XGBoostModel`` and other gradient-boosted trees): exact
  TreeSHAP values;
- torch models: integrated gradients from the mean input, for numeric
  fields (tensors and time series). For a logistic regression this is the
  exact contribution of each input, its coefficient times its deviation
  from the mean.

Code fields (sequences of codes) are not attributed per column here.

:func:`summarize` turns the attributions into aggregates a clinician can
check: importance by clinical factor group, the top features with their
meaning, the shape of each one's effect in bins of at least ``min_cell``
patients, the learned direction compared with the expected one from the
task card, and how stable the top features are across fold models. Nothing
per patient leaves this module's return values.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.metrics.clinical import MIN_CELL

DIRECTION_RHO = 0.1
"""Smallest rank correlation between a value and its contribution taken as a direction."""


def attributions(
    model: torch.nn.Module,
    dataset: SampleDataset,
    max_samples: int = 2000,
    steps: int = 20,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Per-sample contributions of each numeric input column to the logit.

    Args:
        model: A trained PyHealth model.
        dataset: Samples to explain, e.g. the training split.
        max_samples: Most samples used, evenly spaced through the dataset.
        steps: Integrated-gradient steps (torch models).
        batch_size: Samples per forward pass.

    Returns:
        ``{"method", "fields": {field: {"values": (n, p), "contributions":
        (n, p)}}, "skipped": {field: reason}, "patient_ids": [...]}``. Time
        series of shape ``(time, features)`` are reduced to one column per
        feature: contributions summed over time, values averaged.

    Examples:
        >>> out = attributions(model, train_ds)  # doctest: +SKIP
        >>> out["fields"]["tabular"]["contributions"].shape  # doctest: +SKIP
        (2000, 88)
    """
    n = len(dataset)
    if n > max_samples:
        index = np.linspace(0, n - 1, max_samples).round().astype(int).tolist()
        dataset = dataset.subset(index)
    else:
        index = list(range(n))
    patient_of = {}
    for pid, idx in dataset.patient_to_index.items():
        for i in idx:
            patient_of[i] = pid
    patient_ids = [patient_of.get(i) for i in range(len(dataset))]
    loader = get_dataloader(dataset, batch_size)
    trees = hasattr(model, "estimators_") and hasattr(model, "explain")
    fields: dict[str, dict[str, list]] = {}
    skipped: dict[str, str] = {}
    model.eval()
    if not trees:
        means = _numeric_means(loader, model.feature_keys)
    for batch in loader:
        if trees:
            out = model.explain(**batch)
            contrib = {k: np.asarray(v, dtype=float) for k, v in out["attributions"].items()}
        else:
            contrib = _integrated_gradients(model, batch, means, steps)
        for key in model.feature_keys:
            value = _numeric(batch[key])
            if value is None or key not in contrib:
                skipped[key] = "codes: not attributed per column"
                continue
            c = np.asarray(contrib[key], dtype=float)
            if c.ndim > 2 and c.shape[-1] != value.shape[-1]:
                c = c[..., 0]  # one output column (binary)
            v = value.numpy().astype(float)
            if v.ndim == 3:  # (n, time, features): per feature
                c = c.reshape(v.shape).sum(axis=1)
                v = v.mean(axis=1)
            else:
                c = c.reshape(len(v), -1)
                v = v.reshape(len(v), -1)
            entry = fields.setdefault(key, {"values": [], "contributions": []})
            entry["values"].append(v)
            entry["contributions"].append(c)
    return {
        "method": "TreeSHAP (exact)" if trees else
                  f"integrated gradients from the mean input ({steps} steps)",
        "fields": {
            k: {"values": np.concatenate(v["values"]),
                "contributions": np.concatenate(v["contributions"])}
            for k, v in fields.items()
        },
        "skipped": skipped,
        "patient_ids": patient_ids,
        "sample_index": index,
    }


def summarize(
    attrib: dict[str, Any],
    card: Any = None,
    feature_names: dict[str, list[str]] | None = None,
    top_k: int = 10,
    min_cell: int = MIN_CELL,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Aggregates attributions into a clinical explanation.

    Args:
        attrib: The output of :func:`attributions`.
        card: The task's :class:`~pyhealth.tasks.TaskCard`, for column names,
            meanings, groups and expected directions.
        feature_names: Column names by field when the card has none, e.g. a
            tree model's ``feature_names``.
        top_k: Features described in detail.
        min_cell: Smallest number of distinct patients in an effect bin.
        n_bins: Most bins per effect shape.

    Returns:
        ``{"method", "n_samples", "groups": [{"group", "share"}],
        "features": [{"name", "meaning", "group", "share", "learned",
        "expected", "check", "bins": [{"label", "n", "effect"}]}],
        "skipped"}``.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> x = rng.normal(size=(200, 2))
        >>> attrib = {"method": "linear", "patient_ids": list(range(200)),
        ...           "skipped": {}, "fields": {"lab": {
        ...               "values": x, "contributions": x * [0.8, -0.1]}}}
        >>> out = summarize(attrib, feature_names={"lab": ["lactate", "sbp"]})
        >>> [(f["name"], f["learned"]) for f in out["features"]]
        [('lactate', '+'), ('sbp', '-')]
    """
    names, values, contribs, fields = [], [], [], []
    for key, entry in attrib["fields"].items():
        width = entry["values"].shape[1]
        cols = None
        if card is not None and len(card.columns.get(key, [])) == width:
            cols = card.columns[key]
        elif feature_names and len(feature_names.get(key, [])) == width:
            cols = feature_names[key]
        cols = cols or [f"{key}[{j}]" for j in range(width)]
        names += cols
        fields += [key] * width
        values.append(entry["values"])
        contribs.append(entry["contributions"])
    if not names:
        return {"method": attrib["method"], "n_samples": 0, "groups": [],
                "features": [], "skipped": attrib["skipped"]}
    values = np.concatenate(values, axis=1)
    contribs = np.concatenate(contribs, axis=1)
    pids = np.asarray([str(p) for p in attrib["patient_ids"]])
    importance = np.abs(contribs).mean(axis=0)
    total = importance.sum() or 1.0

    def info(name):
        return card.feature(name) if card is not None else None

    groups: dict[str, float] = {}
    for name, imp in zip(names, importance):
        entry = info(name)
        group = entry.group if entry is not None and entry.group else "not in the task card"
        groups[group] = groups.get(group, 0.0) + float(imp / total)
    features = []
    for j in np.argsort(-importance)[:top_k]:
        name = names[j]
        entry = info(name)
        bins = _effect_bins(values[:, j], contribs[:, j], pids, min_cell, n_bins)
        learned = _learned_direction(values[:, j], contribs[:, j], bins)
        expected = entry.direction if entry is not None else None
        features.append({
            "name": name, "field": fields[j],
            "meaning": entry.meaning if entry is not None else "",
            "group": entry.group if entry is not None else "",
            "in_card": entry is not None,
            "share": float(importance[j] / total),
            "learned": learned, "expected": expected,
            "check": _check(expected, learned),
            "bins": bins,
        })
    return {
        "method": attrib["method"], "n_samples": len(pids),
        "groups": [{"group": g, "share": s}
                   for g, s in sorted(groups.items(), key=lambda kv: -kv[1])],
        "features": features, "skipped": attrib["skipped"],
    }


def top_overlap(reference: list[str], others: list[list[str]]) -> dict[str, Any]:
    """How many of the reference top features each other ranking shares.

    Examples:
        >>> top_overlap(["a", "b", "c"], [["a", "b", "d"], ["c", "a", "b"]])
        {'k': 3, 'mean_shared': 2.5, 'min_shared': 2, 'models': 2}
    """
    shared = [len(set(reference) & set(o)) for o in others]
    return {"k": len(reference),
            "mean_shared": float(np.mean(shared)) if shared else None,
            "min_shared": int(min(shared)) if shared else None,
            "models": len(others)}


# ------------------------------------------------------------------ helpers
def _numeric(value: Any) -> torch.Tensor | None:
    if isinstance(value, (tuple, list)):
        return None
    if isinstance(value, torch.Tensor) and value.is_floating_point():
        return value.detach().cpu()
    return None


def _numeric_means(loader, keys) -> dict[str, torch.Tensor]:
    sums: dict[str, torch.Tensor] = {}
    count = 0
    for batch in loader:
        for key in keys:
            value = _numeric(batch[key])
            if value is not None:
                sums[key] = sums.get(key, 0) + value.sum(dim=0)
        count += len(next(iter(batch.values())))
    return {k: v / max(count, 1) for k, v in sums.items()}


def _integrated_gradients(model, batch, means, steps) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    keys = [k for k in model.feature_keys if k in means and _numeric(batch[k]) is not None]
    if not keys:
        return {}
    x = {k: batch[k].float().to(device) for k in keys}
    base = {k: means[k].float().to(device).expand_as(x[k]) for k in keys}
    total = {k: torch.zeros_like(x[k]) for k in keys}
    for alpha in (np.arange(steps) + 0.5) / steps:
        inputs = dict(batch)
        for k in keys:
            inputs[k] = (base[k] + float(alpha) * (x[k] - base[k])).detach()
            inputs[k].requires_grad_(True)
        logit = model(**inputs)["logit"]
        grads = torch.autograd.grad(logit[:, 0].sum(), [inputs[k] for k in keys])
        for k, g in zip(keys, grads):
            total[k] += g.detach()
    return {k: ((x[k] - base[k]) * total[k] / steps).cpu().numpy() for k in keys}


def _effect_bins(value, contrib, pids, min_cell, n_bins) -> list[dict[str, Any]]:
    """Mean contribution by value bin; every bin has min_cell distinct patients."""
    unique = np.unique(value)
    if len(unique) <= n_bins:
        chunks = [np.where(value == u)[0] for u in unique]
    else:
        # A value shared by a bin's worth of samples (e.g. 0 for "unmeasured")
        # gets its own bin; the rest is cut at quantiles. Equal values always
        # fall in the same bin.
        vals, counts = np.unique(value, return_counts=True)
        heavy = vals[counts >= len(value) / n_bins]
        rest = value[~np.isin(value, heavy)]
        k = max(1, n_bins - len(heavy))
        cuts = list(np.quantile(rest, np.linspace(0, 1, k + 1)[1:-1])) if len(rest) else []
        for h in heavy:
            cuts += [h, np.nextafter(h, np.inf)]
        cuts = np.unique(cuts)
        labels = np.searchsorted(cuts, value, side="right")
        chunks = [np.where(labels == b)[0] for b in range(len(cuts) + 1)]
        chunks = [c for c in chunks if len(c)]

    def big(idx):
        return len(set(pids[idx].tolist())) >= min_cell

    merged: list[np.ndarray] = []
    for chunk in chunks:
        if merged and not big(merged[-1]):
            merged[-1] = np.concatenate([merged[-1], chunk])
        else:
            merged.append(chunk)
    while len(merged) > 1 and not big(merged[-1]):
        last = merged.pop()
        merged[-1] = np.concatenate([merged[-1], last])
    merged = [m for m in merged if big(m)]
    out = []
    for i, idx in enumerate(merged):
        lo = None if i == 0 else _edge(value, merged[i - 1], idx)
        hi = None if i == len(merged) - 1 else _edge(value, idx, merged[i + 1])
        if len(np.unique(value[idx])) == 1:
            label = f"= {_fmt(value[idx][0])}"
        elif lo is None:
            label = f"< {_fmt(hi)}"
        elif hi is None:
            label = f"≥ {_fmt(lo)}"
        else:
            label = f"{_fmt(lo)} to < {_fmt(hi)}"
        out.append({"label": label, "n": len(set(pids[idx].tolist())),
                    "effect": float(contrib[idx].mean())})
    return out


def _edge(value, below, above) -> float:
    """A cut between two bins, at the midpoint, so no sample value is shown."""
    return float((value[below].max() + value[above].min()) / 2)


def _fmt(x: float) -> str:
    if x == 0 or not math.isfinite(x):
        return f"{x:g}"
    return f"{x:.2g}" if abs(x) < 10 else f"{x:.3g}"


def _learned_direction(value, contrib, bins) -> str:
    if np.std(value) == 0 or np.std(contrib) == 0:
        return "flat"
    rho = _spearman(value, contrib)
    effects = [b["effect"] for b in bins]
    if len(effects) >= 3:
        middle = min(effects[1:-1])
        spread = max(effects) - min(effects)
        if (effects[0] - middle > 0.25 * spread and effects[-1] - middle > 0.25 * spread
                and abs(rho) < 0.5):
            return "u"
    if rho > DIRECTION_RHO:
        return "+"
    if rho < -DIRECTION_RHO:
        return "-"
    return "flat"


def _spearman(a, b) -> float:
    rho = spearmanr(a, b).statistic
    return 0.0 if rho is None or not math.isfinite(rho) else float(rho)


def _check(expected: str | None, learned: str) -> str:
    if expected is None:
        return "not in the task card"
    if expected == "":
        return "no expectation"
    if learned == "flat":
        return "unclear"
    if expected == learned:
        return "as expected"
    if expected == "u":
        return "one-sided"
    if learned == "u":
        return "unclear"
    return "opposite"
