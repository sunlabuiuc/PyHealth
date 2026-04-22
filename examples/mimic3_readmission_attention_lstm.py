import argparse
import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, Any, List

# Ensure local pyhealth package takes precedence over any installed version
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

from pyhealth.datasets import MIMIC3Dataset, split_by_sample, get_dataloader
from pyhealth.tasks import ReadmissionPredictionMIMIC3
from pyhealth.models import RNN

from pyhealth.models import AttentionLSTM

from pathlib import Path


SEEDS = list(range(1, 1001))
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = Path("results/mimic3_attention_1000")
TOPK = 10
FAITHFULNESS_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
FAITHFULNESS_SEEDS = 10
FAITHFULNESS_RAND_REPEATS = 5


# synthetic MIMIC-III path
DATA_ROOT = str(
    Path(__file__).resolve().parent.parent /
    "examples/MIMICIII_Clinical_Database_Demo"
)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_to_device(x: Any, device: str) -> Any:
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


# compute performance metrics (ROC-AUC, PR-AUC)
def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def topk_overlap(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    k = min(k, len(a), len(b))
    if k == 0:
        return float("nan")
    top_a = set(np.argsort(a)[-k:])
    top_b = set(np.argsort(b)[-k:])
    return len(top_a & top_b) / k


def flatten_attention_dict(attn_dict: Dict[str, np.ndarray]) -> np.ndarray:
    parts = []
    for key in sorted(attn_dict.keys()):
        parts.append(attn_dict[key])
    return np.concatenate(parts, axis=1)


# compute attention instability using top-k overlap
def compute_attention_stability(
    attention_runs: List[np.ndarray], k: int = 10
) -> Dict[str, float]:
    pair_scores = []

    for i in range(len(attention_runs)):
        for j in range(i + 1, len(attention_runs)):
            a_run = attention_runs[i]
            b_run = attention_runs[j]

            if a_run.shape != b_run.shape:
                raise ValueError(
                    f"Mismatched attention shapes: {a_run.shape} vs {b_run.shape}")

            sample_scores = []
            for n in range(a_run.shape[0]):
                sample_scores.append(topk_overlap(a_run[n], b_run[n], k=k))

            pair_scores.append(np.nanmean(sample_scores))

    return {
        "pairwise_topk_overlap_mean": (
            float(np.nanmean(pair_scores)) if pair_scores else float("nan")
        ),
        "pairwise_topk_overlap_std": (
            float(np.nanstd(pair_scores)) if pair_scores else float("nan")
        ),
    }


def build_deletion_mask(
    attn_weights: Dict[str, np.ndarray],
    frac: float,
) -> Dict[str, torch.Tensor]:
    """Build a boolean deletion mask zeroing out top-attended time steps.

    Args:
        attn_weights: dict mapping feature_key -> (B, T) numpy array of
            attention weights.
        frac: fraction of time steps to delete per sample (0.0–1.0).

    Returns:
        dict mapping feature_key -> (B, T) bool tensor, True = delete this step.
    """
    masks = {}
    for key, attn in attn_weights.items():
        if isinstance(attn, torch.Tensor):
            attn = attn.numpy()
        B, T = attn.shape
        n_del = max(1, int(frac * T)) if frac > 0 else 0
        mask = np.zeros((B, T), dtype=bool)
        if n_del > 0:
            top_indices = np.argsort(attn, axis=1)[:, -n_del:]
            for b in range(B):
                mask[b, top_indices[b]] = True
        masks[key] = torch.from_numpy(mask)
    return masks


def build_random_deletion_mask(
    attn_weights: Dict[str, np.ndarray],
    frac: float,
    rng: np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """Build a boolean deletion mask zeroing out randomly selected time steps.

    Args:
        attn_weights: dict mapping feature_key -> (B, T) numpy array.
        frac: fraction of time steps to delete per sample.
        rng: numpy random generator for reproducible draws.

    Returns:
        dict mapping feature_key -> (B, T) bool tensor.
    """
    masks = {}
    for key, attn in attn_weights.items():
        if isinstance(attn, torch.Tensor):
            attn = attn.numpy()
        B, T = attn.shape
        n_del = max(1, int(frac * T)) if frac > 0 else 0
        mask = np.zeros((B, T), dtype=bool)
        if n_del > 0:
            for b in range(B):
                chosen = rng.choice(T, size=n_del, replace=False)
                mask[b, chosen] = True
        masks[key] = torch.from_numpy(mask)
    return masks


def _extract_scalar_probs(y_prob: np.ndarray) -> np.ndarray:
    """Extract scalar positive-class probability from model output."""
    if y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return y_prob[:, 1]
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        return y_prob[:, 0]
    return y_prob.ravel()


@torch.no_grad()
def compute_deletion_faithfulness(
    model: "AttentionLSTM",
    loader,
    device: str,
    fracs: List[float] = None,
    n_rand_repeats: int = 5,
) -> Dict[str, Any]:
    """Compute attention deletion faithfulness curves.

    For each deletion fraction, zeros out the top-attended (or random) time steps
    in the embedded representation and measures how predicted probabilities change.
    A faithful attention mechanism should cause a larger drop when high-attention
    steps are deleted compared to random deletion.

    Args:
        model: trained AttentionLSTM with deletion_mask support in forward().
        loader: DataLoader yielding evaluation batches.
        device: device string, e.g. "cpu" or "cuda".
        fracs: list of deletion fractions to evaluate (default: 0.1–0.9).
        n_rand_repeats: number of random mask draws to average over.

    Returns:
        dict with keys:
            - fracs: list of deletion fractions used.
            - attn_drop_curve: mean |p_clean - p_attn_deleted| per fraction.
            - rand_drop_curve: mean |p_clean - p_rand_deleted| per fraction.
            - faithfulness_score: AUC(attn_drop) - AUC(rand_drop).
              Positive = attention identifies causally important steps.
    """
    if fracs is None:
        fracs = FAITHFULNESS_FRACS

    model.eval()
    rng = np.random.default_rng(seed=0)

    attn_drops: Dict[float, List[float]] = {f: [] for f in fracs}
    rand_drops: Dict[float, List[float]] = {f: [] for f in fracs}

    for batch in tqdm(loader, desc="faithfulness", leave=False):
        batch = move_to_device(batch, device)
        clean_out = model(**batch)

        p_clean = _extract_scalar_probs(
            clean_out["y_prob"].detach().cpu().numpy()
        )
        raw_attn = {
            k: (v.numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
            for k, v in clean_out["attention_weights"].items()
        }

        for frac in fracs:
            # Attention-ordered deletion
            attn_dmask = build_deletion_mask(raw_attn, frac)
            attn_dmask = move_to_device(attn_dmask, device)
            attn_out = model(deletion_mask=attn_dmask, **batch)
            p_attn = _extract_scalar_probs(
                attn_out["y_prob"].detach().cpu().numpy()
            )
            attn_drops[frac].extend(np.abs(p_clean - p_attn).tolist())

            # Random deletion (average over repeats)
            rand_p_list = []
            for _ in range(n_rand_repeats):
                rand_dmask = build_random_deletion_mask(raw_attn, frac, rng)
                rand_dmask = move_to_device(rand_dmask, device)
                rand_out = model(deletion_mask=rand_dmask, **batch)
                rand_p_list.append(
                    _extract_scalar_probs(
                        rand_out["y_prob"].detach().cpu().numpy()
                    )
                )
            p_rand_avg = np.mean(rand_p_list, axis=0)
            rand_drops[frac].extend(np.abs(p_clean - p_rand_avg).tolist())

    attn_curve = [float(np.mean(attn_drops[f])) for f in fracs]
    rand_curve = [float(np.mean(rand_drops[f])) for f in fracs]
    faithfulness_score = float(
        np.trapz(attn_curve, fracs) - np.trapz(rand_curve, fracs)
    )

    return {
        "fracs": fracs,
        "attn_drop_curve": attn_curve,
        "rand_drop_curve": rand_curve,
        "faithfulness_score": faithfulness_score,
    }


def run_faithfulness_extension(
    dataset,
    train_loader,
    val_loader,
    test_loader,
    n_seeds: int = FAITHFULNESS_SEEDS,
) -> Dict[str, Any]:
    """Train n_seeds AttentionLSTM models and compute deletion faithfulness for each.

    Args:
        dataset: the SampleDataset used to build models.
        train_loader: DataLoader for training.
        val_loader: DataLoader for validation (used to select best checkpoint).
        test_loader: DataLoader for faithfulness evaluation.
        n_seeds: number of seeds to train and evaluate.

    Returns:
        dict with:
            - per_seed: list of per-seed faithfulness dicts.
            - faithfulness_score_mean: mean faithfulness score across seeds.
            - faithfulness_score_std: std of faithfulness score across seeds.
    """
    seeds_to_use = SEEDS[:n_seeds]
    per_seed_results = []

    for seed in seeds_to_use:
        print(f"[faithfulness] training seed={seed}...")
        set_seed(seed)
        model = build_attention_model(dataset).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_val_pr = -np.inf
        best_state = None
        for epoch in range(1, EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, DEVICE)
            val_result = evaluate(model, val_loader, DEVICE, expect_attention=False)
            if val_result["pr_auc"] > best_val_pr:
                best_val_pr = val_result["pr_auc"]
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if best_state is not None:
            model.load_state_dict(best_state)

        faith = compute_deletion_faithfulness(
            model,
            test_loader,
            DEVICE,
            fracs=FAITHFULNESS_FRACS,
            n_rand_repeats=FAITHFULNESS_RAND_REPEATS,
        )
        faith["seed"] = seed
        per_seed_results.append(faith)

        print(
            f"[faithfulness] seed={seed} "
            f"faithfulness_score={faith['faithfulness_score']:.4f}"
        )

    scores = [r["faithfulness_score"] for r in per_seed_results]
    return {
        "per_seed": per_seed_results,
        "faithfulness_score_mean": float(np.mean(scores)),
        "faithfulness_score_std": float(np.std(scores)),
    }


def plot_deletion_curves(faithfulness_result: Dict[str, Any], out_dir: Path) -> None:
    """Plot attention-ordered vs. random deletion curves across seeds.

    Args:
        faithfulness_result: output of run_faithfulness_extension().
        out_dir: directory to save the figure.
    """
    fracs = faithfulness_result["per_seed"][0]["fracs"]
    attn_curves = np.array(
        [r["attn_drop_curve"] for r in faithfulness_result["per_seed"]]
    )
    rand_curves = np.array(
        [r["rand_drop_curve"] for r in faithfulness_result["per_seed"]]
    )

    attn_mean, attn_std = attn_curves.mean(0), attn_curves.std(0)
    rand_mean, rand_std = rand_curves.mean(0), rand_curves.std(0)

    plt.figure(figsize=(7, 5))
    plt.plot(
        fracs, attn_mean, marker="o", label="Attention-ordered deletion",
        color="steelblue",
    )
    plt.fill_between(
        fracs, attn_mean - attn_std, attn_mean + attn_std, alpha=0.2,
        color="steelblue",
    )
    plt.plot(fracs, rand_mean, marker="s", label="Random deletion", color="tomato")
    plt.fill_between(
        fracs, rand_mean - rand_std, rand_mean + rand_std, alpha=0.2, color="tomato",
    )

    score_mean = faithfulness_result["faithfulness_score_mean"]
    score_std = faithfulness_result["faithfulness_score_std"]
    plt.xlabel("Fraction of Time Steps Deleted")
    plt.ylabel("Mean |ΔProbability|")
    plt.title(
        f"Deletion Faithfulness (score={score_mean:.4f}±{score_std:.4f})\n"
        f"n={len(faithfulness_result['per_seed'])} seeds"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "deletion_faithfulness_curve.png", dpi=200)
    plt.close()


def save_json(obj: Dict[str, Any], path: Path) -> None:
    def convert(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().tolist()
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, list):
            return [convert(v) for v in x]
        if isinstance(x, (np.float32, np.float64, np.int32, np.int64)):
            return x.item()
        return x

    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)


# load data
def get_label_from_sample(sample):
    if "readmission" in sample:
        y = sample["readmission"]
    elif "label" in sample:
        y = sample["label"]
    elif "y_true" in sample:
        y = sample["y_true"]
    else:
        raise KeyError(
            f"Could not find label key in sample keys: {list(sample.keys())}")

    y = np.asarray(y)
    return int(np.ravel(y)[0])


def label_counts(ds):
    ys = [get_label_from_sample(ds[i]) for i in range(len(ds))]
    ys = np.asarray(ys, dtype=int)
    unique, counts = np.unique(ys, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def has_both_classes(ds):
    counts = label_counts(ds)
    return 0 in counts and 1 in counts


def build_data():
    base_dataset = MIMIC3Dataset(
        root=DATA_ROOT,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        dev=False,
    )

    sample_dataset = base_dataset.set_task(ReadmissionPredictionMIMIC3())

    chosen_seed = None
    for split_seed in range(1, 1001):
        train_ds, val_ds, test_ds = split_by_sample(
            sample_dataset, ratios=[0.6, 0.2, 0.2], seed=split_seed
        )

        if has_both_classes(val_ds) and has_both_classes(test_ds):
            chosen_seed = split_seed
            break

    if chosen_seed is None:
        raise ValueError(
            "Could not find a split seed where both val and test contain both classes.")

    print(f"Using split seed: {chosen_seed}")
    print("train label distribution:", label_counts(train_ds))
    print("val label distribution:", label_counts(val_ds))
    print("test label distribution:", label_counts(test_ds))

    train_loader = get_dataloader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return sample_dataset, train_loader, val_loader, test_loader


# build model
def build_attention_model(dataset):
    return AttentionLSTM(
        dataset=dataset,
        embedding_dim=128,
        hidden_dim=128,
        dropout=0.5,
        num_layers=1,
        bidirectional=False,
    )


def build_rnn_baseline(dataset):
    return RNN(
        dataset=dataset,
        embedding_dim=128,
        hidden_dim=128,
        rnn_type="LSTM",
        dropout=0.5,
        num_layers=1,
        bidirectional=False,
    )


# train loop
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = []

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_to_device(batch, device)
        optimizer.zero_grad()

        output = model(**batch)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        losses.append(loss.detach().cpu().item())

    return float(np.mean(losses)) if losses else float("nan")


# evaluate
@torch.no_grad()
def evaluate(model, loader, device, expect_attention=False):
    model.eval()

    y_true_all = []
    y_prob_all = []
    losses = []

    attention_store: Dict[str, List[np.ndarray]] = {}

    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_to_device(batch, device)
        output = model(**batch)

        losses.append(float(output["loss"].detach().cpu().item()))

        y_true = output["y_true"].detach().cpu().numpy()
        y_prob = output["y_prob"].detach().cpu().numpy()

        # binary output cleanup
        if y_prob.ndim == 2 and y_prob.shape[1] == 2:
            y_prob = y_prob[:, 1]
        elif y_prob.ndim == 2 and y_prob.shape[1] == 1:
            y_prob = y_prob[:, 0]

        if y_true.ndim == 2 and y_true.shape[1] == 1:
            y_true = y_true[:, 0]

        y_true_all.append(y_true)
        y_prob_all.append(y_prob)

        if expect_attention:
            attn = output["attention_weights"]

            # current model returns a dict of CPU tensors
            for feature_key, weights in attn.items():
                if torch.is_tensor(weights):
                    weights_np = weights.numpy()
                else:
                    weights_np = np.asarray(weights)
                attention_store.setdefault(feature_key, []).append(weights_np)

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_prob_all = np.concatenate(y_prob_all, axis=0)

    results = {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "y_true": y_true_all,
        "y_prob": y_prob_all,
        "roc_auc": safe_roc_auc(y_true_all, y_prob_all),
        "pr_auc": safe_pr_auc(y_true_all, y_prob_all),
    }

    if expect_attention:
        results["attention_weights"] = {
            k: np.concatenate(v, axis=0) for k, v in attention_store.items()
        }

    return results


def run_single_seed(model_name, dataset, train_loader, val_loader, test_loader, seed):
    set_seed(seed)

    if model_name == "attention_lstm":
        model = build_attention_model(dataset).to(DEVICE)
        expect_attention = True
    elif model_name == "rnn_baseline":
        model = build_rnn_baseline(dataset).to(DEVICE)
        expect_attention = False
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_pr = -np.inf
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE)
        val_result = evaluate(model, val_loader, DEVICE,
                              expect_attention=False)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result["loss"],
            "val_roc_auc": val_result["roc_auc"],
            "val_pr_auc": val_result["pr_auc"],
        }
        history.append(row)

        if val_result["pr_auc"] > best_val_pr:
            best_val_pr = val_result["pr_auc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

        print(
            f"[{model_name}] seed={seed} epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_roc_auc={val_result['roc_auc']:.4f} "
            f"val_pr_auc={val_result['pr_auc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    test_result = evaluate(model, test_loader, DEVICE,
                           expect_attention=expect_attention)
    test_result["seed"] = seed
    test_result["model_name"] = model_name
    test_result["history"] = history
    return test_result


# compute mean/std across seeds
def summarize_runs(runs):
    roc_aucs = [r["roc_auc"] for r in runs if not np.isnan(r["roc_auc"])]
    pr_aucs = [r["pr_auc"] for r in runs if not np.isnan(r["pr_auc"])]

    return {
        "roc_auc_mean": float(np.mean(roc_aucs)) if roc_aucs else float("nan"),
        "roc_auc_std": float(np.std(roc_aucs)) if roc_aucs else float("nan"),
        "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else float("nan"),
        "pr_auc_std": float(np.std(pr_aucs)) if pr_aucs else float("nan"),
    }


# multi-seed experiment
def run_attention_experiment(dataset, train_loader, val_loader, test_loader):
    runs = []

    for seed in SEEDS:
        result = run_single_seed(
            "attention_lstm",
            dataset,
            train_loader,
            val_loader,
            test_loader,
            seed,
        )
        runs.append(result)

    summary = summarize_runs(runs)

    attention_runs = []
    for result in runs:
        flat_attention = flatten_attention_dict(result["attention_weights"])
        attention_runs.append(flat_attention)

    stability = compute_attention_stability(attention_runs, k=TOPK)

    return {
        "runs": runs,
        "summary": summary,
        "attention_stability": stability,
    }


# ablation study: compare against RNN baseline
def run_rnn_ablation(dataset, train_loader, val_loader, test_loader):
    runs = []

    for seed in SEEDS:
        result = run_single_seed(
            "rnn_baseline",
            dataset,
            train_loader,
            val_loader,
            test_loader,
            seed,
        )
        runs.append(result)

    summary = summarize_runs(runs)

    return {
        "runs": runs,
        "summary": summary,
    }


# plot performance across seeds
def plot_metrics(attention_result, baseline_result, out_dir: Path):
    # Filter separately for ROC and PR so x and y always match
    attn_valid_roc = [
        r for r in attention_result["runs"]
        if not np.isnan(r["roc_auc"])
    ]
    attn_valid_pr = [
        r for r in attention_result["runs"]
        if not np.isnan(r["pr_auc"])
    ]

    rnn_valid_roc = [
        r for r in baseline_result["runs"]
        if not np.isnan(r["roc_auc"])
    ]
    rnn_valid_pr = [
        r for r in baseline_result["runs"]
        if not np.isnan(r["pr_auc"])
    ]

    seeds_attn_roc = [r["seed"] for r in attn_valid_roc]
    roc_attn = [r["roc_auc"] for r in attn_valid_roc]

    seeds_attn_pr = [r["seed"] for r in attn_valid_pr]
    pr_attn = [r["pr_auc"] for r in attn_valid_pr]

    seeds_rnn_roc = [r["seed"] for r in rnn_valid_roc]
    roc_rnn = [r["roc_auc"] for r in rnn_valid_roc]

    seeds_rnn_pr = [r["seed"] for r in rnn_valid_pr]
    pr_rnn = [r["pr_auc"] for r in rnn_valid_pr]

    plt.figure(figsize=(8, 5))
    if len(seeds_attn_roc) > 0:
        plt.plot(seeds_attn_roc, roc_attn, marker="o", label="AttentionLSTM")
    if len(seeds_rnn_roc) > 0:
        plt.plot(seeds_rnn_roc, roc_rnn, marker="o", label="RNN baseline")
    plt.xlabel("Seed")
    plt.ylabel("ROC-AUC")
    plt.title("ROC-AUC Across Seeds")
    if len(seeds_attn_roc) > 0 or len(seeds_rnn_roc) > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_auc_across_seeds.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    if len(seeds_attn_pr) > 0:
        plt.plot(seeds_attn_pr, pr_attn, marker="o", label="AttentionLSTM")
    if len(seeds_rnn_pr) > 0:
        plt.plot(seeds_rnn_pr, pr_rnn, marker="o", label="RNN baseline")
    plt.xlabel("Seed")
    plt.ylabel("PR-AUC")
    plt.title("PR-AUC Across Seeds")
    if len(seeds_attn_pr) > 0 or len(seeds_rnn_pr) > 0:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "pr_auc_across_seeds.png", dpi=200)
    plt.close()


def plot_mean_attention(attention_result, out_dir: Path):
    first_run = attention_result["runs"][0]
    attn_dict = first_run["attention_weights"]

    for feature_key, arr in attn_dict.items():
        mean_attn = arr.mean(axis=0)

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(len(mean_attn)), mean_attn)
        plt.xlabel("Time Step")
        plt.ylabel("Mean Attention")
        plt.title(f"Mean Attention: {feature_key}")
        plt.tight_layout()
        plt.savefig(out_dir / f"mean_attention_{feature_key}.png", dpi=200)
        plt.close()


# full experimental pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faithfulness-only",
        action="store_true",
        help=(
            "Skip the full 1000-seed experiment and RNN ablation. "
            "Load existing summary.json and run only the faithfulness extension."
        ),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    dataset, train_loader, val_loader, test_loader = build_data()

    if args.faithfulness_only:
        summary_path = OUTPUT_DIR / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(
                f"{summary_path} not found. Run without --faithfulness-only first."
            )
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        print("Running sanity check...")
        sanity_model = build_attention_model(dataset).to(DEVICE)
        first_batch = next(iter(train_loader))
        first_batch = move_to_device(first_batch, DEVICE)
        sanity_output = sanity_model(**first_batch)
        print("Sanity output keys:", sanity_output.keys())
        if "attention_weights" not in sanity_output:
            raise ValueError("attention_weights missing from model output")

        print("Running AttentionLSTM multi-seed experiment...")
        attention_result = run_attention_experiment(
            dataset, train_loader, val_loader, test_loader
        )

        print("Running RNN ablation...")
        baseline_result = run_rnn_ablation(
            dataset, train_loader, val_loader, test_loader
        )

        summary = {
            "attention_lstm_summary": attention_result["summary"],
            "attention_stability": attention_result["attention_stability"],
            "rnn_baseline_summary": baseline_result["summary"],
        }
        save_json(attention_result, OUTPUT_DIR / "attention_result_full.json")
        save_json(baseline_result, OUTPUT_DIR / "rnn_baseline_full.json")

    print(
        f"Running deletion faithfulness extension ({FAITHFULNESS_SEEDS} seeds)..."
    )
    faithfulness_result = run_faithfulness_extension(
        dataset, train_loader, val_loader, test_loader, n_seeds=FAITHFULNESS_SEEDS
    )

    summary["faithfulness_extension"] = {
        "faithfulness_score_mean": faithfulness_result["faithfulness_score_mean"],
        "faithfulness_score_std": faithfulness_result["faithfulness_score_std"],
    }

    print("\n===== FINAL SUMMARY =====")
    print(json.dumps(summary, indent=2))

    save_json(summary, OUTPUT_DIR / "summary.json")
    save_json(faithfulness_result, OUTPUT_DIR / "faithfulness_result.json")

    if not args.faithfulness_only:
        plot_metrics(attention_result, baseline_result, OUTPUT_DIR)
        plot_mean_attention(attention_result, OUTPUT_DIR)
    plot_deletion_curves(faithfulness_result, OUTPUT_DIR)

    print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
