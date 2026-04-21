"""PubMedBERT encoder ablation for the amiodarone case study.

This script extends Kaul & Gordon (2024) by testing whether a
domain-specific pretrained language model (PubMedBERT) produces
better priors for conformal meta-analysis than hand-crafted numeric
features on the 21 amiodarone trials from Letelier et al. (2003).

Pipeline:
    1. Load the amiodarone dataset (21 trials, 10 placebo-controlled
       "trusted" + 11 other "untrusted").
    2. For each trial, obtain text = real abstract (if extracted)
       or generated clinical prose as a fallback.
    3. Embed text with PubMedBERT (CLS token, frozen weights).
    4. Train a CMAPriorEncoder MLP head on the untrusted embeddings
       using PyHealth's Trainer.
    5. Run ConformalMetaAnalysisModel on the trusted trials with
       the learned prior.
    6. Compare against the hand-crafted feature baseline and HKSJ.

Ablation dimensions (produces a 5-row result table):
    - Encoder input:  13 hand-crafted features vs 768-dim PubMedBERT
    - MLP head depth: default [64, 32] vs shallow [32] vs deep [128, 64]
    - HKSJ baseline (no encoder at all)

Fallback: if the ``transformers`` package is not installed, the script
skips the BERT rows and runs only the hand-crafted baseline + HKSJ.
This keeps the example runnable without a 440 MB model download.

Expected findings (what the ablation is designed to reveal):
    - If PubMedBERT embeddings encode trial-level information
      that the 13 hand-crafted features miss, the BERT rows should
      show lower Prior MSE than the hand-crafted baseline.
    - Because CMA uses the prior only to set interval centers (the
      kernel is fixed to the hand-crafted features), a better
      prior should translate to narrower ``CMA Width`` with
      coverage staying near ``1 - alpha``.
    - HKSJ ignores priors entirely, so its width is the "no
      learned prior" ceiling: CMA rows should be narrower than
      HKSJ whenever a prior is at least weakly informative.
    - MLP head depth ([32] vs [64, 32] vs [128, 64]) is secondary
      to the input representation on only 11 untrusted training
      trials; expect small differences across rows 2-4.

Usage:
    python amiodarone_trials_conformal_meta_analysis_cma.py

Optional (for the full ablation):
    pip install transformers

Reference:
    Kaul, S. and Gordon, G. J. 2024. "Meta-Analysis with Untrusted
    Data." Proceedings of Machine Learning Research, 259:563-593.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import t as t_dist
from torch.utils.data import DataLoader

from pyhealth.datasets import AmiodaroneTrialDataset, get_dataloader
from pyhealth.datasets.amiodarone_trial_dataset import FEATURE_COLUMNS
from pyhealth.models.cma_prior_encoder import CMAPriorEncoder
from pyhealth.models.conformal_meta_analysis_krr import (
    ConformalMetaAnalysisModel,
)
from pyhealth.tasks.conformal_meta_analysis import ConformalMetaAnalysisTask
from pyhealth.trainer import Trainer


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ABSTRACTS_PATH = Path("./amiodarone_abstracts.json")
DATASET_ROOT = "./data/amiodarone"
PUBMEDBERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


# ---------------------------------------------------------------------
# Text rendering (fallback for trials without real abstracts)
# ---------------------------------------------------------------------
def generate_clinical_prose(row: pd.Series) -> str:
    """Render a trial's structured features as clinical prose.

    Used when a real abstract is not available. Output mimics the
    style of a clinical trial abstract so PubMedBERT processes it
    similarly to a real abstract.
    """
    name = row.get("trial_name", "trial")
    dose = row.get("amiodarone_total_24h_mg", 0) or 0
    comp_intensity = int(row.get("comparison_intensity", 0) or 0)
    af_long = bool(row.get("af_duration_gt_48h", 0) or 0)
    outcome_long = bool(row.get("outcome_time_gt_48h", 0) or 0)
    mean_age = row.get("mean_age", 0) or 0
    la_size = row.get("mean_la_size", 0) or 0
    male_frac = row.get("fraction_male", 0) or 0
    cv_frac = row.get("fraction_cv_disease", 0) or 0
    followup = row.get("followup_fraction", 1.0) or 1.0
    masked_pt = bool(row.get("masked_patients", 0) or 0)
    masked_cg = bool(row.get("masked_caregiver", 0) or 0)
    adequate_concealment = bool(row.get("adequate_concealment", 0) or 0)

    comparison = {
        0: "placebo control",
        1: "low-intensity active comparator",
        2: "high-intensity active comparator",
    }.get(comp_intensity, "active comparator")

    af_str = "persistent" if af_long else "recent-onset"
    outcome_str = (
        "long-term conversion (greater than 48 hours)"
        if outcome_long
        else "short-term conversion (within 48 hours)"
    )
    blinding = (
        "double-blinded"
        if (masked_pt and masked_cg)
        else "single-blinded"
        if masked_pt
        else "open-label"
    )

    return (
        f"Randomized controlled trial ({name}) evaluating amiodarone "
        f"versus {comparison} for conversion of atrial fibrillation to "
        f"sinus rhythm. Patients had {af_str} atrial fibrillation. "
        f"Total amiodarone dose over 24 hours: {dose:.0f} mg. "
        f"Outcome assessed as {outcome_str}. "
        f"Mean patient age {mean_age:.0f} years; "
        f"mean left atrial size {la_size:.1f} cm. "
        f"Male fraction {male_frac:.2f}; "
        f"cardiovascular disease prevalence {cv_frac:.2f}. "
        f"Follow-up completion {followup:.2f}. "
        f"Study was {blinding} with "
        f"{'adequate' if adequate_concealment else 'unclear'} "
        f"allocation concealment."
    )


def get_trial_text(
    row: pd.Series,
    abstracts: Dict[str, str],
) -> Tuple[str, str]:
    """Return ``(text, source)`` where source is 'real' or 'generated'."""
    name = row.get("trial_name", "")
    real = abstracts.get(name, "").strip() if name else ""
    if real and len(real) > 100:
        return real, "real"
    return generate_clinical_prose(row), "generated"


# ---------------------------------------------------------------------
# PubMedBERT embedding (deferred import)
# ---------------------------------------------------------------------
def embed_with_bert(
    texts: List[str],
    model_name: str = PUBMEDBERT,
    device: str = "cpu",
) -> torch.Tensor:
    """Return ``[n, 768]`` CLS embeddings, one per text.

    Deferred-imports the ``transformers`` package so the rest of the
    script works without it.
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)
            out = model(**inputs)
            cls = out.last_hidden_state[:, 0, :].squeeze(0).cpu()
            all_emb.append(cls)
            if (i + 1) % 5 == 0:
                print(f"  embedded {i + 1}/{len(texts)}")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return torch.stack(all_emb)


# ---------------------------------------------------------------------
# HKSJ baseline (Proposition 10)
# ---------------------------------------------------------------------
def hksj_interval(
    Y: np.ndarray,
    V: np.ndarray,
    alpha: float = 0.1,
) -> Tuple[float, float]:
    """Hartung-Knapp-Sidik-Jonkman prediction interval."""
    n = len(Y)
    nu = 0.0
    for _ in range(1000):
        w = 1.0 / (V + nu)
        ate = np.sum(w * Y) / np.sum(w)
        nu_new = max(
            0.0,
            np.sum(w ** 2 * ((Y - ate) ** 2 - V)) / np.sum(w ** 2)
            + 1.0 / np.sum(w),
        )
        if abs(nu_new - nu) < 1e-8:
            break
        nu = nu_new
    w = 1.0 / (V + nu)
    ate = np.sum(w * Y) / np.sum(w)
    var_ate = np.sum((Y - ate) ** 2 * w) / ((n - 1) * np.sum(w))
    half = t_dist.ppf(1 - alpha / 2, df=n - 1) * np.sqrt(nu + var_ate)
    return float(ate - half), float(ate + half)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _batch_from_samples(sample_dataset) -> Dict[str, torch.Tensor]:
    """Stack all samples into a single batch dict via ``get_dataloader``.

    Thin wrapper around PyHealth's ``get_dataloader`` that pulls
    every sample into a single batch. Using the library's loader
    (instead of hand-rolled stacking) keeps this example aligned
    with PyHealth conventions and automatically inherits the
    default collate's handling of tensor / scalar keys.
    """
    loader = get_dataloader(
        sample_dataset,
        batch_size=len(sample_dataset),
        shuffle=False,
    )
    return next(iter(loader))


class _FeatureDataset(torch.utils.data.Dataset):
    """Minimal Dataset wrapper exposing (features, true_effect) samples.

    Used both for initializing ``CMAPriorEncoder`` (to infer input_dim
    via ``__getitem__(0)``) and as the source for a ``DataLoader``
    consumed by PyHealth's ``Trainer``.
    """

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        self._feats = features.float()
        self._tgts = targets.view(-1).float()
        self.input_schema = {"features": "tensor"}
        self.output_schema = {"true_effect": "regression"}

    def __len__(self) -> int:
        return len(self._feats)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self._feats[i],
            "true_effect": self._tgts[i].unsqueeze(-1),
        }


def _collate_encoder_batch(
    items: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Collate function that stacks per-sample tensors into a batch.

    PyHealth's ``Trainer`` passes the collated dict straight into
    ``model(**batch)``, so keys must match the encoder's ``forward``
    signature (``features``, ``true_effect``).
    """
    return {
        "features": torch.stack([item["features"] for item in items]),
        "true_effect": torch.stack([item["true_effect"] for item in items]),
    }


def train_encoder_with_trainer(
    encoder: CMAPriorEncoder,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 1e-3,
) -> CMAPriorEncoder:
    """Train the encoder via PyHealth's ``Trainer``.

    ``CMAPriorEncoder.forward`` already returns the ``{y_pred, y_true,
    loss}`` dict that ``Trainer`` expects, so no wrapper is needed.
    """
    dataset = _FeatureDataset(features, targets)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_encoder_batch,
    )

    trainer = Trainer(
        model=encoder,
        metrics=["mse"],
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=train_loader,
        epochs=epochs,
        optimizer_params={"lr": lr},
        monitor="mse",
        monitor_criterion="min",
        load_best_model_at_last=False,
    )
    encoder.eval()
    return encoder


# ---------------------------------------------------------------------
# Core: run CMA with a given feature representation
# ---------------------------------------------------------------------
def run_cma_with_features(
    u_features: torch.Tensor,
    t_features: torch.Tensor,
    u_batch: Dict[str, torch.Tensor],
    t_batch: Dict[str, torch.Tensor],
    trusted_dataset,
    label: str,
    input_desc: str,
    hidden_dims: Optional[List[int]] = None,
    embed_dim: int = 16,
    alpha: float = 0.1,
) -> Dict:
    """Train encoder on untrusted features, run CMA on trusted.

    The CMA model uses the ORIGINAL hand-crafted features for its
    KRR kernel; the learned encoder only drives the prior mean M.
    This isolates the effect of prior quality from any change in
    the kernel.
    """
    if hidden_dims is None:
        hidden_dims = [64, 32]

    fake_u = _FeatureDataset(u_features, u_batch["true_effect"])

    encoder = CMAPriorEncoder(
        dataset=fake_u,
        hidden_dims=hidden_dims,
        embed_dim=embed_dim,
    )
    train_encoder_with_trainer(
        encoder,
        features=u_features.float(),
        targets=u_batch["true_effect"],
    )

    # Predict M for trusted trials
    with torch.no_grad():
        M = encoder.predict_prior_mean(t_features.float())

    t_with_prior = dict(t_batch)
    t_with_prior["prior_mean"] = M.unsqueeze(-1)

    cma = ConformalMetaAnalysisModel(
        dataset=trusted_dataset,
        alpha=alpha,
    )
    with torch.no_grad():
        out = cma(**t_with_prior)

    lo = out["interval_lower"].cpu().numpy().ravel()
    hi = out["interval_upper"].cpu().numpy().ravel()
    u_true = t_batch["true_effect"].cpu().numpy().ravel()

    finite = np.isfinite(lo) & np.isfinite(hi)
    width = (
        float(np.mean(hi[finite] - lo[finite]))
        if finite.any()
        else np.nan
    )
    coverage = float(np.mean((u_true >= lo) & (u_true <= hi)))
    mse_prior = float(
        torch.mean((M - t_batch["true_effect"].squeeze(-1)) ** 2)
    )

    return {
        "Encoder": label,
        "Input": input_desc,
        "Feature Dim": u_features.shape[1],
        "Prior MSE": round(mse_prior, 4),
        "CMA Width": round(width, 4),
        "CMA Coverage": round(coverage, 3),
    }


# ---------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------
def run_bert_ablation(
    seed: int = 0,
    alpha: float = 0.1,
) -> pd.DataFrame:
    """Run the full ablation and return a results DataFrame."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Detect whether transformers is installed; skip BERT rows if not.
    try:
        import transformers  # noqa: F401

        bert_available = True
    except ImportError:
        print(
            "[INFO] `transformers` not installed. Skipping PubMedBERT "
            "rows; running hand-crafted baseline + HKSJ only. "
            "Install `transformers` to enable the full ablation."
        )
        bert_available = False

    # Load abstracts (empty dict if file not present)
    if ABSTRACTS_PATH.exists():
        abstracts = json.loads(
            ABSTRACTS_PATH.read_text(encoding="utf-8")
        )
    else:
        print(
            f"[INFO] {ABSTRACTS_PATH} not found. Using generated "
            f"prose for all trials. Populate this file with "
            f"{{trial_name: abstract_text}} to use real PDFs."
        )
        abstracts = {}

    # Load dataset and build lookup
    dataset = AmiodaroneTrialDataset(root=DATASET_ROOT)
    csv_path = os.path.join(
        DATASET_ROOT,
        "amiodarone_trials-metadata-pyhealth.csv",
    )
    df = pd.read_csv(csv_path)

    # Build text inputs
    texts: List[str] = []
    n_real, n_generated = 0, 0
    for _, row in df.iterrows():
        text, source = get_trial_text(row, abstracts)
        texts.append(text)
        if source == "real":
            n_real += 1
        else:
            n_generated += 1

    print(
        f"\nText inputs: {n_real} real abstracts, "
        f"{n_generated} generated prose, {len(df)} total\n"
    )

    # Align indices to splits
    untrusted_idx = df.index[df["split"] == "untrusted"].tolist()
    trusted_idx = df.index[df["split"] == "trusted"].tolist()

    # Hand-crafted baseline via the task pipeline
    task_u = ConformalMetaAnalysisTask(
        target_column="log_relative_risk",
        feature_columns=FEATURE_COLUMNS,
        split_column="split",
        split_value="untrusted",
        observed_column=None,
        variance_column=None,
        prior_column=None,
    )
    untrusted = dataset.set_task(task_u)
    u_batch = _batch_from_samples(untrusted)

    task_t = ConformalMetaAnalysisTask(
        target_column="log_relative_risk",
        feature_columns=FEATURE_COLUMNS,
        split_column="split",
        split_value="trusted",
        observed_column="log_relative_risk",
        variance_column="variance",
        prior_column=None,
    )
    trusted = dataset.set_task(task_t)
    t_batch = _batch_from_samples(trusted)

    rows: List[Dict] = []

    # Row 1: hand-crafted 13 features (baseline)
    print("Running hand-crafted baseline...")
    rows.append(
        run_cma_with_features(
            u_features=u_batch["features"],
            t_features=t_batch["features"],
            u_batch=u_batch,
            t_batch=t_batch,
            trusted_dataset=trusted,
            label="MLP",
            input_desc=f"{len(FEATURE_COLUMNS)} hand-crafted features",
            alpha=alpha,
        )
    )

    # Rows 2-4: PubMedBERT variants (only if transformers installed)
    if bert_available:
        print("\nEmbedding trials with PubMedBERT...")
        bert_emb = embed_with_bert(texts, PUBMEDBERT)
        print(f"BERT embeddings shape: {bert_emb.shape}\n")

        u_bert = bert_emb[untrusted_idx]
        t_bert = bert_emb[trusted_idx]

        print("Running PubMedBERT + default MLP...")
        rows.append(
            run_cma_with_features(
                u_features=u_bert,
                t_features=t_bert,
                u_batch=u_batch,
                t_batch=t_batch,
                trusted_dataset=trusted,
                label="PubMedBERT + MLP",
                input_desc=f"{n_real} real / {n_generated} gen",
                hidden_dims=[64, 32],
                embed_dim=16,
                alpha=alpha,
            )
        )

        for arch_name, hd, ed in [
            ("Shallow", [32], 8),
            ("Deep", [128, 64], 16),
        ]:
            print(f"Running PubMedBERT + {arch_name} MLP...")
            rows.append(
                run_cma_with_features(
                    u_features=u_bert,
                    t_features=t_bert,
                    u_batch=u_batch,
                    t_batch=t_batch,
                    trusted_dataset=trusted,
                    label=f"PubMedBERT + {arch_name}",
                    input_desc=f"{n_real} real / {n_generated} gen",
                    hidden_dims=hd,
                    embed_dim=ed,
                    alpha=alpha,
                )
            )

    # Final row: HKSJ baseline
    Y = t_batch["observed_effect"].cpu().numpy().ravel()
    V = t_batch["variance"].cpu().numpy().ravel()
    u_true = t_batch["true_effect"].cpu().numpy().ravel()
    hlo, hhi = hksj_interval(Y, V, alpha=alpha)
    hksj_cov = float(np.mean((u_true >= hlo) & (u_true <= hhi)))
    rows.append(
        {
            "Encoder": "HKSJ (baseline)",
            "Input": "observed Y, V only",
            "Feature Dim": 0,
            "Prior MSE": np.nan,
            "CMA Width": round(hhi - hlo, 4),
            "CMA Coverage": round(hksj_cov, 3),
        }
    )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 80)
    print("PubMedBERT Encoder Ablation for Conformal Meta-Analysis")
    print("=" * 80)

    results = run_bert_ablation()

    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)
    print(results.to_string(index=False))

    # Save for the report
    results.to_csv("bert_ablation_results.csv", index=False)
    print("\nSaved bert_ablation_results.csv")