"""BulkRNABert ablation study: TCGARNASeq + cancer type classification.

Paper: Gélard et al., "BulkRNABert: Cancer prognosis from bulk RNA-seq
based language models", bioRxiv 2024.

Ablations (extensions beyond the paper text):
    1. Binning resolution: B in {32, 64, 128} (paper fixes B=64 without sweep)
    2. Frozen vs IA3 vs full fine-tuning (adds full FT vs paper's frozen/IA3)
    3. Cox survival loss training curve (sanity check on synthetic survival)
    4. Built-in :class:`~pyhealth.models.MLP` on flattened gene bins vs
       :class:`~pyhealth.models.bulk_rna_bert.BulkRNABert` (rubric baseline)

Runs entirely on synthetic data. To use real TCGA data, replace
`make_synthetic_data` with your downloaded rna_seq.csv / clinical.csv.

Usage:
    python examples/tcga_rnaseq_cancer_type_bulk_rna_bert.py
"""

import os
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

if not hasattr(torch, "uint16"):
    torch.uint16 = torch.int16

def _ensure_models_pkg():
    """Register ``pyhealth.models`` without executing ``models/__init__.py``."""
    import importlib
    import sys
    import types
    from pathlib import Path

    if "pyhealth.models" in sys.modules:
        return
    import pyhealth 

    repo = Path(__file__).resolve().parents[1]
    pkg = types.ModuleType("pyhealth.models")
    pkg.__path__ = [str(repo / "pyhealth" / "models")]
    sys.modules["pyhealth.models"] = pkg
    bm = importlib.import_module("pyhealth.models.base_model")
    pkg.BaseModel = bm.BaseModel


def _bulk_rna_bert():
    import importlib

    _ensure_models_pkg()
    return importlib.import_module("pyhealth.models.bulk_rna_bert")


def _mlp_class():
    import importlib

    _ensure_models_pkg()
    return importlib.import_module("pyhealth.models.mlp").MLP


N_PATIENTS = 32
N_GENES = 200
COHORTS = ["BRCA", "LUAD", "BLCA", "GBM", "UCEC"]


def make_synthetic_data(root: str) -> None:
    """Write synthetic rna_seq.csv and clinical.csv to root."""
    np.random.seed(0)
    expr = np.random.exponential(scale=20.0, size=(N_PATIENTS, N_GENES))
    gene_names = [f"GENE{i}" for i in range(N_GENES)]
    cohort_labels = [COHORTS[i % len(COHORTS)] for i in range(N_PATIENTS)]
    patient_ids = [f"TCGA-{i:03d}" for i in range(N_PATIENTS)]

    rnaseq_df = pd.DataFrame(expr, columns=gene_names)
    rnaseq_df.insert(0, "patient_id", patient_ids)
    rnaseq_df.insert(1, "cohort", cohort_labels)
    rnaseq_df.to_csv(os.path.join(root, "rna_seq.csv"), index=False)

    vital = ["dead" if i % 3 == 0 else "alive" for i in range(N_PATIENTS)]
    days_death = [float(200 + i * 10) if v == "dead" else None
                  for i, v in enumerate(vital)]
    days_follow = [None if v == "dead" else float(400 + i * 5)
                   for i, v in enumerate(vital)]

    pd.DataFrame({
        "patient_id": patient_ids,
        "cohort": cohort_labels,
        "vital_status": vital,
        "days_to_death": days_death,
        "days_to_last_follow_up": days_follow,
    }).to_csv(os.path.join(root, "clinical.csv"), index=False)


def _load_tokens(root, n_bins=64):
    """Helper to preprocess and load token tensors from root."""
    from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset

    rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
    clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")
    TCGARNASeqDataset._prepare_metadata(root, n_bins, None, rnaseq_out, clinical_out)
    df = pd.read_csv(rnaseq_out)
    gene_cols = [c for c in df.columns if c not in ("patient_id", "cohort")]
    token_ids = torch.tensor(df[gene_cols].values, dtype=torch.long)
    cohort_to_idx = {c: i for i, c in enumerate(COHORTS)}
    labels = torch.tensor(
        [cohort_to_idx.get(c, 0) for c in df["cohort"]], dtype=torch.long
    )
    return token_ids, labels, gene_cols


# Ablation 1: Binning resolution

def ablation_binning_resolution():
    """Test effect of B in {32, 64, 128} on classification loss."""

    BulkRNABert = _bulk_rna_bert().BulkRNABert

    results = {}
    for n_bins in [32, 64, 128]:
        with tempfile.TemporaryDirectory() as root:
            make_synthetic_data(root)
            token_ids, labels, gene_cols = _load_tokens(root, n_bins=n_bins)

            model = BulkRNABert(
                dataset=None,
                n_genes=len(gene_cols),
                n_bins=n_bins,
                embedding_dim=64,
                n_layers=2,
                n_heads=4,
                ffn_dim=128,
                dropout=0.0,
                mlp_hidden=(32,),
                mode="classification",
                n_classes=len(COHORTS),
            )
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            t0 = time.time()
            for _ in range(20):
                optimizer.zero_grad()
                out = model(token_ids=token_ids, cancer_type=labels)
                out["loss"].backward()
                optimizer.step()
            final_loss = out["loss"].item()
            elapsed = time.time() - t0
            results[n_bins] = final_loss
            print(f"  B={n_bins:3d} | loss={final_loss:.4f} | time={elapsed:.2f}s")

    return results

# Ablation 2: Frozen vs IA3 vs full fine-tuning

def ablation_finetuning_strategy():
    """Compare frozen backbone, IA3, and full fine-tuning."""
    BulkRNABert = _bulk_rna_bert().BulkRNABert

    with tempfile.TemporaryDirectory() as root:
        make_synthetic_data(root)
        token_ids, labels, gene_cols = _load_tokens(root)

        strategies = {
            "frozen_backbone": {"use_ia3": False, "freeze_encoder": True},
            "ia3_finetuning":  {"use_ia3": True,  "freeze_encoder": True},
            "full_finetuning": {"use_ia3": False, "freeze_encoder": False},
        }

        results = {}
        for name, config in strategies.items():
            model = BulkRNABert(
                dataset=None,
                n_genes=len(gene_cols),
                n_bins=64,
                embedding_dim=64,
                n_layers=2,
                n_heads=4,
                ffn_dim=128,
                dropout=0.0,
                mlp_hidden=(32,),
                mode="classification",
                n_classes=len(COHORTS),
                use_ia3=config["use_ia3"],
            )
            if config["freeze_encoder"]:
                for p in model.encoder.parameters():
                    p.requires_grad = False
                for p in model.gene_embedding.parameters():
                    p.requires_grad = False
                for p in model.expr_embedding.parameters():
                    p.requires_grad = False

            n_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
            )
            model.train()
            for _ in range(20):
                optimizer.zero_grad()
                out = model(token_ids=token_ids, cancer_type=labels)
                out["loss"].backward()
                optimizer.step()

            final_loss = out["loss"].item()
            results[name] = (final_loss, n_params)
            print(
                f"  {name:20s} | trainable={n_params:6d} "
                f"| loss={final_loss:.4f}"
            )

    return results


# Ablation 3: Cox survival loss

def ablation_cox_loss():
    """Verify Cox loss decreases during survival model training."""
    BulkRNABert = _bulk_rna_bert().BulkRNABert

    with tempfile.TemporaryDirectory() as root:
        make_synthetic_data(root)
        from pyhealth.datasets.tcga_rnaseq import TCGARNASeqDataset
        rnaseq_out = os.path.join(root, "tcga_rnaseq_tokenized-pyhealth.csv")
        clinical_out = os.path.join(root, "tcga_rnaseq_clinical-pyhealth.csv")
        TCGARNASeqDataset._prepare_metadata(root, 64, None, rnaseq_out, clinical_out)

        df = pd.read_csv(rnaseq_out)
        clinical = pd.read_csv(clinical_out)
        gene_cols = [c for c in df.columns if c not in ("patient_id", "cohort")]
        merged = df.merge(clinical, on="patient_id", how="inner")

        token_ids = torch.tensor(merged[gene_cols].values, dtype=torch.long)
        vital = merged["vital_status"].str.lower().map(
            {"dead": 1, "alive": 0}
        ).fillna(0)
        events = torch.tensor(vital.values, dtype=torch.float32)
        times = torch.tensor(
            merged["days_to_death"].fillna(
                merged["days_to_last_follow_up"]
            ).fillna(365.0).values,
            dtype=torch.float32,
        )

        model = BulkRNABert(
            dataset=None,
            n_genes=len(gene_cols),
            n_bins=64,
            embedding_dim=64,
            n_layers=2,
            n_heads=4,
            ffn_dim=128,
            dropout=0.0,
            mlp_hidden=(32,),
            mode="survival",
            n_classes=1,
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        losses = []
        model.train()
        for step in range(30):
            optimizer.zero_grad()
            out = model(token_ids=token_ids, survival_time=times, event=events)
            out["loss"].backward()
            optimizer.step()
            losses.append(out["loss"].item())
            if step % 10 == 0:
                print(f"  step {step:3d} | loss={losses[-1]:.4f}")

        decreased = losses[-1] < losses[0]
        print(f"\n  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss:   {losses[-1]:.4f}")
        print(f"  Loss decreased: {decreased}")
    return losses


def ablation_mlp_vs_transformer():
    """Compare PyHealth ``MLP`` on flattened bins vs ``BulkRNABert``."""
    from pyhealth.datasets import create_sample_dataset, get_dataloader

    BulkRNABert = _bulk_rna_bert().BulkRNABert
    MLP = _mlp_class()

    mlp_loss = bert_loss = 0.0
    with tempfile.TemporaryDirectory() as root:
        make_synthetic_data(root)
        token_ids, labels, gene_cols = _load_tokens(root, n_bins=64)
        cohort_names = [COHORTS[i % len(COHORTS)] for i in range(len(token_ids))]
        samples = []
        for i in range(len(token_ids)):
            vec = [float(x) for x in token_ids[i].tolist()]
            samples.append(
                {
                    "patient_id": f"p{i}",
                    "expr_vec": vec,
                    "cancer_type": cohort_names[i],
                }
            )
        input_schema = {"expr_vec": "tensor"}
        output_schema = {"cancer_type": "multiclass"}
        sample_ds = create_sample_dataset(
            samples=samples,
            input_schema=input_schema,
            output_schema=output_schema,
            dataset_name="tcga_synth_tabular",
            task_name="TCGACancerTypeTabular",
        )

        mlp = MLP(sample_ds, embedding_dim=32, hidden_dim=32, n_layers=2)
        bert = BulkRNABert(
            dataset=None,
            n_genes=len(gene_cols),
            n_bins=64,
            embedding_dim=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            dropout=0.0,
            mlp_hidden=(32,),
            mode="classification",
            n_classes=len(COHORTS),
        )

        loader = get_dataloader(sample_ds, batch_size=len(samples), shuffle=False)
        batch_mlp = next(iter(loader))

        opt_mlp = optim.Adam(mlp.parameters(), lr=1e-3)
        opt_bert = optim.Adam(bert.parameters(), lr=1e-3)
        mlp.train()
        bert.train()
        out_m = out_b = None
        for _ in range(15):
            opt_mlp.zero_grad()
            out_m = mlp(**batch_mlp)
            out_m["loss"].backward()
            opt_mlp.step()

            opt_bert.zero_grad()
            out_b = bert(token_ids=token_ids, cancer_type=labels)
            out_b["loss"].backward()
            opt_bert.step()

        print(f"  MLP final CE loss:          {out_m['loss'].item():.4f}")
        print(f"  BulkRNABert final CE loss:  {out_b['loss'].item():.4f}")
        print("\nConclusion: compare tabular MLP vs sequence transformer on same bins.")
        mlp_loss = float(out_m["loss"].item())
        bert_loss = float(out_b["loss"].item())
    return mlp_loss, bert_loss


if __name__ == "__main__":
    ablation_binning_resolution()
    ablation_finetuning_strategy()
    ablation_cox_loss()
    ablation_mlp_vs_transformer()
    print("\nAll ablations complete.")