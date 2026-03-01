"""
EHR Generation with a GPT-2 style Transformer on MIMIC-III (PyHealth)
======================================================================

This example applies the :class:`~pyhealth.models.generators.EHRGPTBaseline`
model to MIMIC-III data and generates synthetic patient EHR sequences.

The pipeline:

1. Load MIMIC-III via **PyHealth** and apply the ``EHRGenerationMIMIC3`` task
   to obtain per-patient nested visit sequences.
2. Serialise the nested sequences into plain text using ``VISIT_DELIM``
   separators (e.g. ``"250.00 401.9 VISIT_DELIM 272.0 428.0"``).
3. Train a word-level GPT-2 model via :meth:`EHRGPTBaseline.fit`.
4. Sample synthetic sequences via :meth:`EHRGPTBaseline.generate` and
   save the resulting ``(SUBJECT_ID, HADM_ID, ICD9_CODE)`` DataFrame.

References
----------
- *Accelerating Reproducible Research in Synthetic EHR Generation* (CHIL 2026)

Usage
-----
.. code-block:: bash

    # Full vocabulary (~6,955 ICD-9 codes) – recommended
    python ehr_generation_mimic3_transformer.py \\
        --mimic3_root /path/to/mimic-iii-clinical-database-1.4 \\
        --output_dir  ./synthetic_output

    # Optional: replicate the legacy 3-digit vocabulary
    python ehr_generation_mimic3_transformer.py \\
        --mimic3_root /path/to/mimic-iii \\
        --truncate_icd \\
        --output_dir  ./synthetic_output_3digit
"""

import argparse
import os

import torch

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.models.generators import EHRGPTBaseline, samples_to_sequences
from pyhealth.tasks import EHRGenerationMIMIC3


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # ------------------------------------------------------------------
    # STEP 1: Load MIMIC-III via PyHealth
    # ------------------------------------------------------------------
    print("\nSTEP 1: Loading MIMIC-III dataset ...")
    base_dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["diagnoses_icd"],
    )
    base_dataset.stats()

    # ------------------------------------------------------------------
    # STEP 2: Apply EHRGenerationMIMIC3 task
    # ------------------------------------------------------------------
    print("\nSTEP 2: Applying EHRGenerationMIMIC3 task ...")
    task = EHRGenerationMIMIC3(
        min_visits=args.min_visits,
        truncate_icd=args.truncate_icd,
    )
    sample_dataset = base_dataset.set_task(task)
    print(f"  Total patients: {len(sample_dataset)}")

    train_dataset, _, _ = split_by_patient(sample_dataset, [0.8, 0.1, 0.1])
    print(f"  Training patients: {len(train_dataset)}")

    # ------------------------------------------------------------------
    # STEP 3: Serialise to text sequences
    # ------------------------------------------------------------------
    print("\nSTEP 3: Serialising patient sequences ...")
    train_samples = list(train_dataset)
    text_data = samples_to_sequences(train_samples)
    max_len = max(len(seq.split()) for seq in text_data)
    print(f"  Max sequence length (tokens): {max_len}")

    # ------------------------------------------------------------------
    # STEP 4 - 6: Build tokeniser, initialise GPT-2, train
    # ------------------------------------------------------------------
    print("\nSTEP 4-6: Building tokeniser and training GPT-2 ...")
    model = EHRGPTBaseline(
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        max_seq_len=args.max_seq_len,
    )
    model.fit(
        sequences=text_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"  Vocabulary size : {len(model.tokenizer)}")
    print(f"  Model parameters: {n_params:.1f}M")

    # ------------------------------------------------------------------
    # STEP 7: Generate synthetic EHRs
    # ------------------------------------------------------------------
    print(f"\nSTEP 7: Generating {args.num_synthetic} synthetic patients ...")
    synthetic_df = model.generate(
        n_patients=args.num_synthetic,
        batch_size=args.gen_batch_size,
    )
    print(
        f"  Generated {synthetic_df['SUBJECT_ID'].nunique()} patients, "
        f"{synthetic_df.shape[0]} (patient, visit, code) rows"
    )

    out_csv = os.path.join(args.output_dir, "synthetic_ehr.csv")
    synthetic_df.to_csv(out_csv, index=False)
    print(f"  Synthetic data saved to: {out_csv}")


# -- CLI entry point -----------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 transformer for synthetic EHR generation (MIMIC-III).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        required=True,
        help="Path to the MIMIC-III root directory (raw CSV/CSV.GZ files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ehr_generation_output",
        help="Directory to save checkpoints and synthetic data.",
    )
    parser.add_argument(
        "--min_visits",
        type=int,
        default=1,
        help="Minimum valid admissions a patient must have.",
    )
    parser.add_argument(
        "--truncate_icd",
        action="store_true",
        default=False,
        help="Truncate ICD-9 codes to 3-digit prefixes.",
    )
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension.")
    parser.add_argument("--n_layer", type=int, default=8, help="Number of transformer layers.")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads.")
    parser.add_argument(
        "--max_seq_len", type=int, default=512, help="Maximum token sequence length."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument(
        "--num_synthetic", type=int, default=10000, help="Synthetic patients to generate."
    )
    parser.add_argument(
        "--gen_batch_size", type=int, default=512, help="Generation batch size."
    )
    main(parser.parse_args())
