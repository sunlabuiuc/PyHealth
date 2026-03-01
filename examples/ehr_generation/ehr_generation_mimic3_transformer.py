"""
EHR Generation with a GPT-2 style Transformer on MIMIC-III (PyHealth)
======================================================================

This example trains a GPT-2 style decoder-only transformer to synthesise
longitudinal patient EHR sequences consisting of ICD-9 diagnosis codes.

The pipeline:

1. Load MIMIC-III via **PyHealth** and apply the ``EHRGenerationMIMIC3`` task
   to obtain per-patient nested visit sequences.
2. Serialise the nested sequences into plain text using ``VISIT_DELIM``
   separators (e.g. ``"250.00 401.9 VISIT_DELIM 272.0 428.0"``).
3. Train a word-level GPT-2 model on the serialised text.
4. Sample synthetic text sequences and deserialise them back to a long-form
   ``(SUBJECT_ID, HADM_ID, ICD9_CODE)`` DataFrame for downstream evaluation.

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

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from torch.utils.data import Dataset
from tqdm import trange
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)

from pyhealth.datasets import MIMIC3Dataset, split_by_patient
from pyhealth.tasks import EHRGenerationMIMIC3

# ── Constants ─────────────────────────────────────────────────────────────────

VISIT_DELIM = "VISIT_DELIM"


# ── 1. Sequence helpers ────────────────────────────────────────────────────────


def samples_to_sequences(samples: list) -> list[str]:
    """Convert PyHealth ``EHRGenerationMIMIC3`` samples to text sequences.

    Each sample's ``conditions`` field is a ``List[List[str]]`` (visits × codes).
    Adjacent visits are joined by ``VISIT_DELIM`` so the full patient history
    becomes a space-separated string.

    Args:
        samples: List of dicts with at least a ``"conditions"`` key.

    Returns:
        A list of strings, one per patient, e.g.
        ``"250.00 401.9 VISIT_DELIM 272.0 428.0 VISIT_DELIM 250.00"``.
    """
    sequences = []
    for sample in samples:
        visit_texts = [" ".join(visit_codes) for visit_codes in sample["conditions"]]
        sequences.append(f" {VISIT_DELIM} ".join(visit_texts))
    return sequences


def sequences_to_dataframe(sequences: list[str]) -> pd.DataFrame:
    """Deserialise generated text sequences back to long-form ``(SUBJECT_ID, HADM_ID, ICD9_CODE)``.

    Assigns synthetic sequential identifiers; the real MIMIC-III IDs are not
    recovered (generation is unconditional).

    Args:
        sequences: Generated text sequences from the transformer.

    Returns:
        A ``pd.DataFrame`` with columns ``SUBJECT_ID``, ``HADM_ID``, ``ICD9_CODE``.
    """
    rows = []
    for subj_idx, seq in enumerate(sequences):
        for hadm_idx, visit_str in enumerate(seq.strip().split(VISIT_DELIM)):
            for code in visit_str.strip().split():
                if code:
                    rows.append(
                        {
                            "SUBJECT_ID": subj_idx,
                            "HADM_ID": hadm_idx,
                            "ICD9_CODE": code,
                        }
                    )
    return pd.DataFrame(rows)


# ── 2. PyTorch Dataset ─────────────────────────────────────────────────────────


class EHRTextDataset(Dataset):
    """Tokenises a list of EHR text sequences for causal language modelling.

    Args:
        sequences: Plain-text patient sequences (one string per patient).
        tokenizer: A HuggingFace ``PreTrainedTokenizerFast``.
        max_length: Maximum token length; longer sequences are truncated.
    """

    def __init__(
        self,
        sequences: list[str],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
    ) -> None:
        self.input_ids = []
        for txt in sequences:
            enc = tokenizer(txt, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(enc["input_ids"]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict:
        return {"input_ids": self.input_ids[idx], "labels": self.input_ids[idx]}


# ── 3. Tokeniser builder ───────────────────────────────────────────────────────


def build_tokenizer(text_data: list[str]) -> PreTrainedTokenizerFast:
    """Build and train a word-level tokeniser on the EHR text corpus.

    Special tokens:
    * ``[UNK]``  – unknown token
    * ``[PAD]``  – padding
    * ``[BOS]``  – beginning-of-sequence
    * ``[EOS]``  – end-of-sequence

    The ``VISIT_DELIM`` delimiter token is treated as a regular vocabulary
    word so that its visit-boundary semantics are learned by the model.

    Args:
        text_data: List of space-separated code sequences.

    Returns:
        A ``PreTrainedTokenizerFast`` wrapping the trained word-level tokeniser.
    """
    tokenizer_obj = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tokenizer_obj.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    word_trainer = trainers.WordLevelTrainer(special_tokens=special_tokens)
    tokenizer_obj.train_from_iterator(text_data, trainer=word_trainer)

    tokenizer_obj.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer_obj.token_to_id("[BOS]")),
            ("[EOS]", tokenizer_obj.token_to_id("[EOS]")),
        ],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )


# ── 4. Main pipeline ───────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # STEP 1: Load MIMIC-III via PyHealth
    # ------------------------------------------------------------------
    print("\nSTEP 1: Loading MIMIC-III dataset …")
    base_dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["diagnoses_icd"],
    )
    base_dataset.stats()

    # ------------------------------------------------------------------
    # STEP 2: Apply EHRGenerationMIMIC3 task
    # ------------------------------------------------------------------
    print("\nSTEP 2: Applying EHRGenerationMIMIC3 task …")
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
    print("\nSTEP 3: Serialising patient sequences …")
    train_samples = list(train_dataset)
    text_data = samples_to_sequences(train_samples)
    max_len = max(len(seq.split()) for seq in text_data)
    print(f"  Max sequence length: {max_len} tokens")

    # ------------------------------------------------------------------
    # STEP 4: Build tokeniser
    # ------------------------------------------------------------------
    print("\nSTEP 4: Building word-level tokeniser …")
    tokenizer = build_tokenizer(text_data)
    print(f"  Vocabulary size: {len(tokenizer)}")

    train_torch_dataset = EHRTextDataset(text_data, tokenizer, max_length=args.max_seq_len)

    # ------------------------------------------------------------------
    # STEP 5: Initialise GPT-2 style decoder model
    # ------------------------------------------------------------------
    print("\nSTEP 5: Initialising GPT-2 model …")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.max_seq_len,
        n_ctx=args.max_seq_len,
        n_embd=512,
        n_layer=8,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model parameters: {num_params:.1f}M")

    # ------------------------------------------------------------------
    # STEP 6: Train
    # ------------------------------------------------------------------
    print("\nSTEP 6: Training …")
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=50,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        use_cpu=not torch.cuda.is_available(),
        save_strategy="epoch",
    )

    hf_trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_torch_dataset,
    )
    hf_trainer.train()

    model_save_path = os.path.join(args.output_dir, "transformer_ehr_model")
    hf_trainer.save_model(model_save_path)
    print(f"  Model saved to: {model_save_path}")

    # ------------------------------------------------------------------
    # STEP 7: Generate synthetic EHRs
    # ------------------------------------------------------------------
    print(f"\nSTEP 7: Generating {args.num_synthetic} synthetic patients …")
    model.eval()

    all_syn: list[pd.DataFrame] = []
    start_subj_id = 0
    for batch_start in trange(0, args.num_synthetic, args.gen_batch_size):
        batch_end = min(batch_start + args.gen_batch_size, args.num_synthetic)
        bsz = batch_end - batch_start

        batch_input_ids = torch.tensor(
            [[tokenizer.bos_token_id]] * bsz, device=device
        )
        with torch.no_grad():
            generated = model.generate(
                batch_input_ids,
                max_new_tokens=args.max_seq_len,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = [
            tokenizer.decode(seq, skip_special_tokens=True) for seq in generated
        ]
        batch_df = sequences_to_dataframe(decoded)
        batch_df["SUBJECT_ID"] += start_subj_id
        start_subj_id += bsz
        all_syn.append(batch_df)

    synthetic_df = pd.concat(all_syn, ignore_index=True)
    print(f"  Generated {synthetic_df['SUBJECT_ID'].nunique()} patients, "
          f"{synthetic_df.shape[0]} (patient, visit, code) rows")

    out_csv = os.path.join(args.output_dir, "synthetic_ehr.csv")
    synthetic_df.to_csv(out_csv, index=False)
    print(f"  Synthetic data saved to: {out_csv}")


# ── CLI entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a GPT-2 transformer for synthetic EHR generation (MIMIC-III).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        required=True,
        help="Path to the MIMIC-III root directory containing raw CSV/CSV.GZ files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ehr_generation_output",
        help="Directory to save the trained model and synthetic data.",
    )
    parser.add_argument(
        "--min_visits",
        type=int,
        default=1,
        help="Minimum number of valid admissions a patient must have.",
    )
    parser.add_argument(
        "--truncate_icd",
        action="store_true",
        default=False,
        help="Truncate ICD-9 codes to 3-digit prefixes (reduces vocab to ~1,071 codes).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum token sequence length.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size.",
    )
    parser.add_argument(
        "--num_synthetic",
        type=int,
        default=10000,
        help="Number of synthetic patients to generate.",
    )
    parser.add_argument(
        "--gen_batch_size",
        type=int,
        default=512,
        help="Generation batch size.",
    )
    main(parser.parse_args())
