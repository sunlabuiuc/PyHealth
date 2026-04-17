"""
GPT-2 Baseline for Synthetic EHR Generation
============================================

This module provides a self-contained GPT-2 decoder-only language model for
generating synthetic longitudinal EHR sequences composed of ICD-9 diagnosis
codes.

Design
------
Patient histories are first serialised as space-separated code sequences where
consecutive visits are separated by the special ``VISIT_DELIM`` token:

    ``"250.00 401.9 VISIT_DELIM 272.0 428.0 VISIT_DELIM 250.00"``

A word-level HuggingFace tokeniser is then trained on this corpus, and a
GPT-2 causal language model is fine-tuned on the resulting token IDs.  At
inference time, sequences are sampled autoregressively and deserialised back
to a long-form ``(SUBJECT_ID, HADM_ID, ICD9_CODE)`` DataFrame.

References
----------
- *Accelerating Reproducible Research in Synthetic EHR Generation* (CHIL 2026)

Typical usage
-------------
.. code-block:: python

    from pyhealth.models.generators import EHRGPTBaseline
    from pyhealth.tasks.ehr_generation import samples_to_sequences

    model = EHRGPTBaseline(n_embd=256, n_layer=4, n_head=4)
    model.fit(text_sequences, output_dir="./checkpoints", epochs=20)
    synthetic_df = model.generate(n_patients=1000)
"""

import os
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
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

__all__ = [
    "VISIT_DELIM",
    "samples_to_sequences",
    "sequences_to_dataframe",
    "build_tokenizer",
    "EHRTextDataset",
    "EHRGPTBaseline",
]

# ── Constants ──────────────────────────────────────────────────────────────────

VISIT_DELIM = "VISIT_DELIM"


# ── Sequence helpers ───────────────────────────────────────────────────────────


def samples_to_sequences(samples: list) -> list[str]:
    """Convert ``EHRGenerationMIMIC3`` samples to VISIT_DELIM-delimited text.

    Each sample's ``conditions`` field is a ``List[List[str]]`` (visits × codes).
    Adjacent visits are joined by ``VISIT_DELIM`` so the full patient history
    becomes a single space-separated string.

    Args:
        samples: List of dicts with at least a ``"conditions"`` key.

    Returns:
        One string per patient, e.g.
        ``"250.00 401.9 VISIT_DELIM 272.0 428.0 VISIT_DELIM 250.00"``.

    Examples:
        >>> samples = [{"conditions": [["250.00", "401.9"], ["272.0"]]}]
        >>> samples_to_sequences(samples)
        ['250.00 401.9 VISIT_DELIM 272.0']
    """
    sequences: list[str] = []
    for sample in samples:
        visit_texts = [" ".join(visit_codes) for visit_codes in sample["conditions"]]
        sequences.append(f" {VISIT_DELIM} ".join(visit_texts))
    return sequences


def sequences_to_dataframe(sequences: list[str]) -> pd.DataFrame:
    """Deserialise generated text sequences to long-form EHR rows.

    Assigns synthetic sequential identifiers; original MIMIC-III IDs are not
    preserved (generation is unconditional).

    Args:
        sequences: Generated text sequences from :meth:`EHRGPTBaseline.generate`.

    Returns:
        A ``pd.DataFrame`` with columns ``SUBJECT_ID``, ``HADM_ID``,
        ``ICD9_CODE``.

    Examples:
        >>> sequences_to_dataframe(["250.00 VISIT_DELIM 401.9"])
           SUBJECT_ID  HADM_ID ICD9_CODE
        0           0        0    250.00
        1           0        1     401.9
    """
    rows: list[dict] = []
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


# ── Tokeniser ──────────────────────────────────────────────────────────────────


def build_tokenizer(text_data: list[str]) -> PreTrainedTokenizerFast:
    """Build and train a word-level tokeniser on an EHR text corpus.

    Uses the HuggingFace ``tokenizers`` library.  Special tokens:

    * ``[UNK]``  – unknown token
    * ``[PAD]``  – padding
    * ``[BOS]``  – beginning-of-sequence
    * ``[EOS]``  – end-of-sequence

    ``VISIT_DELIM`` is treated as a regular vocabulary word so the model
    learns its visit-boundary semantics.

    Args:
        text_data: List of space-separated code sequences (one per patient).

    Returns:
        A ``PreTrainedTokenizerFast`` wrapping the trained word-level model.

    Note:
        The ``Whitespace`` pre-tokeniser splits on punctuation, so ICD-9 codes
        such as ``"250.00"`` are stored as the sub-tokens ``["250", ".", "00"]``.
        This is intentional: it drastically reduces the vocabulary size while
        preserving code structure.
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


# ── PyTorch Dataset ────────────────────────────────────────────────────────────


class EHRTextDataset(Dataset):
    """Tokenises EHR text sequences for causal language-model training.

    Each sequence is tokenised, truncated/padded to ``max_length``, and stored
    as a fixed-length ``LongTensor``.  The ``labels`` field mirrors
    ``input_ids`` so the HuggingFace ``Trainer`` can compute the standard
    next-token prediction loss.

    Args:
        sequences: Plain-text patient sequences (one string per patient).
        tokenizer: A trained :class:`~transformers.PreTrainedTokenizerFast`.
        max_length: Token budget; longer sequences are right-truncated.

    Examples:
        >>> from pyhealth.models.generators import build_tokenizer, EHRTextDataset
        >>> tok = build_tokenizer(["250.00 VISIT_DELIM 401.9"])
        >>> ds  = EHRTextDataset(["250.00 VISIT_DELIM 401.9"], tok, max_length=16)
        >>> len(ds)
        1
        >>> ds[0]["input_ids"].shape
        torch.Size([16])
    """

    def __init__(
        self,
        sequences: list[str],
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 512,
    ) -> None:
        self.input_ids: list[torch.Tensor] = []
        for txt in sequences:
            enc = tokenizer(
                txt,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(enc["input_ids"]))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx]
        return {"input_ids": ids, "labels": ids}


# ── Main model class ───────────────────────────────────────────────────────────


class EHRGPTBaseline(nn.Module):
    """GPT-2 decoder-only language model for synthetic EHR generation.

    Wraps a HuggingFace ``GPT2LMHeadModel`` and exposes a high-level API
    (:meth:`fit`, :meth:`generate`) that matches the training pipeline
    described in *Accelerating Reproducible Research in Synthetic EHR
    Generation* (CHIL 2026).

    Architecture
    ------------
    * Word-level ICD-9 tokeniser (``VISIT_DELIM`` as vocabulary entry)
    * GPT-2 transformer decoder with configurable depth and width
    * Autoregressive next-token prediction objective

    Args:
        n_embd: Embedding and hidden dimension. Default: 512.
        n_layer: Number of transformer decoder layers. Default: 8.
        n_head: Number of self-attention heads. Default: 8.
        max_seq_len: Maximum token sequence length. Default: 512.

    Attributes:
        tokenizer: The fitted :class:`~transformers.PreTrainedTokenizerFast`
            (``None`` until :meth:`fit` is called).
        model: The underlying :class:`~transformers.GPT2LMHeadModel`
            (``None`` until :meth:`fit` is called).

    Examples:
        .. code-block:: python

            from pyhealth.models.generators import EHRGPTBaseline, samples_to_sequences

            gpt = EHRGPTBaseline(n_embd=256, n_layer=4, n_head=4)
            gpt.fit(text_sequences, output_dir="./ckpt", epochs=10, batch_size=32)
            df = gpt.generate(n_patients=500)
    """

    def __init__(
        self,
        n_embd: int = 512,
        n_layer: int = 8,
        n_head: int = 8,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.max_seq_len = max_seq_len

        # Populated by fit()
        self.tokenizer: Optional[PreTrainedTokenizerFast] = None
        self.model: Optional[GPT2LMHeadModel] = None

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict:
        """Run a forward pass through the GPT-2 model.

        Delegates directly to :class:`~transformers.GPT2LMHeadModel`.

        Args:
            input_ids: Token ID tensor of shape ``(batch, seq_len)``.
            **kwargs: Additional arguments forwarded to GPT2LMHeadModel.

        Returns:
            The ``CausalLMOutputWithCrossAttentions`` dict-like object from
            HuggingFace (contains ``logits``, ``loss`` when ``labels`` are
            supplied, etc.).

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before forward().")
        return self.model(input_ids, **kwargs)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        sequences: list[str],
        output_dir: str = "./ehr_gpt_output",
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
    ) -> "EHRGPTBaseline":
        """Build the tokeniser, initialise GPT-2, and train on ``sequences``.

        This method is idempotent: calling it again re-initialises the
        tokeniser and model from scratch.

        Args:
            sequences: List of VISIT_DELIM-delimited patient text sequences
                produced by :func:`samples_to_sequences`.
            output_dir: Directory for HuggingFace ``Trainer`` checkpoints.
            epochs: Training epochs.
            batch_size: Per-device training batch size.
            learning_rate: Peak learning rate (cosine schedule).
            warmup_steps: Linear warm-up steps.

        Returns:
            ``self`` (fluent API).
        """
        os.makedirs(output_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── tokeniser ─────────────────────────────────────────────────────────
        self.tokenizer = build_tokenizer(sequences)

        # ── dataset ───────────────────────────────────────────────────────────
        train_ds = EHRTextDataset(sequences, self.tokenizer, max_length=self.max_seq_len)

        # ── model ─────────────────────────────────────────────────────────────
        config = GPT2Config(
            vocab_size=len(self.tokenizer),
            n_positions=self.max_seq_len,
            n_ctx=self.max_seq_len,
            n_embd=self.n_embd,
            n_layer=self.n_layer,
            n_head=self.n_head,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        self.model = GPT2LMHeadModel(config).to(device)

        # ── training ──────────────────────────────────────────────────────────
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "checkpoints"),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            logging_steps=50,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            use_cpu=not torch.cuda.is_available(),
            save_strategy="epoch",
        )

        hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_ds,
        )
        hf_trainer.train()

        # Persist model and tokeniser side-by-side
        model_dir = os.path.join(output_dir, "gpt_ehr_model")
        hf_trainer.save_model(model_dir)
        self.tokenizer.save_pretrained(model_dir)

        return self

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    def generate(
        self,
        n_patients: int = 1000,
        batch_size: int = 512,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> pd.DataFrame:
        """Sample synthetic EHR sequences and return a long-form DataFrame.

        Args:
            n_patients: Number of synthetic patients to generate.
            batch_size: Generation batch size (GPU memory permitting).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling probability threshold.

        Returns:
            A ``pd.DataFrame`` with columns ``SUBJECT_ID``, ``HADM_ID``,
            ``ICD9_CODE``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Call fit() before generate().")

        device = next(self.model.parameters()).device
        self.model.eval()

        all_dfs: list[pd.DataFrame] = []
        start_subj = 0

        for batch_start in trange(0, n_patients, batch_size, desc="Generating"):
            bsz = min(batch_size, n_patients - batch_start)
            prompt = torch.tensor(
                [[self.tokenizer.bos_token_id]] * bsz, device=device
            )
            with torch.no_grad():
                generated = self.model.generate(
                    prompt,
                    max_new_tokens=self.max_seq_len,
                    do_sample=True,
                    top_k=top_k,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            decoded = [
                self.tokenizer.decode(seq, skip_special_tokens=True)
                for seq in generated
            ]
            batch_df = sequences_to_dataframe(decoded)
            batch_df["SUBJECT_ID"] += start_subj
            start_subj += bsz
            all_dfs.append(batch_df)

        return pd.concat(all_dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the GPT-2 model weights and tokeniser to ``path``.

        Args:
            path: Directory to save into (created if absent).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Nothing to save – call fit() first.")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, **init_kwargs) -> "EHRGPTBaseline":
        """Load a previously saved :class:`EHRGPTBaseline` from ``path``.

        Args:
            path: Directory created by :meth:`save`.
            **init_kwargs: Forwarded to ``__init__`` (overrides defaults for
                ``n_embd``, ``n_layer``, ``n_head``, ``max_seq_len``).

        Returns:
            A fully initialised :class:`EHRGPTBaseline` ready for
            :meth:`generate`.
        """
        instance = cls(**init_kwargs)
        instance.tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        instance.model = GPT2LMHeadModel.from_pretrained(path)
        return instance

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        fitted = self.model is not None
        return (
            f"EHRGPTBaseline("
            f"n_embd={self.n_embd}, n_layer={self.n_layer}, "
            f"n_head={self.n_head}, max_seq_len={self.max_seq_len}, "
            f"fitted={fitted})"
        )
