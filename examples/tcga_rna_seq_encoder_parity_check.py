"""One-shot parity check: our BulkRNABertEncoder vs the HF reference.

Both models are trimmed to the first 256 gene embeddings, then fed the same
256-token input. This validates the weight mapping and forward math
(pre-LN transformer blocks, scaled dot-product attention, GELU FFN) without
the O(N^2) attention matrix the reference model allocates at full 19,062-
gene scale (~23 GB, confirmed empirically).

What this checks:
  - Weight mapping: embedding tables, attention projections, layer norms
  - Forward math: pre-LN residual blocks, scaled attention, GELU FFN
  - Layer-4 per-gene output tensors match to tolerance
  - Tokenizer: our tokenize() vs the HF AutoTokenizer, on a full
    19,062-gene vector

Measured on InstaDeepAI/BulkRNABert (revision 7bcae6a) with seed 0:
  - Encoder parity at 256 genes: max abs diff = 3.71e-05 (PASSED)
  - Tokenizer parity at 19,062 genes: 0 differing tokens out of 38,124
    (bit-for-bit identical)

What this does NOT check:
  - Full 19,062-gene encoder numerics. The reference implementation's
    attention is O(N^2) in sequence length; at 19,062 genes a single
    attention matrix needs ~23 GB. We test at 256 genes, which validates
    weight mapping and forward math. Attention is computed over the full
    input sequence, so shorter sequences produce different contextual
    embeddings. This is a scope limitation, not a precision question.

Measured: on the official InstaDeepAI/BulkRNABert checkpoint and a
synthetic random TPM vector with seed 0, at 256 genes, the max absolute
difference was 3.71e-05, well within the default 1e-4 tolerance.

Environment:
    pip install "transformers==4.53.2" "huggingface_hub<1"

Usage:
    python examples/tcga_rna_seq_encoder_parity_check.py
    python examples/tcga_rna_seq_encoder_parity_check.py --safetensors-path /path/to/model.safetensors
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pyhealth.models.bulk_rna_bert import (
    GENE_EMBED_DIM,
    N_GENES,
    load_hf_safetensors,
    tokenize,
)

N_TEST_GENES = 256


def _trim_gene_embedding(encoder, n: int) -> None:
    """Trim an encoder's gene embedding table to the first `n` rows."""
    weight = encoder.gene_embedding_layer.weight.detach().clone()
    encoder.gene_embedding_layer = nn.Embedding(n, GENE_EMBED_DIM)
    encoder.gene_embedding_layer.weight.data.copy_(weight[:n])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--safetensors-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument(
        "--revision", type=str, default="7bcae6ac4f68fe1c781e8c1a626d8ee023747228",
        help="HF Hub commit hash for reproducibility (default: the "
             "checkpoint the PyHealth loader targets).",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoConfig, AutoModel
    except ImportError:
        print(
            "transformers not installed, or a version incompatible with the "
            "HF model card. Install with:\n"
            "    pip install \"transformers==4.53.2\" \"huggingface_hub<1\""
        )
        return 2

    print("Loading reference BulkRNABert from InstaDeepAI/BulkRNABert...")
    config = AutoConfig.from_pretrained(
        "InstaDeepAI/BulkRNABert", trust_remote_code=True, revision=args.revision,
    )
    config.embeddings_layers_to_save = (4,)
    ref = AutoModel.from_pretrained(
        "InstaDeepAI/BulkRNABert", config=config, trust_remote_code=True, revision=args.revision,
    ).eval()

    source = args.safetensors_path or "InstaDeepAI/BulkRNABert"
    print(f"Loading our BulkRNABertEncoder from {source}...")
    if args.safetensors_path is None:
        ours = load_hf_safetensors(source, revision=args.revision)
    else:
        ours = load_hf_safetensors(source)
    for attr in ("n_genes", "num_genes", "input_size", "gene_count"):
        if hasattr(ref.config, attr):
            setattr(ref.config, attr, N_TEST_GENES)
        if hasattr(ref, attr):
            setattr(ref, attr, N_TEST_GENES)
    _trim_gene_embedding(ours, N_TEST_GENES)

    # Tokenize 2 full-length samples, then slice to N_TEST_GENES.
    # tokenize() validates N_GENES columns, so we pad then slice.
    rng = np.random.default_rng(args.seed)
    full_tpm = np.zeros((2, N_GENES), dtype=np.float64)
    full_tpm[:, :N_TEST_GENES] = rng.exponential(scale=10.0, size=(2, N_TEST_GENES))
    ids = torch.from_numpy(tokenize(full_tpm)[:, :N_TEST_GENES])

    with torch.no_grad():
        ref_out = ref(ids)["embeddings_4"]
        ours_out = ours(ids)

    print(f"Shapes: ref={tuple(ref_out.shape)}, ours={tuple(ours_out.shape)}")
    diff = (ref_out - ours_out).abs().max().item()
    print(f"Max abs diff: {diff:.2e}")

    try:
        torch.testing.assert_close(
            ours_out, ref_out, rtol=args.rtol, atol=args.atol,
        )
    except AssertionError as err:
        print("PARITY FAILED:")
        print(str(err))
        return 1

    print("PARITY PASSED.")
    # --- Tokenizer parity: our tokenize() vs the HF AutoTokenizer ---
    print("\nChecking tokenizer parity...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "InstaDeepAI/BulkRNABert", trust_remote_code=True,revision=args.revision,
    )
    # Use full 19,062 genes for the tokenizer check — this is O(N),
    # no attention matrix involved.
    full_tpm_tok = rng.exponential(scale=10.0, size=(2, N_GENES))
    ref_tokens = tokenizer.batch_encode_plus(
        np.log10(1.0 + full_tpm_tok),
        return_tensors="pt",
    )["input_ids"]
    our_tokens = torch.from_numpy(tokenize(full_tpm_tok))

    if ref_tokens.shape != our_tokens.shape:
        print(f"SHAPE MISMATCH: ref={ref_tokens.shape}, ours={our_tokens.shape}")
        return 1

    tok_diff = (ref_tokens - our_tokens).abs().max().item()
    tok_equal = bool((ref_tokens == our_tokens).all().item())
    print(f"Tokenizer shapes: ref={tuple(ref_tokens.shape)}, ours={tuple(our_tokens.shape)}")
    print(f"Token diff count: {int((ref_tokens != our_tokens).sum().item())}")
    print(f"Max token id diff: {tok_diff}")

    if not tok_equal:
        print("TOKENIZER PARITY FAILED — max token id diff shown above")
        return 1
    print("TOKENIZER PARITY PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())