"""Extract per-sample BulkRNABert embeddings from a pre-trained checkpoint.

Given a BulkRNABert checkpoint directory (``params.pt`` + ``config.json``) and
the preprocessed expression CSV used during pre-training, this script runs a
no-mask forward pass through the encoder and saves the mean-pooled last-layer
output as a ``.npy`` matrix of shape ``(n_samples, embed_dim)``. Row ``i`` of
the output corresponds to row ``i`` of the input CSV, so a downstream
classifier can be trained with
:func:`pyhealth.datasets.load_tcga_cancer_classification_5cohort`.

Usage example:

    python examples/tcga_rnaseq_extract_embeddings_bulk_rna_bert.py \\
        --ckpt-dir output/bulk_rna_bert_pretrain_tcga_discrete_refinit/step_600 \\
        --csv-path path/to/tcga_preprocessed.csv \\
        --output-path output/embeddings/tcga_discrete_refinit_step600.npy \\
        --batch-size 32

Previous: ``tcga_rnaseq_mlm_bulk_rna_bert.py``. Next:
``tcga_cancer_classification_5cohort_bulk_rna_bert.py``.

Author: Yohei Shibata (NetID: yoheis2)
Paper: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
       (Gelard et al., PMLR 259, 2025)
Paper link: https://proceedings.mlr.press/v259/gelard25a.html
Description: CLI that loads a BulkRNABert pre-training checkpoint and writes
    a ``(n_samples, embed_dim)`` mean-pooled embedding matrix to ``.npy`` for
    the downstream head-only classifier.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhealth.models import (  # noqa: E402
    BulkRNABert,
    BulkRNABertConfig,
    load_expression_csv,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--ckpt-dir", type=Path, required=True,
                   help="Directory containing params.pt and config.json.")
    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--output-path", type=Path, required=True,
                   help="Destination .npy file for the (N, embed_dim) matrix.")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--already-log-normalized", action="store_true",
        help="Set if CSV values are already log10(TPM+1) rather than raw TPM.",
    )
    p.add_argument("--device", type=str, default=None,
                   help="Override device (e.g. 'cpu' or 'cuda:0').")
    return p.parse_args()


def _load_model(ckpt_dir: Path) -> tuple[BulkRNABert, float | None]:
    config_path = ckpt_dir / "config.json"
    params_path = ckpt_dir / "params.pt"
    if not config_path.exists() or not params_path.exists():
        raise FileNotFoundError(
            f"ckpt_dir must contain config.json and params.pt (got {ckpt_dir})"
        )
    with open(config_path) as f:
        cfg_dict = json.load(f)
    # normalization_factor is a preprocessing scalar, not a model-config
    # field; saved alongside the PyHealth config at checkpoint time.
    norm_factor = cfg_dict.pop("normalization_factor", None)
    cfg = BulkRNABertConfig(**cfg_dict)
    model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
    state = torch.load(params_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model, norm_factor


def main() -> None:
    args = _parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ckpt] loading {args.ckpt_dir}", flush=True)
    model, norm_factor = _load_model(args.ckpt_dir)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(
        f"[data] loading {args.csv_path} (mode={model.config.expression_mode}"
        f"{f', norm_factor={norm_factor}' if norm_factor is not None else ''}) ...",
        flush=True,
    )
    t0 = time.time()
    load_kwargs = dict(
        mode=model.config.expression_mode,
        n_bins=model.config.n_bins,
        already_log_normalized=args.already_log_normalized,
    )
    if norm_factor is not None:
        load_kwargs["normalization_factor"] = norm_factor
    data, _ = load_expression_csv(args.csv_path, **load_kwargs)
    print(
        f"[data] loaded tensor {tuple(data.shape)} dtype={data.dtype} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )

    n_samples = data.shape[0]
    embed_dim = model.config.embed_dim
    out = np.empty((n_samples, embed_dim), dtype=np.float32)

    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, n_samples, args.batch_size):
            end = min(start + args.batch_size, n_samples)
            batch = data[start:end]
            emb = model.encode(batch)
            out[start:end] = emb.detach().to("cpu", dtype=torch.float32).numpy()
            if start // args.batch_size % 20 == 0:
                print(
                    f"[encode] {end}/{n_samples} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    np.save(args.output_path, out)
    print(
        f"[done] wrote {args.output_path} shape={out.shape} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
