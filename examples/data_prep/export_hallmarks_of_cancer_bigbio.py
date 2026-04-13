#!/usr/bin/env python3
"""Export BigBio Hallmarks of Cancer to a CSV for :class:`HallmarksOfCancerDataset`.

This script is **not** run by PyHealth tests. Use it once to materialize data
under a directory you pass to ``HallmarksOfCancerDataset(root=...)``.

Requires::

    pip install datasets pandas

Data source (GPL-3.0): https://huggingface.co/datasets/bigbio/hallmarks_of_cancer
Use config ``hallmarks_of_cancer_bigbio_text`` for sentence-level ``text`` and
``labels`` fields.

Example::

    python examples/data_prep/export_hallmarks_of_cancer_bigbio.py \\
        --output-dir ~/data/hallmarks_of_cancer
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write hallmarks_of_cancer.csv into",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset  # noqa: WPS433 (runtime optional dep)

    ds = load_dataset(
        "bigbio/hallmarks_of_cancer",
        "hallmarks_of_cancer_bigbio_text",
    )
    split_map = {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }

    rows: list[dict[str, str]] = []
    for split_name, hf_split in split_map.items():
        for i, rec in enumerate(ds[hf_split]):
            sid = str(rec.get("id", f"{split_name}_{i}"))
            doc = str(rec.get("document_id", ""))
            text = rec.get("text", "")
            if not isinstance(text, str):
                text = str(text)
            labels = rec.get("labels", [])
            if not isinstance(labels, list):
                labels = [str(labels)]
            labels_str = "##".join(str(x) for x in labels)
            rows.append(
                {
                    "sentence_id": sid,
                    "document_id": doc,
                    "text": text.replace("\r\n", "\n").replace("\r", "\n"),
                    "labels": labels_str,
                    "split": split_name,
                }
            )

    out_path = args.output_dir / "hallmarks_of_cancer.csv"
    import pandas as pd  # noqa: WPS433

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
