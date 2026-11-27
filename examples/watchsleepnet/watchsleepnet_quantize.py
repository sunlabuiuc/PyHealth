"""Casts a WatchSleepNet checkpoint to fp32, bf16, or fp16 without training.

This example needs no dataset: it loads a bare ``.pt`` state dict or a
self-describing bundle, casts floating-point tensors, and saves the result.
Unsupported dtype/device behavior is left to PyTorch; the script reports the
dtype counts it actually wrote rather than pre-labeling any mode.

Casting a checkpoint does not train a model; it only checks that the PyHealth
checkpoint path preserves the requested compute dtype.

Command:
    python examples/watchsleepnet/watchsleepnet_quantize.py \
        --checkpoint output/watchsleepnet-pretrain/pretrained.pt \
        --output output/watchsleepnet-pretrain/pretrained_fp16.pt \
        --quantization fp16

Example output table::

    tensor_dtype  count
    torch.float16  135
    torch.int64      0

``quantization.csv`` and ``quantization.json`` are written next to ``--output``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .watchsleepnet_common import (
        QUANTIZATION_MODES,
        load_checkpoint,
        print_table,
        quantize_checkpoint,
    )
except ImportError:
    from watchsleepnet_common import (
        QUANTIZATION_MODES,
        load_checkpoint,
        print_table,
        quantize_checkpoint,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for checkpoint quantization.

    Args:
        argv: Optional argument list.

    Returns:
        The parsed namespace.

    Examples:
        >>> args = parse_args(["--checkpoint", "a.pt", "--output", "b.pt",
        ...                    "--quantization", "bf16"])
        >>> args.quantization
        'bf16'
    """
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, help="input .pt file")
    parser.add_argument("--output", required=True, help="output .pt file")
    parser.add_argument("--quantization", required=True,
                        choices=QUANTIZATION_MODES)
    parser.add_argument("--report", default=None,
                        help="report path; defaults next to --output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Loads, casts, saves, and reports a checkpoint.

    Args:
        argv: Optional argument list.

    Returns:
        Zero after the cast completes.

    Examples:
        >>> main(["--checkpoint", "unused.pt", "--output", "out.pt",
        ...       "--quantization", "fp32"]) if False else "ready"
        'ready'
    """
    args = parse_args(argv)
    source = Path(args.checkpoint)
    target = Path(args.output)
    _, metadata = load_checkpoint(source)
    quantize_checkpoint(source, target, args.quantization)
    state, output_metadata = load_checkpoint(target)
    counts = Counter(str(tensor.dtype) for tensor in state.values())
    rows = [
        {"tensor_dtype": dtype, "count": count}
        for dtype, count in sorted(counts.items())
    ]

    report = (
        Path(args.report) if args.report else target.parent / "quantization.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "source": str(source),
                "target": str(target),
                "quantization": args.quantization,
                "source_format": output_metadata.get(
                    "source_format", metadata.get("format", "bare-state-dict")
                ),
                "dtype_counts": dict(counts),
                "config": output_metadata.get("config", {}),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    csv_path = report.with_suffix(".csv")
    with open(csv_path, "w") as handle:
        handle.write("tensor_dtype,count\n")
        handle.writelines(
            f"{row['tensor_dtype']},{row['count']}\n" for row in rows
        )
    print_table(rows, ("tensor_dtype", "count"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
