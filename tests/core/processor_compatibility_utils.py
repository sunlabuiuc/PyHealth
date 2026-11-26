"""Builds the processor and model compatibility table in the docs.

For each input processor, this script makes a tiny binary-label dataset
with one field of that type. It then records which layer ``EmbeddingModel``
creates for the field. Finally it tries one forward and backward pass of
each general-purpose model on it. The result is written to
``docs/api/processors/compatibility.rst``, and
``tests/core/test_processor_compatibility.py`` checks that every pairing
the page marks as working still runs.

Usage (from the repository root)::

    python -m tests.core.processor_compatibility_utils            # rewrite the page
    python -m tests.core.processor_compatibility_utils --errors   # also print errors

Examples:
    >>> from tests.core.processor_compatibility_utils import (  # doctest: +SKIP
    ...     runs, sample_dataset)
    >>> ds = sample_dataset("timeseries")  # doctest: +SKIP
    >>> runs(ds, "RNN")  # doctest: +SKIP
    (True, '')
"""

from __future__ import annotations

import argparse
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

PAGE = Path(__file__).resolve().parents[2] / "docs/api/processors/compatibility.rst"

# General-purpose models that take any dataset with default arguments.
MODELS = ["LogisticRegression", "MLP", "RNN", "CNN", "TCN", "Transformer",
          "RETAIN", "AdaCare", "ConCare", "StageNet", "GRASP", "Agent", "Deepr"]

_T0 = datetime(2020, 1, 1)  # noqa: DTZ001 - PyHealth timestamps are naive
_CODES = list("abcdef")


def _value(key: str, i: int):
    """The i-th sample's value for a field of processor ``key``."""
    n = 2 + i % 3  # lengths differ, so padding is exercised
    codes = [_CODES[(i + j) % 6] for j in range(n)]
    steps = [float(k) for k in range(n)]
    hours = [_T0 + timedelta(hours=k) for k in range(n)]
    pairs = [[float(k), 1.0] for k in range(n)]
    return {
        "sequence": codes,
        "multi_hot": codes,
        "nested_sequence": [codes[:2], codes[1:]],
        "nested_multihot": [codes[:2], codes[1:]],
        "deep_nested_sequence": [[codes[:2], codes[1:]], [codes[:1]]],
        "tensor": [float(i), 1.0, 2.0],
        "nested_sequence_floats": [[float(i), 1.0], [2.0, None]][: 1 + i % 2],
        "deep_nested_sequence_floats": [[[float(i), 1.0], [2.0, 3.0]]],
        "timeseries": (hours, pairs),
        "temporal_timeseries": ([_T0 + timedelta(hours=2 * k) for k in range(n)],
                                np.array(pairs)),
        "stagenet": (steps, codes),
        "stagenet_tensor": (steps, pairs),
        "text": " ".join(codes),
        "tuple_time_text": ([f"note {c}" for c in codes], steps),
        "cehr": codes,
    }[key]


# Input processors probed, in the order of the table.
PROCESSORS = ["sequence", "nested_sequence", "deep_nested_sequence", "multi_hot",
              "nested_multihot", "tensor", "nested_sequence_floats",
              "deep_nested_sequence_floats", "timeseries", "temporal_timeseries",
              "stagenet", "stagenet_tensor", "text", "tuple_time_text", "cehr"]


def sample_dataset(key: str, n: int = 8):
    """A binary-label dataset with one input field ``x`` of processor ``key``.

    Examples:
        >>> sample_dataset("sequence").input_schema  # doctest: +SKIP
        {'x': 'sequence'}
    """
    from pyhealth.datasets import create_sample_dataset

    samples = [{"patient_id": f"p{i}", "x": _value(key, i), "label": i % 2}
               for i in range(n)]
    return create_sample_dataset(samples, {"x": key}, {"label": "binary"},
                                 dataset_name=key)


def embedding_layer(dataset) -> str:
    """The layer ``EmbeddingModel`` creates for field ``x``, or the error.

    Examples:
        >>> embedding_layer(sample_dataset("tensor"))  # doctest: +SKIP
        'Linear'
    """
    from pyhealth.models import EmbeddingModel

    try:
        layers = EmbeddingModel(dataset, embedding_dim=8).embedding_layers
    except Exception as e:  # noqa: BLE001 - any failure is the answer
        return f"error: {type(e).__name__}"
    return type(layers["x"]).__name__ if "x" in layers else "none"


def runs(dataset, model_name: str) -> tuple[bool, str]:
    """Whether a model takes a forward and backward pass, and the error if not.

    Examples:
        >>> runs(sample_dataset("sequence"), "RNN")  # doctest: +SKIP
        (True, '')
    """
    import pyhealth.models
    from pyhealth.datasets import get_dataloader

    try:
        torch.manual_seed(0)
        model = getattr(pyhealth.models, model_name)(dataset)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        loss = model(**batch)["loss"]
        loss.backward()
        if not torch.isfinite(loss):
            return False, "loss is not finite"
    except Exception as e:  # noqa: BLE001 - any failure is the answer
        return False, f"{type(e).__name__}: {e}".replace("\n", " ")[:160]
    return True, ""


def probe() -> dict[str, dict[str, object]]:
    """Runs every processor and model pairing.

    Examples:
        >>> probe()["sequence"]["RNN"]  # doctest: +SKIP
        (True, '')
    """
    out = {}
    for key in PROCESSORS:
        ds = sample_dataset(key)
        out[key] = {"embedding": embedding_layer(ds)}
        for name in MODELS:
            out[key][name] = runs(ds, name)
    return out


_EMBEDDING_TEXT = {"Embedding": "``nn.Embedding``", "Linear": "``nn.Linear``",
                   "none": "none (passed through)"}


def render(results: dict[str, dict[str, object]]) -> str:
    """The compatibility page as reStructuredText.

    Examples:
        >>> render(probe()).splitlines()[0]  # doctest: +SKIP
        'Processor and model compatibility'
    """
    lines = [
        "Processor and model compatibility",
        "=================================",
        "",
        "Which layer :class:`~pyhealth.models.EmbeddingModel` builds for each",
        "input processor, and which general-purpose models train on a field of",
        "that type. Each pairing was tried on a tiny binary-label dataset with",
        "one input field, with one forward and backward pass and the model's",
        "default arguments. ✓ means it ran with a finite loss. ✗ means it",
        "raised an error or gave a loss that is not finite.",
        "",
        "The table is generated by ``tests/core/processor_compatibility_utils.py``.",
        "Run ``python -m tests.core.processor_compatibility_utils --errors`` from",
        "the repository root to rebuild it and see each error.",
        "``tests/core/test_processor_compatibility.py`` checks that every ✓",
        "still runs.",
        "",
        "Notes:",
        "",
        "- ✗ can mean the model does not support that input, or a bug. Both",
        "  usually surface as a shape or type error, not a clear message.",
        "- ``text`` and ``tuple_time_text`` are meant for text models such as",
        "  :class:`~pyhealth.models.TransformersModel`, and ``cehr`` for",
        "  CEHR-style models. ``EmbeddingModel`` does not handle them, so every",
        "  model below fails on them.",
        "- Image, signal, audio and graph fields, and",
        "  :class:`~pyhealth.processors.JaggedFeatureProcessor` fields, are read",
        "  by the models built for them and are not listed. ``raw`` and",
        "  ``ignore`` fields are not embedded, so they are not listed either.",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "   :stub-columns: 1",
        "",
        "   * - Processor",
        "     - Embedding",
    ]
    lines += [f"     - {name}" for name in MODELS]
    for key, row in results.items():
        emb = str(row["embedding"])
        lines.append(f"   * - ``{key}``")
        lines.append(f"     - {_EMBEDDING_TEXT.get(emb, emb)}")
        lines += [f"     - {'✓' if row[name][0] else '✗'}" for name in MODELS]
    return "\n".join(lines) + "\n"


def read_page(text: str) -> dict[str, dict[str, str]]:
    """The table cells of a rendered page, by processor and column.

    Examples:
        >>> read_page(render(probe()))["sequence"]["RNN"]  # doctest: +SKIP
        '✓'
    """
    rows = re.findall(r"^   \* - (.+?)\n((?:^     - .*\n)+)", text, flags=re.MULTILINE)
    header, body = rows[0], rows[1:]
    columns = re.findall(r"^     - (.*)$", header[1], flags=re.MULTILINE)
    out = {}
    for key, cells in body:
        values = re.findall(r"^     - (.*)$", cells, flags=re.MULTILINE)
        out[key.strip("`")] = dict(zip(columns, values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--errors", action="store_true", help="print each error")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    results = probe()
    PAGE.write_text(render(results), encoding="utf-8")
    print(f"wrote {PAGE}")
    if args.errors:
        for key, row in results.items():
            for name in MODELS:
                ok, err = row[name]
                if not ok:
                    print(f"{key:28s} {name:18s} {err}")


if __name__ == "__main__":
    main()
