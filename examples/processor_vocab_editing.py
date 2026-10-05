"""Editing a fitted processor's vocabulary, then fitting it again.

Processors that keep a ``code_vocab`` (SequenceProcessor,
NestedSequenceProcessor, DeepNestedSequenceProcessor, StageNetProcessor and
NestedMultiHotProcessor) can drop codes with ``remove``, keep a chosen subset
with ``retain`` and take new codes with ``add``. This script edits a
SequenceProcessor that way, fits it on more samples, and looks the codes up in
an embedding table with ``vocab_size()`` rows. Runs on CPU in a few seconds;
no data download.

Usage:
    python examples/processor_vocab_editing.py
"""

import torch

from pyhealth.processors import SequenceProcessor


def main():
    processor = SequenceProcessor()
    processor.fit([{"codes": ["A", "B", "C", "D", "E"]}], "codes")
    print("fitted:      ", processor.code_vocab)

    processor.remove({"A", "B"})
    print("after remove:", processor.code_vocab)

    processor.add({"X"})
    print("after add:   ", processor.code_vocab)

    processor.fit([{"codes": ["Y", "Z"]}], "codes")
    print("refitted:    ", processor.code_vocab)

    embedding = torch.nn.Embedding(processor.vocab_size(), 8)
    # A was removed above, so it maps to <unk>
    indices = processor.process(["A", "C", "X", "Y", "Z"])
    print("indices:     ", indices.tolist())
    print("embedded:    ", tuple(embedding(indices).shape))


if __name__ == "__main__":
    main()
