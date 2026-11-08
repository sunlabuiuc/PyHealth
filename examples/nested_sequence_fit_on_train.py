"""Fit a nested-sequence processor on training data, then apply it to test data.

Fitting processors on the training split only avoids leaking test information,
but a test patient can then have a visit with more codes than any training
visit. ``NestedSequenceProcessor`` keeps the width it learned in ``fit()``:
longer visits are truncated (keeping the first codes, with a one-time warning)
so every sample has the same shape and batches stack. Use ``padding`` to leave
room for longer visits.

Runs in a second on CPU with synthetic data; no download.

Usage:
    python examples/nested_sequence_fit_on_train.py
"""

import torch

from pyhealth.processors import NestedSequenceProcessor

train = [
    {"visits": [["I10", "E11"], ["I10"]]},
    {"visits": [["J45", "E11", "I10"]]},  # longest training visit: 3 codes
]
test_visits = [["I10"], ["J45", "E11", "I10", "N18", "E78"]]  # 5 codes


def main():
    processor = NestedSequenceProcessor()
    processor.fit(train, "visits")
    print("fitted width:", processor.size())

    rows = processor.process(test_visits)  # warns once: 5 codes > 3
    print("test sample shape:", tuple(rows.shape))
    print("batch of train + test stacks:", tuple(torch.cat(
        [processor.process(train[1]["visits"]), rows]).shape))

    roomy = NestedSequenceProcessor(padding=2)  # width = 3 + 2
    roomy.fit(train, "visits")
    print("with padding=2, width:", roomy.size(),
          "-> test shape:", tuple(roomy.process(test_visits).shape))


if __name__ == "__main__":
    main()
