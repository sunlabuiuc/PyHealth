"""A split that reserves gold-labeled subjects for testing, as DiPro does."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import chain

import numpy as np
from sklearn.model_selection import train_test_split

from .splitter import Split


@dataclass(frozen=True)
class GoldTestSplit(Split):
    """A 7:1:2 patient split that reserves gold-labeled subjects for testing.

    Follows the split of DiPro (Liu et al., 2025; commit 64b12ec,
    ``split_train_and_test.py``, MIT License, Copyright (c) 2025 LIU Chen @ The
    Hong Kong Polytechnic University): every subject with manually validated
    labels goes to the test part, which random subjects then fill up to 20% of
    all subjects. An eighth of the others goes to validation. Subjects are
    sorted before each ``train_test_split``.

    If the gold subjects already exceed 20%, the test part holds only them.

    Args:
        gold_subjects: Subjects of the test part. Those without a sample are
            ignored.
        seed: ``random_state`` of both ``train_test_split`` calls. DiPro uses 0.

    Examples:
        >>> from pyhealth.datasets import GoldTestSplit
        >>> split = GoldTestSplit(gold_subjects=["p9"])
        >>> samples = {f"p{i}": [i] for i in range(10)}
        >>> train, val, test = split.split_indices(samples)
        >>> sorted(len(part) for part in (train, val, test))
        [1, 2, 7]
        >>> 9 in test
        True
    """

    gold_subjects: Sequence[str]
    seed: int = 0

    TEST_FRACTION = 0.2
    VAL_FRACTION_OF_REST = 1 / 8

    def __post_init__(self) -> None:
        """Stores the gold subjects sorted, as strings, for a stable cache key."""
        subjects = tuple(sorted({str(subject) for subject in self.gold_subjects}))
        object.__setattr__(self, "gold_subjects", subjects)

    def to_dict(self) -> dict:
        """JSON description used in the processed-sample cache key.

        Returns:
            ``{"kind": "gold_test", "gold_subjects": [...], "seed": seed}``.
        """
        return {
            "kind": "gold_test",
            "gold_subjects": list(self.gold_subjects),
            "seed": int(self.seed),
        }

    def split_indices(
        self, patient_to_index: Mapping[str, Sequence[int]]
    ) -> list[np.ndarray]:
        """Returns the sample indices of the train, validation and test parts.

        Args:
            patient_to_index: Subject id to the indices of its samples.

        Returns:
            Three sorted ``int64`` arrays. Each subject's samples land in
            exactly one part.
        """
        subjects = sorted(patient_to_index, key=str)
        gold = set(self.gold_subjects)
        fixed = [subject for subject in subjects if str(subject) in gold]
        rest = [subject for subject in subjects if str(subject) not in gold]

        # A share of all subjects applied to ``rest``, exactly as DiPro does.
        share = (len(subjects) * self.TEST_FRACTION - len(fixed)) / max(len(subjects), 1)
        train_val, test = rest, []
        if share > 0 and len(rest) > 1:
            train_val, test = train_test_split(
                rest, test_size=share, random_state=self.seed
            )
        train, val = train_val, []
        if len(train_val) > 1:
            train, val = train_test_split(
                train_val, test_size=self.VAL_FRACTION_OF_REST, random_state=self.seed
            )

        parts = []
        for part in (train, val, [*test, *fixed]):
            indices = np.fromiter(
                chain.from_iterable(patient_to_index[subject] for subject in part),
                dtype=np.int64,
            )
            indices.sort()
            parts.append(indices)
        return parts
