"""PyHealth processor for MedMod's EHR tensor: standardize, then float32.

``normalizer=<file>`` loads a fixed MedMod normalizer file and learns nothing.
``PH_NORMALIZER``, the task's default, is the file MedMod's released code loads
for this task (the phenotyping one, via the ``is`` bug in ehr_funcs; see
discretizer.py).

``normalizer=None`` learns the statistics (``learns_statistics``), and PyHealth
fits them on the training part only when the task is set with MedMod's split::

    train, val, test = dataset.set_task(task, split=MedModSplit())

The statistics follow MedMod's create_normalizer_state.py:87-93 and
preprocessing.py:193-215: means and stds over every bin of every training stay,
all 76 columns. Either way only the 12 continuous columns are standardized
(preprocessing.py:224-232, 254). MedMod computes in float64 and its trainers
cast to float32 (``torch.from_numpy(x).float()``); so do we.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from ..datasets.medmod.discretizer import Discretizer, Normalizer
from .base_processor import FeatureProcessor


class MedModEHRProcessor(FeatureProcessor):
    """Standardizes the task's unnormalized ``ehr`` (bins x 76) to float32.

    Args:
        normalizer: a MedMod normalizer file, used as is (the task passes
            ``PH_NORMALIZER`` by default). ``None``: learn means and stds in
            ``fit``, which PyHealth calls on the training part when the task is
            set with ``split=MedModSplit()``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets.medmod.discretizer import PH_NORMALIZER
        >>> from pyhealth.processors.medmod_ehr_processor import MedModEHRProcessor
        >>> processor = MedModEHRProcessor(PH_NORMALIZER)
        >>> processor.process(np.zeros((48, 76))).dtype
        torch.float32
    """

    learns_statistics = True
    stores_tensor = True

    def __init__(self, normalizer: Optional[str | Path] = None):
        self.normalizer_file = None if normalizer is None else str(normalizer)
        # An instance attribute: PyHealth reads it to warn about fitting on
        # all samples, which only matters when statistics are learned.
        self.learns_statistics = normalizer is None
        fields = Discretizer().continuous_columns()
        self.normalizer = Normalizer(fields=fields)
        if normalizer is not None:
            self.normalizer.load_params(normalizer)

    def fit(self, samples: Iterable[Dict[str, Any]], field: str) -> None:
        if self.normalizer_file is not None:
            return
        self.normalizer.fit(np.asarray(s[field], dtype=float) for s in samples)

    def process(self, value: Any) -> torch.Tensor:
        x = self.normalizer.transform(np.asarray(value, dtype=float))
        return torch.from_numpy(x.astype(np.float32))

    def is_token(self) -> bool:
        return False

    def schema(self) -> tuple[str, ...]:
        return ("value",)

    def dim(self) -> tuple[int, ...]:
        return (2,)

    def spatial(self) -> tuple[bool, ...]:
        return (True, False)  # time bins, then features
