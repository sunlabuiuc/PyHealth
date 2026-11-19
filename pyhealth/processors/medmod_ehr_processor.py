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

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

    def __init__(self, normalizer: str | Path | None = None):
        # Imported here: pyhealth.datasets imports the tasks, which import this
        # module, so a module-level import is circular when this module is
        # imported first.
        from ..datasets.medmod.discretizer import Discretizer, Normalizer

        self.normalizer_file = None if normalizer is None else str(normalizer)
        # An instance attribute: PyHealth reads it to warn about fitting on
        # all samples, which only matters when statistics are learned.
        self.learns_statistics = normalizer is None
        fields = Discretizer().continuous_columns()
        self.normalizer = Normalizer(fields=fields)
        if normalizer is not None:
            self.normalizer.load_params(normalizer)

    def fit(self, samples: Iterable[dict[str, Any]], field: str) -> None:
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


class MedModEHRSequenceProcessor(MedModEHRProcessor):
    """``MedModEHRProcessor`` for EHR of varying length (bins x 76 per stay).

    Each sample is standardized as in :class:`MedModEHRProcessor`. A batch is
    zero-padded at the end to its longest stay and comes with the true
    lengths, as MedMod's ``pad_zeros`` / ``my_collate_fusion``
    (src/data/utils.py:160-170, 399-406) feed its LSTM, which packs them
    (src/models/encoders.py:44-45). The batch value is ``(ehr, lengths)``:
    ``ehr`` is ``[batch, max_bins, 76]`` float32, ``lengths`` int64.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets.medmod.discretizer import PH_NORMALIZER
        >>> from pyhealth.processors.medmod_ehr_processor import (
        ...     MedModEHRSequenceProcessor,
        ... )
        >>> processor = MedModEHRSequenceProcessor(PH_NORMALIZER)
        >>> x = [processor.process(np.zeros((n, 76))) for n in (3, 5)]
        >>> ehr, lengths = processor.collate(x)
        >>> tuple(ehr.shape), lengths.tolist()
        ((2, 5, 76), [3, 5])
    """

    def collate(self, values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.tensor([v.shape[0] for v in values], dtype=torch.long)
        ehr = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
        return ehr, lengths

    def schema(self) -> tuple[str, ...]:
        return ("value", "lengths")
