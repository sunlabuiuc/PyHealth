"""PTB-XL classification from HeartLang heartbeat sentences."""

import os
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .ptbxl_multilabel_classification import PTBXLMultilabelClassification

PTBXL_100HZ_MEAN = -0.000819
"""Mean of every sample of the 17,418 PTB-XL 1.0.3 training ECGs (folds 1-8)
at 100 Hz, in mV.

HeartLang follows the PTB-XL benchmark (Strodthoff et al., 2021), which
standardizes every sample with one mean and standard deviation fitted on the
training folds. The two constants can be recomputed with (``root`` is the
PTB-XL 1.0.3 download)::

    import numpy as np, pandas as pd, wfdb

    db = pd.read_csv(f"{root}/ptbxl_database.csv")
    files = db.loc[db.strat_fold <= 8, "filename_lr"]
    x = np.stack([wfdb.rdsamp(f"{root}/{f}")[0] for f in files])
    print(x.mean(), x.std())  # -0.000819, 0.235129
"""

PTBXL_100HZ_STD = 0.235129
"""Standard deviation of the same samples as :data:`PTBXL_100HZ_MEAN`, in mV."""


class PTBXLHeartLangClassification(PTBXLMultilabelClassification):
    """PTB-XL multi-label classification with ECGs as HeartLang sentences.

    Works with :class:`~pyhealth.datasets.PTBXLDataset`.

    Builds the inputs of :class:`~pyhealth.models.HeartLang`: each ECG is
    standardized, split into heartbeat "words" around the QRS complexes of
    the first selected lead, lead I by default
    (:func:`~pyhealth.models.heartlang.ecg_to_heartbeat_sentence`), and
    returned with the lead and time index of every word. Labels follow
    :class:`~pyhealth.tasks.PTBXLMultilabelClassification` and keep every
    SCP statement whatever its likelihood (``min_likelihood`` is fixed at 0,
    not an argument), which matches the PTB-XL benchmark that HeartLang uses
    (Jin et al., ICLR 2025). Use
    :func:`~pyhealth.datasets.split_by_strat_fold` for its split: folds 1-8
    train, 9 validation, 10 test.

    The published model was trained on 100 Hz ECGs: its 96-sample words and
    the default standardization constants assume that rate, so build the
    dataset with ``PTBXLDataset(..., sampling_rate=100)``. Other rates raise
    ``ValueError`` unless ``sampling_rate=None``. QRS detection uses ``wfdb``
    (the ``pyhealth[ptbxl]`` extra); an ECG in which no QRS complex is
    detected is kept as an all-padding sentence, with a warning.

    Args:
        scp_statements_path: ``scp_statements.csv`` of the PTB-XL download;
            use ``dataset.scp_statements_path``.
        label_set: ``"superdiagnostic"`` (5 classes, default),
            ``"subdiagnostic"``, ``"form"``, ``"rhythm"``, ``"diagnostic"``
            or ``"all"``.
        signal_mean: Mean subtracted from every sample. Defaults to
            :data:`PTBXL_100HZ_MEAN`, as in the paper. ``None`` together with
            ``signal_std=None`` z-scores each lead of each record instead.
        signal_std: Standard deviation every sample is divided by. Defaults
            to :data:`PTBXL_100HZ_STD`.
        leads: Lead indices to use (I, II, III, aVR, aVL, aVF, V1-V6 order).
            Defaults to all 12. QRS complexes are detected on the first one.
        max_len: Words per sentence (256 for the published model).
        token_len: Samples per word (96 for the published model).
        sampling_rate: The sampling rate the ECGs must have, in Hz (100 for
            the published model). ``None`` accepts any rate; then choose
            ``token_len`` and the standardization for that rate yourself.

    Raises:
        ValueError: From ``set_task``, if an ECG's sampling rate is not
            ``sampling_rate``.

    Attributes:
        task_name (str): ``"PTBXLHeartLangClassification"``.
        input_schema (dict): ``ecg_sentence``, ``lead_ids`` and ``time_ids``,
            all tensors.
        output_schema (dict): ``{"labels": "multilabel"}``.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset, split_by_strat_fold
        >>> from pyhealth.tasks import PTBXLHeartLangClassification
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl/1.0.3")  # doctest: +SKIP
        >>> task = PTBXLHeartLangClassification(  # doctest: +SKIP
        ...     scp_statements_path=dataset.scp_statements_path)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> train, val, test = split_by_strat_fold(samples)  # doctest: +SKIP
        >>> samples[0]["ecg_sentence"].shape  # doctest: +SKIP
        torch.Size([256, 96])
    """

    task_name: str = "PTBXLHeartLangClassification"
    input_schema: ClassVar[dict[str, str]] = {
        "ecg_sentence": "tensor",
        "lead_ids": "tensor",
        "time_ids": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {"labels": "multilabel"}

    def __init__(
        self,
        scp_statements_path: str | os.PathLike,
        label_set: str = "superdiagnostic",
        signal_mean: float | None = PTBXL_100HZ_MEAN,
        signal_std: float | None = PTBXL_100HZ_STD,
        leads: Sequence[int] | None = None,
        max_len: int = 256,
        token_len: int = 96,
        sampling_rate: int | None = 100,
    ) -> None:
        if (signal_mean is None) != (signal_std is None):
            raise ValueError("give both signal_mean and signal_std, or neither")
        if signal_std is not None and signal_std <= 0:
            raise ValueError("signal_std must be positive")
        self.signal_mean = signal_mean
        self.signal_std = signal_std
        self.leads = None if leads is None else [int(i) for i in leads]
        self.max_len = max_len
        self.token_len = token_len
        self.sampling_rate = sampling_rate
        super().__init__(
            scp_statements_path,
            label_set=label_set,
            min_likelihood=0.0,
            standardize="record" if signal_mean is None else None,
            remove_baseline=False,
        )

    def _prepare_signal(self, signal_file: str, sampling_rate: int) -> np.ndarray:
        """Loads one ECG and applies the global standardization.

        Returns:
            The standardized ``(12, 10 * sampling_rate)`` signal.

        Raises:
            ValueError: If ``sampling_rate`` is not the one the task expects.
        """
        if self.sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(
                f"PTBXLHeartLangClassification expects {self.sampling_rate} Hz "
                f"ECGs (the rate of the pretrained HeartLang model) but got "
                f"{sampling_rate} Hz; use PTBXLDataset(..., sampling_rate="
                f"{self.sampling_rate}), or pass sampling_rate=None with your "
                "own token_len and standardization"
            )
        signal = super()._prepare_signal(signal_file, sampling_rate)
        if self.signal_mean is not None:
            signal = (signal - self.signal_mean) / self.signal_std
        return np.ascontiguousarray(signal, dtype=np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Builds one sample per labelled ECG of a patient.

        Args:
            patient: A patient from :class:`~pyhealth.datasets.PTBXLDataset`.

        Returns:
            Dicts with ``patient_id``, ``strat_fold``, ``ecg_sentence``
            (float32 ``(max_len, token_len)``), ``lead_ids`` and ``time_ids``
            (``(max_len,)``) and ``labels``.
        """
        from pyhealth.models.heartlang import ecg_to_heartbeat_sentence

        samples = []
        for sample in super().__call__(patient):
            signal = sample.pop("signal")
            rate = signal.shape[1] // 10
            words, lead_ids, time_ids = ecg_to_heartbeat_sentence(
                signal, sampling_rate=rate, max_len=self.max_len,
                token_len=self.token_len, leads=self.leads,
            )
            sample.update(ecg_sentence=words, lead_ids=lead_ids, time_ids=time_ids)
            samples.append(sample)
        return samples
