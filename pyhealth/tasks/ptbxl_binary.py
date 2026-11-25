"""Binary single-diagnosis tasks on PTB-XL.

Paper (TaskAug, the method these tasks are used with):
    Aniruddh Raghu, Divya Shanmugam, Eugene Pomerantsev, John Guttag, and
    Collin M. Stultz. "Data Augmentation for Electrocardiograms."
    Proceedings of the Conference on Health, Inference, and Learning (CHIL),
    PMLR 174:282-310, 2022. https://proceedings.mlr.press/v174/raghu22a.html

Author:
    Lisha Ranganath (GitHub: @LishaRanganath) and Meghaben D Dhalwadi

Description:
    Each task labels an ECG 1 if it carries the chosen diagnostic superclass
    (MI, HYP, STTC or CD) and 0 otherwise. They reuse the helpers in
    ``pyhealth.tasks.ptbxl`` and the records produced by ``PTBXLDataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask
from pyhealth.tasks.ptbxl import (
    PTBXL_DIAGNOSTIC_SUPERCLASSES,
    aggregate_diagnostic_superclasses,
    load_diagnostic_class_map,
)


class PTBXLBinaryClassification(BaseTask):
    """Binary classification of one PTB-XL diagnostic superclass.

    Args:
        scp_statements_path (str | Path): Path to ``scp_statements.csv``.
        diagnostic_class (str): One of ``NORM``, ``MI``, ``STTC``, ``CD``,
            ``HYP``. The label is 1 when the record has this superclass.

    Examples:
        >>> from pyhealth.tasks.ptbxl_binary import PTBXLBinaryClassification
        >>> task = PTBXLBinaryClassification(
        ...     "/data/ptb-xl/1.0.3/scp_statements.csv", "HYP")
        >>> task.diagnostic_class
        'HYP'
    """

    task_name: str = "PTBXLBinaryClassification"
    input_schema: ClassVar[dict[str, str]] = {"signal": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"label": "binary"}

    def __init__(self, scp_statements_path: str | Path, diagnostic_class: str) -> None:
        if diagnostic_class not in PTBXL_DIAGNOSTIC_SUPERCLASSES:
            raise ValueError(
                f"diagnostic_class must be one of {PTBXL_DIAGNOSTIC_SUPERCLASSES}"
            )
        self.scp_statements_path = Path(scp_statements_path)
        self.diagnostic_class = diagnostic_class
        super().__init__()

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Build one sample per ECG record for the patient."""
        from pyhealth.datasets.ptbxl import load_ptbxl_record

        class_map = load_diagnostic_class_map(self.scp_statements_path)
        samples: list[dict[str, Any]] = []
        for event in patient.get_events(event_type="records"):
            labels = aggregate_diagnostic_superclasses(event.scp_codes, class_map)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": str(event.record_id),
                    "signal": load_ptbxl_record(event.signal_file),
                    "label": int(self.diagnostic_class in labels),
                    "strat_fold": int(float(event.strat_fold)),
                }
            )
        return samples


class PTBXLMI(PTBXLBinaryClassification):
    """Myocardial infarction (MI) vs. rest on PTB-XL.

    An ECG gets label 1 when it carries the ``MI`` diagnostic superclass.

    Args:
        scp_statements_path (str | Path): Path to ``scp_statements.csv``.

    Examples:
        >>> from pyhealth.tasks import PTBXLMI
        >>> task = PTBXLMI("/data/ptb-xl/1.0.3/scp_statements.csv")  # doctest: +SKIP
        >>> task.diagnostic_class  # doctest: +SKIP
        'MI'
    """

    task_name: str = "PTBXL_MI"

    def __init__(self, scp_statements_path: str | Path) -> None:
        super().__init__(scp_statements_path, "MI")


class PTBXLHYP(PTBXLBinaryClassification):
    """Hypertrophy (HYP) vs. rest on PTB-XL.

    An ECG gets label 1 when it carries the ``HYP`` diagnostic superclass.

    Args:
        scp_statements_path (str | Path): Path to ``scp_statements.csv``.

    Examples:
        >>> from pyhealth.tasks import PTBXLHYP
        >>> task = PTBXLHYP("/data/ptb-xl/1.0.3/scp_statements.csv")  # doctest: +SKIP
        >>> task.diagnostic_class  # doctest: +SKIP
        'HYP'
    """

    task_name: str = "PTBXL_HYP"

    def __init__(self, scp_statements_path: str | Path) -> None:
        super().__init__(scp_statements_path, "HYP")


class PTBXLSTTC(PTBXLBinaryClassification):
    """ST/T change (STTC) vs. rest on PTB-XL.

    An ECG gets label 1 when it carries the ``STTC`` diagnostic superclass.

    Args:
        scp_statements_path (str | Path): Path to ``scp_statements.csv``.

    Examples:
        >>> from pyhealth.tasks import PTBXLSTTC
        >>> task = PTBXLSTTC("/data/ptb-xl/1.0.3/scp_statements.csv")  # doctest: +SKIP
        >>> task.diagnostic_class  # doctest: +SKIP
        'STTC'
    """

    task_name: str = "PTBXL_STTC"

    def __init__(self, scp_statements_path: str | Path) -> None:
        super().__init__(scp_statements_path, "STTC")


class PTBXLCD(PTBXLBinaryClassification):
    """Conduction disturbance (CD) vs. rest on PTB-XL.

    An ECG gets label 1 when it carries the ``CD`` diagnostic superclass.

    Args:
        scp_statements_path (str | Path): Path to ``scp_statements.csv``.

    Examples:
        >>> from pyhealth.tasks import PTBXLCD
        >>> task = PTBXLCD("/data/ptb-xl/1.0.3/scp_statements.csv")  # doctest: +SKIP
        >>> task.diagnostic_class  # doctest: +SKIP
        'CD'
    """

    task_name: str = "PTBXL_CD"

    def __init__(self, scp_statements_path: str | Path) -> None:
        super().__init__(scp_statements_path, "CD")
