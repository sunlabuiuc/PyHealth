# Authors:     Paul Garcia (alanpg2), Rogelio Medina (orm9), Cesar Nava (can14)
# Paper:       PTB-XL, a large publicly available electrocardiography dataset
# Link:        https://physionet.org/content/ptb-xl/1.0.3/
# Description: PyHealth dataset wrapper for PTB-XL 12-lead ECG records
#              with four binary diagnostic superclass labels (MI, HYP, STTC, CD).

"""PTB-XL ECG dataset for PyHealth.

Reference:
    Wagner et al. (2020). PTB-XL, a large publicly available electrocardiography
    dataset. Scientific Data, 7(154).
    https://physionet.org/content/ptb-xl/1.0.3/
"""
from __future__ import annotations

import ast
import os
from typing import Dict, Optional

import pandas as pd

from pyhealth.datasets import BaseDataset


class PTBXLDataset(BaseDataset):
    """PTB-XL large publicly available ECG dataset.

    PTB-XL contains 21799 10-second 12-lead ECG records from 18869 patients,
    annotated with SCP-ECG diagnostic statements.  Records are mapped to four
    binary diagnostic superclass labels: MI (myocardial infarction), HYP
    (hypertrophy), STTC (ST/T-change), and CD (conduction disturbance).

    The dataset must be downloaded from PhysioNet before use::

        wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

    On first instantiation a ``ptbxl_metadata.csv`` file is generated inside
    *root*.  Subsequent instantiations reuse this cached file.

    Args:
        root: Path to the PTB-XL root directory (must contain
            ``ptbxl_database.csv``, ``scp_statements.csv``, and the
            ``records100/`` or ``records500/`` subdirectories).
        dataset_name: Optional name for this dataset instance.
        config_path: Path to a custom YAML schema config.  Defaults to the
            built-in ``configs/ptbxl.yaml``.
        sampling_rate: Waveform sampling rate to load.  Must be ``100`` or
            ``500`` (Hz).  Default: ``100``.
        dev: If ``True``, restrict to the first 100 patients for fast
            development iterations.

    Examples:
        >>> dataset = PTBXLDataset(root="/data/ptb-xl/")
        >>> print(dataset.stats())
        >>> from pyhealth.tasks.ecg_classification import ECGBinaryClassification
        >>> sample_ds = dataset.set_task(ECGBinaryClassification(task_label="MI"))
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        sampling_rate: int = 100,
        dev: bool = False,
    ) -> None:
        if sampling_rate not in (100, 500):
            raise ValueError("sampling_rate must be 100 or 500")
        self.sampling_rate = sampling_rate

        metadata_path = os.path.join(root, "ptbxl_metadata.csv")
        if not os.path.exists(metadata_path):
            self._prepare_metadata(root)

        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__), "configs", "ptbxl.yaml"
            )

        super().__init__(
            root=root,
            tables=["ecg_records"],
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            dev=dev,
        )

    # ------------------------------------------------------------------
    # Metadata preparation
    # ------------------------------------------------------------------

    def _prepare_metadata(self, root: str) -> None:
        """Parse raw PTB-XL files and write ``ptbxl_metadata.csv``.

        Reads ``ptbxl_database.csv`` for record-level information and
        ``scp_statements.csv`` for the diagnostic superclass mapping.  The
        resulting CSV contains one row per ECG record with columns required
        by ``configs/ptbxl.yaml``.

        Args:
            root: PTB-XL root directory.

        Returns:
            None.  Writes ``ptbxl_metadata.csv`` to *root* as a side effect.

        Raises:
            FileNotFoundError: If ``ptbxl_database.csv`` is absent.
        """
        db_path = os.path.join(root, "ptbxl_database.csv")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"ptbxl_database.csv not found in {root}. "
                "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/"
            )

        df = pd.read_csv(db_path, index_col="ecg_id")
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

        # Build {scp_code -> diagnostic_superclass} mapping from scp_statements
        superclass_map: Dict[str, str] = {}
        scp_path = os.path.join(root, "scp_statements.csv")
        if os.path.exists(scp_path):
            scp_df = pd.read_csv(scp_path, index_col=0)
            scp_df = scp_df[scp_df["diagnostic"] == 1]
            superclass_map = scp_df["diagnostic_class"].to_dict()

        filename_col = "filename_hr" if self.sampling_rate == 500 else "filename_lr"

        def _binary_label(scp_codes: Dict[str, float], superclass: str) -> int:
            """Return 1 if any code in *scp_codes* maps to *superclass*.

            Args:
                scp_codes: Mapping of SCP statement code to likelihood score.
                superclass: Diagnostic superclass string (e.g. ``"MI"``).

            Returns:
                ``1`` if the record belongs to *superclass*, ``0`` otherwise.
            """
            for code, likelihood in scp_codes.items():
                if likelihood > 0 and superclass_map.get(code) == superclass:
                    return 1
            return 0

        records = []
        for ecg_id, row in df.iterrows():
            rec_date = row.get("recording_date", "2000-01-01 00:00:00")
            if pd.isna(rec_date):
                rec_date = "2000-01-01 00:00:00"

            scp = row["scp_codes"]
            records.append(
                {
                    "patient_id": str(int(row["patient_id"])),
                    "ecg_id": str(ecg_id),
                    "recording_date": str(rec_date),
                    # Store absolute path so the task can load without root
                    "filename": os.path.join(root, str(row[filename_col])),
                    "mi_label": _binary_label(scp, "MI"),
                    "hyp_label": _binary_label(scp, "HYP"),
                    "sttc_label": _binary_label(scp, "STTC"),
                    "cd_label": _binary_label(scp, "CD"),
                }
            )

        pd.DataFrame(records).to_csv(
            os.path.join(root, "ptbxl_metadata.csv"), index=False
        )

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    @property
    def default_task(self) -> "ECGBinaryClassification":  # noqa: F821
        """Return the default MI binary classification task.

        Returns:
            An :class:`~pyhealth.tasks.ecg_classification.ECGBinaryClassification`
            instance configured for MI (myocardial infarction) prediction.
        """
        from pyhealth.tasks.ecg_classification import ECGBinaryClassification

        return ECGBinaryClassification()
