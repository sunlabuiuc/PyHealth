"""PTB-XL 12-lead ECG dataset for PyHealth."""

import ast
import logging
import os
from pathlib import Path

import pandas as pd

from ..tasks import PTBXLMultilabelClassification
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

# Label sets that can be derived from ``scp_statements.csv``.
LABEL_SETS = (
    "all",
    "diagnostic",
    "subdiagnostic",
    "superdiagnostic",
    "form",
    "rhythm",
)


class PTBXLDataset(BaseDataset):
    """Base ECG dataset for PTB-XL.

    PTB-XL is a large, publicly available collection of 21,799 clinical
    12-lead ECGs (10 seconds each) from 18,869 patients, annotated with
    71 SCP-ECG statements covering diagnostic, form and rhythm categories.
    Records are provided at 100 Hz and 500 Hz. The dataset is open access
    (no credentialing) and can be downloaded from
    https://physionet.org/content/ptb-xl/1.0.3/ .

    Each ECG record is stored as one event of type ``"ptbxl"``. Several ECGs
    may belong to the same patient. The ``strat_fold`` attribute contains the
    recommended 10-fold split (folds 1-8 train, 9 validation, 10 test) which is
    patient-wise, so it can be used directly for leakage-free splits.

    Citations:
        Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al. (2020).
        PTB-XL, a large publicly available electrocardiography dataset.
        Scientific Data, 7, 154.

    Args:
        root: Root directory of the raw data. It must contain
            ``ptbxl_database.csv``, ``scp_statements.csv`` and the
            ``records100`` / ``records500`` folders.
        dataset_name: Optional name of the dataset. Defaults to ``"ptbxl"``.
        config_path: Optional path to a YAML config. Defaults to the config
            shipped in ``pyhealth/datasets/configs/ptbxl.yaml``.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of parallel workers for data processing.
        dev: If True, only a small subset of patients is loaded.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl/1.0.3")
        >>> dataset.stats()
        >>> samples = dataset.set_task()  # 12-lead, 100 Hz, diagnostic labels
        >>> print(samples[0]["signal"].shape)
        torch.Size([12, 1000])
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ptbxl.yaml"
        metadata_path = os.path.join(root, "ptbxl-metadata-pyhealth.csv")
        if not os.path.exists(metadata_path):
            self.prepare_metadata(root)
        super().__init__(
            root=root,
            tables=["ptbxl"],
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def _aggregate_labels(
        scp_codes: dict[str, float],
        scp_statements: pd.DataFrame,
    ) -> dict[str, str]:
        """Builds the per-label-set label strings of one ECG.

        Args:
            scp_codes: Mapping from SCP code to likelihood for one ECG.
            scp_statements: The ``scp_statements.csv`` table indexed by code.

        Returns:
            A dict mapping each label set name to a string of the form
            ``"CODE:likelihood;CODE:likelihood"``. Aggregated sets
            (super/sub diagnostic) use the maximum likelihood of the members.
            Empty label sets map to the empty string.
        """
        per_set: dict[str, dict[str, float]] = {name: {} for name in LABEL_SETS}
        for code, likelihood in scp_codes.items():
            if code not in scp_statements.index:
                continue
            row = scp_statements.loc[code]
            likelihood = float(likelihood)
            per_set["all"][code] = likelihood
            if row.get("diagnostic") == 1:
                per_set["diagnostic"][code] = likelihood
                for key, column in (
                    ("subdiagnostic", "diagnostic_subclass"),
                    ("superdiagnostic", "diagnostic_class"),
                ):
                    parent = row.get(column)
                    if isinstance(parent, str) and parent:
                        previous = per_set[key].get(parent, float("-inf"))
                        per_set[key][parent] = max(previous, likelihood)
            if row.get("form") == 1:
                per_set["form"][code] = likelihood
            if row.get("rhythm") == 1:
                per_set["rhythm"][code] = likelihood
        return {
            name: ";".join(f"{c}:{v}" for c, v in sorted(labels.items()))
            for name, labels in per_set.items()
        }

    def prepare_metadata(self, root: str) -> None:
        """Creates ``ptbxl-metadata-pyhealth.csv`` from the raw PTB-XL files.

        The CSV has one row per ECG with absolute record paths (without file
        extension), the recommended fold, and label strings for every label
        set (see :data:`LABEL_SETS`).

        Args:
            root: Root directory containing the raw PTB-XL files.

        Raises:
            FileNotFoundError: If ``ptbxl_database.csv`` or
                ``scp_statements.csv`` is missing from ``root``.
        """
        database_path = os.path.join(root, "ptbxl_database.csv")
        statements_path = os.path.join(root, "scp_statements.csv")
        for path in (database_path, statements_path):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{path} not found. Download PTB-XL 1.0.3 from "
                    "https://physionet.org/content/ptb-xl/1.0.3/ and pass the "
                    "extracted folder as `root`."
                )
        database = pd.read_csv(database_path, index_col="ecg_id")
        statements = pd.read_csv(statements_path, index_col=0)

        rows: list[dict[str, object]] = []
        for ecg_id, record in database.iterrows():
            scp_codes = ast.literal_eval(record["scp_codes"])
            labels = self._aggregate_labels(scp_codes, statements)
            row: dict[str, object] = {
                "patient_id": str(int(record["patient_id"])),
                "ecg_id": int(ecg_id),
                "age": record.get("age"),
                "sex": record.get("sex"),
                "strat_fold": int(record["strat_fold"]),
                "record_path_lr": os.path.join(root, record["filename_lr"]),
                "record_path_hr": os.path.join(root, record["filename_hr"]),
                "scp_codes": record["scp_codes"],
            }
            row.update({f"labels_{name}": text for name, text in labels.items()})
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(root, "ptbxl-metadata-pyhealth.csv"), index=False
        )

    @property
    def default_task(self) -> PTBXLMultilabelClassification:
        """Returns the default task for this dataset.

        Returns:
            PTBXLMultilabelClassification: Multi-label diagnosis from the
                100 Hz, 12-lead ECG using the ``"diagnostic"`` label set.
        """
        return PTBXLMultilabelClassification()
