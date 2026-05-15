import io
import logging
import os
import tarfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class HiRIDDataset(BaseDataset):
    """HiRID v1.1.1 ICU dataset.

    The HiRID (High time Resolution ICU Dataset) is a freely accessible
    critical care dataset containing data from ~34,000 ICU admissions
    to the Department of Intensive Care Medicine of the Bern University
    Hospital, Switzerland.

    The dataset provides high-resolution time-series data (2-minute
    intervals) including vital signs, lab values, ventilator parameters,
    and medication records.

    Dataset link:
        https://physionet.org/content/hirid/1.1.1/

    Paper:
        Hyland et al. "Early prediction of circulatory failure in the
        intensive care unit using machine learning." Nature Medicine, 2020.

    Args:
        root: Root directory of the HiRID dataset.
        stage: Data processing stage to use. One of ``"merged"``,
            ``"raw"``, or ``"imputed"``.
        dataset_name: Name of the dataset. Defaults to ``"hirid"``.
        config_path: Path to YAML config. Defaults to built-in config.
        **kwargs: Additional arguments passed to
            :class:`~pyhealth.datasets.BaseDataset`.

    Attributes:
        stage: The selected data processing stage.

    Examples:
        >>> from pyhealth.datasets import HiRIDDataset
        >>> dataset = HiRIDDataset(
        ...     root="/path/to/hirid/1.1.1",
        ...     stage="merged",
        ... )
        >>> dataset.stats()
        >>> patient = dataset.get_patient("1")
    """

    MERGED_COLUMN_MAP: Dict[str, str] = {
        "vm1": "heart_rate",
        "vm3": "systolic_bp_invasive",
        "vm4": "diastolic_bp_invasive",
        "vm5": "mean_arterial_pressure",
        "vm13": "cardiac_output",
        "vm20": "spo2",
        "vm28": "rass",
        "vm62": "peak_inspiratory_pressure",
        "vm136": "lactate_arterial",
        "vm146": "lactate_venous",
        "vm172": "inr",
        "vm174": "serum_glucose",
        "vm176": "c_reactive_protein",
        "pm41": "dobutamine",
        "pm42": "milrinone",
        "pm43": "levosimendan",
        "pm44": "theophyllin",
        "pm87": "non_opioid_analgesics",
    }

    def __init__(
        self,
        root: str,
        stage: str = "merged",
        dataset_name: str = "hirid",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if stage not in {"merged", "raw", "imputed"}:
            raise ValueError(
                "stage must be one of 'merged', 'raw', or 'imputed', "
                f"got '{stage}'"
            )

        self.stage = stage

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "hirid.yaml")

        self._verify_data(root, stage)
        self._prepare_data(root, stage)

        if stage == "merged":
            tables = ["general_table", "merged_stage"]
        elif stage == "raw":
            tables = ["general_table", "observation_tables", "pharma_records"]
        else:
            tables = ["general_table", "imputed_stage"]

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs,
        )

    def _verify_data(self, root: str, stage: str) -> None:
        if not os.path.exists(root):
            raise FileNotFoundError(f"Dataset path does not exist: {root}")

        general_table_path = os.path.join(root, "general_table.csv")
        if not os.path.exists(general_table_path):
            raise FileNotFoundError(
                f"Required file not found: {general_table_path}"
            )

        requirements = self._stage_requirements(root, stage)
        for processed_path, tar_path in requirements:
            if os.path.exists(processed_path):
                continue
            if not os.path.exists(tar_path):
                raise FileNotFoundError(f"Required file not found: {tar_path}")

    def _prepare_data(self, root: str, stage: str) -> None:
        if stage == "merged":
            self._prepare_merged(root)
        elif stage == "raw":
            self._prepare_raw(root)
        else:
            self._prepare_imputed(root)

    def _prepare_merged(self, root: str) -> None:
        output_path = os.path.join(root, "hirid-merged-pyhealth.csv")
        if os.path.exists(output_path):
            logger.info("Processed file exists, skipping: %s", output_path)
            return

        tar_path = os.path.join(root, "merged_stage", "merged_stage_csv.tar.gz")
        self._extract_and_concat_tar(
            tar_path=tar_path,
            output_path=output_path,
            column_map=self.MERGED_COLUMN_MAP,
        )

    def _prepare_raw(self, root: str) -> None:
        observation_output_path = os.path.join(
            root, "hirid-observations-pyhealth.csv"
        )
        if os.path.exists(observation_output_path):
            logger.info(
                "Processed file exists, skipping: %s",
                observation_output_path,
            )
        else:
            observation_tar_path = os.path.join(
                root,
                "raw_stage",
                "observation_tables_csv.tar.gz",
            )
            self._extract_and_concat_tar(
                tar_path=observation_tar_path,
                output_path=observation_output_path,
            )

        pharma_output_path = os.path.join(root, "hirid-pharma-pyhealth.csv")
        if os.path.exists(pharma_output_path):
            logger.info("Processed file exists, skipping: %s", pharma_output_path)
            return

        pharma_tar_path = os.path.join(
            root,
            "raw_stage",
            "pharma_records_csv.tar.gz",
        )
        self._extract_and_concat_tar(
            tar_path=pharma_tar_path,
            output_path=pharma_output_path,
        )

    def _prepare_imputed(self, root: str) -> None:
        output_path = os.path.join(root, "hirid-imputed-pyhealth.csv")
        if os.path.exists(output_path):
            logger.info("Processed file exists, skipping: %s", output_path)
            return

        tar_path = os.path.join(root, "imputed_stage", "imputed_stage_csv.tar.gz")
        self._extract_and_concat_tar(
            tar_path=tar_path,
            output_path=output_path,
            column_map=self.MERGED_COLUMN_MAP,
        )

    def _extract_and_concat_tar(
        self,
        tar_path: str,
        output_path: str,
        column_map: Optional[Dict[str, str]] = None,
    ) -> None:
        logger.info("Extracting %s -> %s", tar_path, output_path)
        first = True

        with tarfile.open(tar_path, "r:gz") as tar:
            members = sorted(
                (
                    member
                    for member in tar.getmembers()
                    if member.isfile() and member.name.endswith(".csv")
                ),
                key=lambda member: member.name,
            )
            if not members:
                raise ValueError(f"No CSV files found in archive: {tar_path}")

            for index, member in enumerate(members, start=1):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue

                with io.TextIOWrapper(extracted) as buffer:
                    df = pd.read_csv(buffer)

                if column_map:
                    df.rename(columns=column_map, inplace=True)

                df.to_csv(
                    output_path,
                    mode="w" if first else "a",
                    header=first,
                    index=False,
                )
                first = False

                if index % 1000 == 0:
                    logger.info("Processed %s/%s patient files", index, len(members))

        logger.info("Created %s", output_path)

    def _stage_requirements(self, root: str, stage: str) -> List[Tuple[str, str]]:
        if stage == "merged":
            return [
                (
                    os.path.join(root, "hirid-merged-pyhealth.csv"),
                    os.path.join(
                        root,
                        "merged_stage",
                        "merged_stage_csv.tar.gz",
                    ),
                )
            ]

        if stage == "raw":
            return [
                (
                    os.path.join(root, "hirid-observations-pyhealth.csv"),
                    os.path.join(
                        root,
                        "raw_stage",
                        "observation_tables_csv.tar.gz",
                    ),
                ),
                (
                    os.path.join(root, "hirid-pharma-pyhealth.csv"),
                    os.path.join(
                        root,
                        "raw_stage",
                        "pharma_records_csv.tar.gz",
                    ),
                ),
            ]

        return [
            (
                os.path.join(root, "hirid-imputed-pyhealth.csv"),
                os.path.join(
                    root,
                    "imputed_stage",
                    "imputed_stage_csv.tar.gz",
                ),
            )
        ]
