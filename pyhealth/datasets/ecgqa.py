"""
ECG Question Answering dataset.

This dataset provides natural language question-answer pairs linked to
ECG recordings via ecg_id. It is an annotation layer on top of ECG
recordings from PTB-XL or MIMIC-IV-ECG.

The QA data originates from the ECG-QA dataset (Oh et al., 2024),
restructured for few-shot learning by Tang et al. (CHIL 2025).

Dataset link:
    Dataset is available at https://github.com/Tang-Jia-Lu/FSL_ECG_QA

Dataset paper:
    J. Tang, T. Xia, Y. Lu, C. Mascolo, and A. Saeed, "Electrocardiogram-language model
    for few-shot question answering with meta learning,"
    arXiv preprint arXiv:2410.14464, 2024.

Dataset paper link:
    https://arxiv.org/abs/2410.14464

Author:
    Jovian Wang (jovianw2@illinois.edu)
    Matthew Pham (mdpham2@illinois.edu)
    Yiyun Wang (yiyunw3@illinois.edu)
"""
import hashlib
import json
import logging
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

_VALID_ECG_SOURCES = {"ptbxl": "ptbxl", "mimic": "mimic-iv-ecg"}

"""
Three question types are supported:
    - single-verify: yes/no questions about ECG findings
    - single-choose: multi-choice questions (answer is one option, "both", or "none")
    - single-query: open-ended questions with free-form answers

Args:
    root: directory that holds (or will hold) the paraphrased QA splits as
        train/, valid/, test/ subdirectories of JSON files.
    dataset_name: name of the dataset. Default is "ecg_qa".
    config_path: path to the YAML config file. Default uses built-in config.
    download: if True, download the chosen variant from GitHub into ``root``
        before loading. Defaults to False.
    ecg_source: which underlying ECG dataset the QA pairs are grounded in.
        One of ``"ptbxl"`` (PTB-XL) or ``"mimic"`` (MIMIC-IV-ECG).
        Defaults to ``"ptbxl"``.
    include_demographics: if True, download the modified variant whose
        question text includes patient sex and age. Defaults to False
        (the original Tang et al. release).

Examples:
    >>> from pyhealth.datasets import ECGQADataset
    >>> # Use a pre-downloaded local copy
    >>> dataset = ECGQADataset(
    ...     root="/path/to/ecgqa/ptbxl/paraphrased/",
    ... )
    >>> # Or download the modified PTB-XL variant on the fly
    >>> dataset = ECGQADataset(
    ...     root="./ecg_qa_ptbxl_demo",
    ...     download=True,
    ...     ecg_source="ptbxl",
    ...     include_demographics=True,
    ... )
    >>> dataset.stats()
"""
class ECGQADataset(BaseDataset):


    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        download: bool = False,
        ecg_source: str = "ptbxl",
        include_demographics: bool = False,
        **kwargs,
    ) -> None:
        if ecg_source not in _VALID_ECG_SOURCES:
            raise ValueError(
                f"ecg_source must be one of {sorted(_VALID_ECG_SOURCES)}, "
                f"got {ecg_source!r}"
            )

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ecg_qa.yaml"

        self.root = root

        if download:
            self._download_data(root, ecg_source, include_demographics)
        self._verify_data(root)

        self.prepare_metadata()

        super().__init__(
            root=root,
            tables=["ecg_qa"],
            dataset_name=dataset_name or "ecg_qa",
            config_path=config_path,
            **kwargs,
        )

    def prepare_metadata(self) -> None:
        """Build and save a metadata CSV from all ECG-QA JSON files.

        Scans train/, valid/, test/ subdirectories under root, loads all
        JSON files, filters to single-* question types, and writes a
        single CSV with columns:
            patient_id, ecg_id, question, answer, question_type,
            attribute_type, template_id, question_id, sample_id, attribute
        """
        root = Path(self.root)
        csv_path = root / "ecg-qa-pyhealth.csv"
        if csv_path.exists():
            return

        data = []
        for split_dir in ("train", "valid", "test"):
            for fpath in sorted((root / split_dir).glob("*.json")):
                with open(fpath, "r") as f:
                    data.extend(json.load(f))

        rows: list[dict] = []
        for record in data:
            qt = record.get("question_type", "")
            if not qt.startswith("single-"):
                continue

            ecg_id = record["ecg_id"][0]
            answer = ";".join(record["answer"])
            attribute = ";".join(record.get("attribute", []))

            rows.append({
                "patient_id": f"{ecg_id:05d}",
                "ecg_id": ecg_id,
                "question": record["question"],
                "answer": answer,
                "question_type": qt,
                "attribute_type": record.get("attribute_type", ""),
                "template_id": record.get("template_id", 0),
                "question_id": record.get("question_id", 0),
                "sample_id": record.get("sample_id", 0),
                "attribute": attribute,
            })

        if not rows:
            raise ValueError("No single-* question type records found in JSON data")

        df = pd.DataFrame(rows)
        df.sort_values(["patient_id", "question_type", "template_id"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Wrote metadata to {csv_path}")

    def _download_data(
        self, root: str, ecg_source: str, include_demographics: bool
    ) -> None:
        """Downloads the requested ECG-QA dataset from GitHub into ``root``.

        Fetches a commit-pinned tarball from the original ``Tang-Jia-Lu`` repo
        (or a modified fork when ``include_demographics`` is True),
        verifies its MD5, and extracts only the
        ``ecgqa/<ecg_source>/paraphrased/{train,valid,test}/`` subtree directly
        into ``root``. The tarball is deleted after extraction.

        Args:
            root: directory the splits will land in.
            ecg_source: ``"ptbxl"`` or ``"mimic"``.
            include_demographics: selects the modified variant when True.

        Raises:
            ValueError: if the downloaded tarball fails MD5 verification or
                if it contains an unsafe path during extraction.
        """
        # URLs are pinned to specific commit SHAs so the MD5s below stay stable
        # even if either repo gains new commits later.
        if include_demographics:
            url = (
                "https://github.com/jovianw/FSL_ECG_QA/archive/"
                "2e2d4ac185d6069c741d083269ea40ca01bfd50b.tar.gz"
            )
            expected_md5 = "e65c4b6ae127103ad92a33ec9246039e"
            archive_prefix = "FSL_ECG_QA-2e2d4ac185d6069c741d083269ea40ca01bfd50b"
        else:
            url = (
                "https://github.com/Tang-Jia-Lu/FSL_ECG_QA/archive/"
                "b0ec9bd84ae2337052ca977941e37a703dcb492e.tar.gz"
            )
            expected_md5 = "894b4af304e99c48ecd62a914ba3ba2b"
            archive_prefix = "FSL_ECG_QA-b0ec9bd84ae2337052ca977941e37a703dcb492e"

        os.makedirs(root, exist_ok=True)
        archive_path = os.path.join(root, "ecgqa-download.tar.gz")

        logger.info(f"Downloading {url} -> {archive_path}")
        urllib.request.urlretrieve(url, archive_path)

        logger.info(f"Checking MD5 checksum for {archive_path}...")
        with open(archive_path, "rb") as f:
            file_md5 = hashlib.md5(f.read()).hexdigest()
        if file_md5 != expected_md5:
            msg = (
                f"Invalid MD5 checksum for {archive_path}: "
                f"expected {expected_md5}, got {file_md5}"
            )
            logger.error(msg)
            raise ValueError(msg)

        ecg_source_dir = _VALID_ECG_SOURCES[ecg_source]
        prefix = f"{archive_prefix}/ecgqa/{ecg_source_dir}/paraphrased/"
        abs_root = os.path.abspath(root)

        logger.info(f"Extracting {prefix}* from {archive_path} into {root}")
        with tarfile.open(archive_path, "r:gz") as tar:
            for member in tar.getmembers():
                if not member.name.startswith(prefix):
                    continue
                rel = member.name[len(prefix):]
                if not rel:
                    continue

                target_path = os.path.abspath(os.path.join(abs_root, rel))
                if os.path.commonpath([abs_root]) != os.path.commonpath(
                    [abs_root, target_path]
                ):
                    msg = f"Unsafe path detected in tar file: '{member.name}'!"
                    logger.error(msg)
                    raise ValueError(msg)

                member.name = rel
                tar.extract(member, path=root)

        os.remove(archive_path)
        logger.info("Download complete")

    def _verify_data(self, root: str) -> None:
        """Verifies the presence and structure of the dataset directory.

        Checks that ``root`` exists, that ``train/``, ``valid/``, and ``test/``
        subdirectories are present, and that each contains at least one
        ``*.json`` file.

        Args:
            root: directory expected to hold the dataset splits.

        Raises:
            FileNotFoundError: if ``root`` or any required split directory is
                missing.
            ValueError: if a split directory contains no JSON files.
        """
        if not os.path.exists(root):
            msg = f"Dataset path does not exist: {root}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        for split in ("train", "valid", "test"):
            split_dir = os.path.join(root, split)
            if not os.path.isdir(split_dir):
                msg = (
                    f"Dataset path must contain a '{split}' subdirectory: "
                    f"{split_dir}"
                )
                logger.error(msg)
                raise FileNotFoundError(msg)
            if not list(Path(split_dir).glob("*.json")):
                msg = f"Dataset '{split}' directory must contain JSON files!"
                logger.error(msg)
                raise ValueError(msg)

    @property
    def default_task(self):
        """Returns the default task for the ECG-QA dataset: ECGQA."""
        from pyhealth.tasks import ECGQA
        return ECGQA()
