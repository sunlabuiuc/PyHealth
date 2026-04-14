"""VQA-RAD dataset for medical Visual Question Answering.

The VQA-RAD dataset (Lau et al., 2018) contains 315 radiology images
with 3,515 question-answer pairs spanning multiple imaging modalities
(CT, MRI, X-ray) and organs (head, chest, abdomen). Questions are both
open-ended and closed-ended (yes/no).

The dataset is commonly used to evaluate medical VQA models such as
MedFlamingo (Moor et al., 2023).

Download:
    The dataset can be obtained from:
    https://osf.io/89kps/

    Expected directory structure after download::

        root/
            VQA_RAD Dataset Public.json

    The official OSF archive may keep images in ``VQA_RAD Image Folder/``
    rather than ``images/``. This loader accepts either layout and rewrites
    the raw export into ``vqarad-metadata-pyhealth.csv`` for the standard
    PyHealth pipeline.

Citation:
    Lau, J. J., Gayen, S., Ben Abacha, A., & Demner-Fushman, D. (2018).
    A dataset of clinically generated visual questions and answers about
    radiology images. Scientific Data, 5, 180251.
"""

import json
import logging
from functools import wraps
from pathlib import Path
from typing import Optional

import pandas as pd

from ..processors import ImageProcessor
from ..tasks import MedicalVQATask
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class VQARADDataset(BaseDataset):
    """Dataset for VQA-RAD (Visual Question Answering in Radiology).

    Loads the VQA-RAD JSON file and converts it into a flat CSV that the
    PyHealth ``BaseDataset`` pipeline can ingest. Each row represents one
    (image, question, answer) triplet.

    Args:
        root: Root directory containing the VQA-RAD data files.
            Expected to contain ``VQA_RAD Dataset Public.json`` and an
            ``images/`` subdirectory with the radiology images.
        dataset_name: Optional name. Defaults to ``"vqarad"``.
        config_path: Optional path to a YAML config. If ``None``, uses the
            bundled ``configs/vqarad.yaml``.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of parallel workers. Defaults to 1.
        dev: If ``True``, loads a small subset for development.

    Examples:
        >>> from pyhealth.datasets import VQARADDataset
        >>> dataset = VQARADDataset(root="/path/to/vqarad")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir=None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "vqarad.yaml"

        metadata_path = Path(root) / "vqarad-metadata-pyhealth.csv"
        if not metadata_path.exists():
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["vqarad"],
            dataset_name=dataset_name or "vqarad",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self, root: str) -> None:
        """Convert the raw VQA-RAD JSON into a flat CSV.

        The JSON file contains a list of QA entries, each with fields like
        ``"IMAGES_PATH"``, ``"QUESTION"``, and ``"ANSWER"``. This method
        normalizes them into a CSV with columns matching the YAML config.

        Args:
            root: Root directory containing ``VQA_RAD Dataset Public.json``.
        """
        root_path = Path(root)
        json_path = root_path / "VQA_RAD Dataset Public.json"
        if not json_path.exists():
            raise FileNotFoundError(
                f"Expected VQA-RAD JSON at {json_path}. "
                "Download the dataset from https://osf.io/89kps/"
            )

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []
        for entry in data:
            image_name = entry.get("IMAGE_PATH", entry.get("IMAGES_PATH", ""))
            rows.append(
                {
                    "image_path": str(root_path / "images" / image_name),
                    "question": entry.get("QUESTION", ""),
                    "answer": str(entry.get("ANSWER", "")),
                    "answer_type": entry.get("ANSWER_TYPE", ""),
                    "question_type": entry.get("QUESTION_TYPE", ""),
                    "image_organ": entry.get("IMAGE_ORGAN", ""),
                }
            )

        metadata_path = root_path / "vqarad-metadata-pyhealth.csv"
        pd.DataFrame(rows).to_csv(metadata_path, index=False)
        logger.info("Saved VQA-RAD metadata (%s rows) to %s", len(rows), metadata_path)

    @property
    def default_task(self) -> MedicalVQATask:
        """Returns the default task for this dataset."""
        return MedicalVQATask()

    @wraps(BaseDataset.set_task)
    def set_task(self, *args, **kwargs):
        input_processors = kwargs.get("input_processors", None)
        if input_processors is None:
            input_processors = {}

        if "image" not in input_processors:
            input_processors["image"] = ImageProcessor(mode="RGB", image_size=224)

        kwargs["input_processors"] = input_processors
        return super().set_task(*args, **kwargs)

    set_task.__doc__ = (
        f"{set_task.__doc__}\n"
        "        Note:\n"
        "            If no image processor is provided, a default RGB "
        "`ImageProcessor(mode='RGB', image_size=224)` is injected so VQA-RAD "
        "images are loaded with the expected channel format and resolution."
    )
