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
import os
from functools import wraps
from pathlib import Path
from typing import Optional

import pandas as pd

from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.processors.image_processor import ImageProcessor

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
            Expected to contain ``VQA_RAD Dataset Public.json`` and either
            an ``images/`` subdirectory or the original OSF
            ``VQA_RAD Image Folder/`` directory with the radiology images.
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
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "vqarad.yaml"

        metadata_csv = os.path.join(root, "vqarad-metadata-pyhealth.csv")
        if not os.path.exists(metadata_csv):
            self.prepare_metadata(root)

        default_tables = ["vqarad"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "vqarad",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self, root: str) -> None:
        """Convert the raw VQA-RAD JSON into a flat CSV.

        The raw VQA-RAD export may come from different mirrors. This method
        accepts both the original OSF field names (for example
        ``image_name``, ``question``, ``answer``) and alternate uppercase
        field names (for example ``IMAGE_PATH``, ``QUESTION``, ``ANSWER``),
        then normalizes them into a CSV with columns matching the YAML config.

        Args:
            root: Root directory containing ``VQA_RAD Dataset Public.json``.
        """
        json_path = os.path.join(root, "VQA_RAD Dataset Public.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"Expected VQA-RAD JSON at {json_path}. "
                "Download the dataset from https://osf.io/89kps/"
            )

        with open(json_path, "r") as f:
            data = json.load(f)

        image_root = self._resolve_image_root(root)
        rows = []
        for entry in data:
            image_name = (
                entry.get("IMAGE_PATH")
                or entry.get("IMAGES_PATH")
                or entry.get("image_name")
                or ""
            )
            image_path = os.path.join(image_root, image_name) if image_name else ""
            rows.append(
                {
                    "image_path": image_path,
                    "question": entry.get("QUESTION", entry.get("question", "")),
                    "answer": str(entry.get("ANSWER", entry.get("answer", ""))),
                    "answer_type": entry.get(
                        "ANSWER_TYPE", entry.get("answer_type", "")
                    ),
                    "question_type": entry.get(
                        "QUESTION_TYPE", entry.get("question_type", "")
                    ),
                    "image_organ": entry.get(
                        "IMAGE_ORGAN", entry.get("image_organ", "")
                    ),
                }
            )

        df = pd.DataFrame(rows)

        # Filter out rows whose image file is missing so that the processor
        # pipeline does not fail on incomplete dataset downloads.
        before = len(df)
        df = df[df["image_path"].apply(lambda p: bool(p) and os.path.isfile(p))]
        skipped = before - len(df)
        if skipped:
            logger.warning(
                f"Skipped {skipped} entries with missing image files "
                f"(out of {before} total)."
            )

        out_path = os.path.join(root, "vqarad-metadata-pyhealth.csv")
        df.to_csv(out_path, index=False)
        logger.info(f"Saved VQA-RAD metadata ({len(df)} rows) to {out_path}")

    @staticmethod
    def _resolve_image_root(root: str) -> str:
        """Finds the VQA-RAD image directory for the supported raw layouts."""
        candidate_dirs = [
            os.path.join(root, "images"),
            os.path.join(root, "VQA_RAD Image Folder"),
        ]

        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                return candidate

        raise FileNotFoundError(
            "Expected VQA-RAD images in either "
            f"{candidate_dirs[0]} or {candidate_dirs[1]}."
        )

    @property
    def default_task(self) -> MedicalVQATask:
        """Returns the default task for this dataset.

        Returns:
            A :class:`~pyhealth.tasks.MedicalVQATask` instance.
        """
        return MedicalVQATask()

    @wraps(BaseDataset.set_task)
    def set_task(
        self,
        *args,
        image_processor: Optional[FeatureProcessor] = None,
        **kwargs,
    ) -> SampleDataset:
        """Set a task and inject the default image processor when needed."""
        input_processors = kwargs.get("input_processors", None)

        if input_processors is None:
            input_processors = {}

        if image_processor is None:
            image_processor = ImageProcessor(mode="RGB", image_size=224)

        if "image" not in input_processors:
            input_processors["image"] = image_processor

        kwargs["input_processors"] = input_processors
        return super().set_task(*args, **kwargs)
