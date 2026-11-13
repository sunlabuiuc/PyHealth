"""Medical Segmentation Decathlon datasets.

The MSD tasks share the same ``dataset.json`` layout. These loaders build a
small metadata table with image and label paths; pixel data is read later by
``NiftiProcessor``.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from .base_dataset import BaseDataset
from .imaging.io import (
    _absolute_path,
    _download_archives,
    _resolve_metadata_dir,
    _strip_nii_suffix,
)

logger = logging.getLogger(__name__)


class MSDSegmentationDataset(BaseDataset):
    """Base loader for one Medical Segmentation Decathlon task.

    The dataset writes ``msd-metadata-pyhealth.csv`` to a local metadata
    directory. It never reads NIfTI pixels and never writes into the source
    dataset directory.

    Getting the data:
        - Source: http://medicaldecathlon.com/ (Simpson et al., 2019,
          https://arxiv.org/abs/1902.09063)
        - Access: open, licensed CC-BY-SA 4.0.
        - Download: pass ``download=True`` to fetch and extract the task's
          ``TaskXX_<Name>.tar`` from the MONAI mirror
          (https://msd-for-monai.s3-us-west-2.amazonaws.com/). The site also
          offers the same archives via AWS and Google Drive.
        - ``root``: the extracted task folder holding ``dataset.json``,
          ``imagesTr/``, and ``labelsTr/``.

    Args:
        root: Root directory of one MSD task, containing ``dataset.json``.
        dataset_name: Optional dataset name.
        config_path: Optional dataset table configuration.
        cache_dir: Directory for processed caches and default metadata.
        metadata_dir: Explicit directory for the generated metadata CSV.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        download: Download and extract the MSD task when it is missing.
            Archive checksums are not verified automatically.
        refresh_metadata: Regenerate the metadata CSV even if it exists.

    Examples:
        >>> from pyhealth.datasets import MSDHippocampusDataset
        >>> dataset = MSDHippocampusDataset(  # doctest: +SKIP
        ...     root="/path/to/Task04_Hippocampus"
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "msd"
    default_table = "cases"
    download_url: str | None = None

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        download: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        """Build metadata and initialize the MSD dataset.

        Args:
            root: Root directory of one MSD task.
            dataset_name: Optional dataset name.
            config_path: Optional table configuration path.
            cache_dir: Directory for processed caches.
            metadata_dir: Explicit metadata output directory.
            num_workers: Number of workers for dataset processing.
            dev: Whether to limit the dataset to 1000 patients.
            download: Download and extract the MSD task when it is missing.
            refresh_metadata: Regenerate the metadata CSV.
        """
        source_root = Path(root).expanduser().resolve()
        if download and not (source_root / "dataset.json").exists():
            if self.download_url is None:
                raise ValueError("This MSD task has no download URL.")
            _download_archives(
                [self.download_url],
                source_root,
                archive_type="tar",
                expected=("dataset.json",),
                extract_to_parent=True,
            )
        dataset_json = source_root / "dataset.json"
        if not dataset_json.exists():
            raise FileNotFoundError(
                f"MSD dataset.json not found under {source_root}. "
                "Pass download=True to download this task."
            )

        name = dataset_name or self.default_dataset_name
        self.source_root = source_root
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / "msd-metadata-pyhealth.csv"

        if refresh_metadata or not self.metadata_path.exists():
            self.prepare_metadata()

        if config_path is None:
            config_path = str(
                Path(__file__).parent / "configs" / "msd.yaml"
            )

        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def _modality_values(specification: dict) -> list[str]:
        """Return ordered channel modality names from ``dataset.json``."""
        modalities = specification.get("channel_names") or specification.get(
            "modality", {}
        )
        if isinstance(modalities, dict):
            values = list(modalities.values())
        else:
            values = list(modalities)
        return [str(value) for value in values]

    def prepare_metadata(self) -> None:
        """Create the MSD metadata CSV from ``dataset.json``."""
        with open(self.source_root / "dataset.json") as handle:
            specification = json.load(handle)

        modalities = self._modality_values(specification)
        modality = ",".join(modalities)
        channels = len(modalities) if modalities else 1
        labels = json.dumps(
            specification.get("labels", {}), ensure_ascii=False
        )
        tensor_image_size = str(
            specification.get("tensorImageSize", "")
        )

        rows = []
        for split, key in (("train", "training"), ("test", "test")):
            for item in specification.get(key, []):
                if isinstance(item, dict):
                    image = item.get("image", "")
                    label = item.get("label", "")
                else:
                    image = item
                    label = ""
                if not image:
                    continue
                rows.append(
                    {
                        "patient_id": _strip_nii_suffix(image),
                        "record_id": _strip_nii_suffix(image),
                        "image_path": _absolute_path(self.source_root, image),
                        "label_path": (
                            _absolute_path(self.source_root, label)
                            if label
                            else ""
                        ),
                        "modality": modality,
                        "channels": channels,
                        "split": split,
                        "labels": labels,
                        "tensor_image_size": tensor_image_size,
                    }
                )

        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)
        logger.info(
            "Wrote %d MSD cases to %s", len(rows), self.metadata_path
        )

    @property
    def _specification(self) -> dict:
        """Return the parsed ``dataset.json`` specification."""
        with open(self.source_root / "dataset.json") as handle:
            return json.load(handle)

    @property
    def modality(self) -> str:
        """Return the dataset modality declared in ``dataset.json``."""
        return ",".join(self._modality_values(self._specification))

    @property
    def default_normalization(self) -> str:
        """Return the nnU-Net-style default normalisation mode."""
        return "ct" if "ct" in self.modality.lower() else "mri"

    @property
    def num_classes(self) -> int:
        """Return the number of labels declared in ``dataset.json``."""
        labels = self._specification.get("labels", {})
        return len(labels) if labels else 2

    @property
    def default_task(self):
        """Return the default MSD segmentation task."""
        from ..tasks.msd_segmentation import MSDSegmentationTask

        return MSDSegmentationTask(
            normalize=self.default_normalization,
            num_classes=self.num_classes,
        )


class MSDHippocampusDataset(MSDSegmentationDataset):
    """MSD Task04 Hippocampus MRI segmentation dataset.

    The dataset contains one T1-weighted MRI volume and a left/right
    hippocampus mask per training case.

    Getting the data:
        - Source: http://medicaldecathlon.com/ (``Task04_Hippocampus``)
        - Access: open, licensed CC-BY-SA 4.0.
        - Download: pass ``download=True`` to fetch and extract
          ``Task04_Hippocampus.tar``; see :class:`MSDSegmentationDataset` for details.
        - ``root``: the extracted ``Task04_Hippocampus`` folder holding
          ``dataset.json``.

    Examples:
        >>> from pyhealth.datasets import MSDHippocampusDataset
        >>> dataset = MSDHippocampusDataset(  # doctest: +SKIP
        ...     root="/path/to/Task04_Hippocampus", download=True
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "msd_hippocampus"
    download_url = (
        "https://msd-for-monai.s3-us-west-2.amazonaws.com/"
        "Task04_Hippocampus.tar"
    )


class MSDProstateDataset(MSDSegmentationDataset):
    """MSD Task05 Prostate MRI segmentation dataset.

    Prostate cases contain two MRI channels (T2 and ADC).

    Getting the data:
        - Source: http://medicaldecathlon.com/ (``Task05_Prostate``)
        - Access: open, licensed CC-BY-SA 4.0.
        - Download: pass ``download=True`` to fetch and extract
          ``Task05_Prostate.tar``; see :class:`MSDSegmentationDataset` for details.
        - ``root``: the extracted ``Task05_Prostate`` folder holding
          ``dataset.json``.

    Examples:
        >>> from pyhealth.datasets import MSDProstateDataset
        >>> dataset = MSDProstateDataset(  # doctest: +SKIP
        ...     root="/path/to/Task05_Prostate", download=True
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "msd_prostate"
    download_url = (
        "https://msd-for-monai.s3-us-west-2.amazonaws.com/"
        "Task05_Prostate.tar"
    )


class MSDSpleenDataset(MSDSegmentationDataset):
    """MSD Task09 Spleen CT segmentation dataset.

    Spleen cases contain one abdominal CT volume and a spleen mask.

    Getting the data:
        - Source: http://medicaldecathlon.com/ (``Task09_Spleen``)
        - Access: open, licensed CC-BY-SA 4.0.
        - Download: pass ``download=True`` to fetch and extract
          ``Task09_Spleen.tar``; see :class:`MSDSegmentationDataset` for details.
        - ``root``: the extracted ``Task09_Spleen`` folder holding
          ``dataset.json``.

    Examples:
        >>> from pyhealth.datasets import MSDSpleenDataset
        >>> dataset = MSDSpleenDataset(  # doctest: +SKIP
        ...     root="/path/to/Task09_Spleen", download=True
        ... )
        >>> dataset.stats()  # doctest: +SKIP
    """

    default_dataset_name = "msd_spleen"
    download_url = (
        "https://msd-for-monai.s3-us-west-2.amazonaws.com/"
        "Task09_Spleen.tar"
    )
