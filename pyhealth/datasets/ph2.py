# Contributor: Aayan Verma (aayanv2)
# Contributor: Bryson Lee (brysonl3)
# Contributor: Xin Zhang (xinz13)
# Paper: A Study of Artifacts on Melanoma Classification under Diffusion-Based
# Perturbations (Jin and Ghassemi, CHIL 2025)
# https://proceedings.mlr.press/v287/jin25b.html
"""PH2 dermoscopy dataset (200 images with lesion masks).

PH2 is the 200-image dermoscopy set whose lesion masks the paper's lesion-only,
background-only and bounding-box views need. Pair this dataset with
:class:`~pyhealth.tasks.PH2ArtifactClassification`.
"""

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

#: Clinical diagnoses, indexed by the PH2 clinical-diagnosis code 0, 1, 2.
PH2_DIAGNOSES = ("common_nevus", "atypical_nevus", "melanoma")

_METADATA_FILE = "ph2-metadata-pyhealth.csv"
_IMAGE_FOLDER_SUFFIX = "_Dermoscopic_Image"
_MASK_FOLDER_SUFFIX = "_lesion"
_LABEL_FILE_STEM = "PH2_dataset"


class PH2Dataset(BaseDataset):
    """PH2 dermoscopy dataset: 200 images with expert lesion segmentations.

    200 dermoscopic images (80 common nevi, 80 atypical nevi, 40 melanomas)
    acquired at the Dermatology Service of Hospital Pedro Hispano, Matosinhos,
    Portugal. Every image comes with a binary lesion mask, which is what makes
    the lesion-only, background-only and bounding-box views of
    :class:`~pyhealth.tasks.PH2ArtifactClassification` possible. Each image
    is its own PyHealth patient (``IMDxxx``): the dataset does not link images
    to people.

    Getting the data:
        - Source: https://www.fc.up.pt/addi/ph2%20database.html
        - Access: the archive is released for research and educational use;
          the authors ask users to cite the paper below. Download it from
          the page above (the page may ask for a short form) and unpack it.
        - ``root`` layout (the folder that contains the ``IMD*`` folders and
          the annotation file; sub-folders are searched recursively)::

              PH2Dataset/                                    <- root
                  PH2_dataset.txt    (or .xlsx / .csv)
                  PH2 Dataset images/
                      IMD002/
                          IMD002_Dermoscopic_Image/IMD002.bmp
                          IMD002_lesion/IMD002_lesion.bmp
                      IMD003/ ...

        - Annotations: the labels are read from the ``Clinical Diagnosis``
          column (0 common nevus, 1 atypical nevus, 2 melanoma) of the
          archive's ``PH2_dataset.txt``, or of ``PH2_dataset.csv`` or
          ``PH2_dataset.xlsx`` (needs ``openpyxl``). The first column must be
          the image name (``IMD002``).

    Licence:
        Research and educational use only; cite the paper below.

    Citation:
        Mendonça, T., Ferreira, P. M., Marques, J. S., Marcal, A. R. S.,
        Rozeira, J. PH2 - A dermoscopic image database for research and
        benchmarking. 35th Annual International Conference of the IEEE
        Engineering in Medicine and Biology Society (EMBC), 2013.
        https://doi.org/10.1109/EMBC.2013.6610779

    Args:
        root: Folder holding the annotation file and the ``IMD*`` folders.
        dataset_name: Name of the dataset. Defaults to ``"ph2"``.
        config_path: Table configuration YAML. Defaults to the bundled one.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the metadata CSV. Defaults to
            ``<cache_dir or the PyHealth cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import PH2Dataset
        >>> dataset = PH2Dataset(root="/path/to/PH2Dataset")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task()  # doctest: +SKIP
    """

    default_dataset_name = "ph2"
    default_table = "ph2"

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        self.source_root = Path(root).expanduser().resolve()
        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_FILE
        marker = self.metadata_dir / "ph2-source.txt"
        source_key = f"root={self.source_root}"
        with FileLock(str(self.metadata_path) + ".lock"):
            current = marker.read_text().strip() if marker.exists() else None
            if (
                refresh_metadata
                or not self.metadata_path.exists()
                or current != source_key
            ):
                self.prepare_metadata()
                marker.write_text(source_key)

        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "ph2.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _read_diagnoses(self) -> dict[str, str]:
        """Return ``{image name: diagnosis}`` from the annotation file.

        Called by :meth:`prepare_metadata`. Uses the first of ``PH2_dataset.txt``,
        ``.csv`` or ``.xlsx`` (in that order) found under the source folder.

        Returns:
            Dict mapping an image name such as ``"IMD003"`` to one of
            ``PH2_DIAGNOSES``.

        Raises:
            FileNotFoundError: If no annotation file is found.
            ImportError: If only the ``.xlsx`` exists and ``openpyxl`` is missing.
            ValueError: If the file has no ``Clinical Diagnosis`` column or an
                unknown diagnosis code.
        """
        found = sorted(self.source_root.rglob(f"{_LABEL_FILE_STEM}.*"))
        candidates = [
            path
            for suffix in (".txt", ".csv", ".xlsx")  # the order the docs list
            for path in found
            if path.suffix.lower() == suffix
        ]
        for path in candidates:
            if path.suffix.lower() == ".xlsx":
                try:
                    table = pd.read_excel(path, header=None)
                except ImportError as error:
                    raise ImportError(
                        f"Reading {path.name} needs openpyxl "
                        "(pip install openpyxl), or export it to "
                        f"{_LABEL_FILE_STEM}.csv."
                    ) from error
            elif path.suffix.lower() == ".csv":
                table = pd.read_csv(path, header=None)
            else:
                table = self._read_pipe_table(path)
            return self._parse_annotations(table, path)
        raise FileNotFoundError(
            f"No {_LABEL_FILE_STEM}.txt, .csv or .xlsx found under "
            f"{self.source_root}. It holds the clinical diagnoses and ships "
            "with the PH2 archive."
        )

    @staticmethod
    def _read_pipe_table(path: Path) -> pd.DataFrame:
        """Read the archive's ``PH2_dataset.txt``, a ``||``/``|`` separated table.

        Empty cells (such as a missing histological diagnosis) are kept, so
        columns stay aligned. Called by :meth:`_read_diagnoses`.

        Args:
            path: Path to ``PH2_dataset.txt``.

        Returns:
            A table of strings with one row per text line and no header row
            marked; :meth:`_parse_annotations` finds the header by its text.
        """
        rows = []
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip().replace("||", "|").strip("|")
            if line:
                rows.append([cell.strip() for cell in line.split("|")])
        return pd.DataFrame(rows)

    @staticmethod
    def _parse_annotations(table: pd.DataFrame, path: Path) -> dict[str, str]:
        """Map image name to diagnosis, finding the header row by its text.

        Called by :meth:`_read_diagnoses` for every supported file format.

        Args:
            table: Raw annotation table, with a possible preamble above the
                header row and the image name in the first column.
            path: Path the table was read from, used in error messages.

        Returns:
            Dict mapping an image name such as ``"IMD003"`` to one of
            ``PH2_DIAGNOSES``. Rows whose name does not start with ``IMD`` or
            whose code is empty are skipped.

        Raises:
            ValueError: If there is no ``Clinical Diagnosis`` column or a code
                is not 0, 1 or 2.
        """
        text = table.astype(str).apply(lambda col: col.str.strip().str.lower())
        header_rows = text.index[
            text.apply(lambda row: row.str.contains("clinical diagnosis").any(), axis=1)
        ]
        if len(header_rows) == 0:
            raise ValueError(f"{path.name}: no 'Clinical Diagnosis' column found.")
        header = int(header_rows[0])
        column = int(
            text.loc[header].str.contains("clinical diagnosis").to_numpy().argmax()
        )
        diagnoses = {}
        for _, row in table.iloc[header + 1 :].iterrows():
            name, code = str(row.iloc[0]).strip(), row.iloc[column]
            if (
                not name.upper().startswith("IMD")
                or pd.isna(code)
                or str(code).strip() == ""
            ):
                continue
            if int(code) not in range(len(PH2_DIAGNOSES)):
                raise ValueError(f"{path.name}: {name} has clinical diagnosis {code}.")
            diagnoses[name] = PH2_DIAGNOSES[int(code)]
        return diagnoses

    def prepare_metadata(self) -> None:
        """Write the metadata CSV, one row per image, to ``metadata_dir``.

        Images listed in the annotation file but missing a dermoscopic image
        or a lesion mask on disk are dropped with a warning, so a partial
        copy still loads.

        Raises:
            FileNotFoundError: If the annotation file is missing, or no image
                with a lesion mask is found.
            ValueError: If the annotation file has no ``Clinical Diagnosis``
                column or an unknown diagnosis code.

        Examples:
            >>> dataset = PH2Dataset(root="/path/to/PH2Dataset")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        diagnoses = self._read_diagnoses()
        rows, dropped = [], []
        for name in sorted(diagnoses):
            images = list(
                self.source_root.rglob(f"{name}{_IMAGE_FOLDER_SUFFIX}/{name}.bmp")
            )
            masks = list(
                self.source_root.rglob(
                    f"{name}{_MASK_FOLDER_SUFFIX}/{name}{_MASK_FOLDER_SUFFIX}.bmp"
                )
            )
            if not images or not masks:
                dropped.append(name)
                continue
            rows.append(
                {
                    "patient_id": name,
                    "image_id": name,
                    "image_path": str(images[0]),
                    "mask_path": str(masks[0]),
                    "diagnosis": diagnoses[name],
                }
            )
        if dropped:
            logger.warning(
                "Dropping %d annotated images without a dermoscopic image or "
                "lesion mask on disk: %s",
                len(dropped),
                dropped[:5],
            )
        if not rows:
            raise FileNotFoundError(
                f"No PH2 images with lesion masks found under {self.source_root}."
            )
        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)
        logger.info("Wrote %d PH2 images to %s", len(rows), self.metadata_path)

    @property
    def default_task(self) -> Any:
        """The default task, :class:`~pyhealth.tasks.PH2ArtifactClassification`.

        Returns:
            A task with the whole-image view, no artifact and 224 px images.

        Examples:
            >>> dataset = PH2Dataset(root="/path/to/PH2Dataset")  # doctest: +SKIP
            >>> dataset.default_task.view  # doctest: +SKIP
            'whole'
        """
        from ..tasks.ph2_artifact_classification import PH2ArtifactClassification

        return PH2ArtifactClassification()
