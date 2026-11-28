"""Colonoscopy polyp detection: the five-set PraNet benchmark with boxes."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

POLYP_TEST_SETS = ("CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir")
"""The five test sets of the PraNet polyp benchmark."""

_METADATA_FILE = "polyp-metadata-pyhealth.csv"


class PolypDataset(BaseDataset):
    """Polyp bounding boxes on colonoscopy frames (PraNet benchmark).

    The benchmark of Fan et al. (PraNet, MICCAI 2020) pools five public
    polyp segmentation datasets. Its training set has 1,450 images (900 from
    Kvasir-SEG, 550 from CVC-ClinicDB); its test sets are CVC-300 (60
    images, from EndoScene), CVC-ClinicDB (62), CVC-ColonDB (380),
    ETIS-LaribPolypDB (196) and Kvasir (100). MIU-VL (Qin et al., ICLR 2023)
    turned the masks into COCO bounding boxes, one per polyp, and this class
    reads those annotations. Each image is its own patient: the sets do not
    identify patients. Every event has ``subset`` (a test set name or
    ``"train"``), ``path``, ``width``, ``height`` and ``boxes``, a JSON list
    of COCO ``[x, y, width, height]`` boxes in pixels.

    Getting the data:
        - Boxes: MIU-VL's ``POLYP.zip`` (50 MB), linked from
          https://github.com/MembrAI/MIU-VL (Google Drive:
          https://drive.google.com/file/d/10ISx1yXxfE20nKq6UqquUAD5Egk3hyqi).
          It holds ``DATA/POLYP`` with the COCO annotations and the training
          images, but empty test image folders.
        - Test images: PraNet's ``TestDataset.zip`` (327 MB), linked from
          https://github.com/DengPingFan/PraNet (Google Drive:
          https://drive.google.com/file/d/1Y2z7FD5p5y31vkZwQQomXFRB0HutHyao).
        - Access: open; both are public Google Drive links (``gdown`` can
          fetch them). No account or data-use agreement.
        - Unpack: copy each ``TestDataset/<set>/images`` folder into
          ``DATA/POLYP/val/<set>/images``::

              unzip POLYP.zip && unzip TestDataset.zip
              for s in CVC-300 CVC-ClinicDB CVC-ColonDB ETIS-LaribPolypDB Kvasir; do
                  cp TestDataset/$s/images/* DATA/POLYP/val/$s/images/
              done

        - ``root`` layout (``DATA/POLYP``, the layout MIU-VL's code reads)::

              POLYP/                                  <- root
                  annotations/CVC-300_val.json ... Kvasir_val.json
                  annotations/Polyp_train.json
                  train/images/*.png
                  val/CVC-300/images/*.png ...

          ``root`` is only read. The metadata CSV goes to ``metadata_dir``.

    Licence:
        The source datasets are released for research use by their authors
        (Kvasir-SEG: CC BY 4.0; CVC-ClinicDB, CVC-ColonDB, ETIS-Larib and
        EndoScene: research use, cite the papers). Check each source before
        other uses.

    Citation:
        Fan, D.-P. et al. PraNet: Parallel Reverse Attention Network for
        Polyp Segmentation. MICCAI 2020. Qin, Z. et al. Medical Image
        Understanding with Pretrained Vision Language Models: A
        Comprehensive Study. ICLR 2023. And the source datasets: Jha et al.
        (Kvasir-SEG, MMM 2020), Bernal et al. (CVC-ClinicDB, CMIG 2015;
        CVC-ColonDB, Pattern Recognition 2012), Silva et al. (ETIS-Larib,
        IJCARS 2014), Vázquez et al. (EndoScene, J. Healthc. Eng. 2017).

    Args:
        root: The ``DATA/POLYP`` folder.
        subsets: Which sets to index: names from :data:`POLYP_TEST_SETS`
            and ``"train"``. Defaults to the five test sets.
        dataset_name: Name of the dataset. Defaults to ``"polyp"``.
        config_path: Table configuration YAML. Defaults to the bundled one.
        cache_dir: Directory for processed caches.
        metadata_dir: Directory for the metadata CSV. Defaults to
            ``<cache_dir or the PyHealth cache>/metadata/<dataset_name>``.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Raises:
        ValueError: On an unknown subset.
        FileNotFoundError: If an annotation file or an image is missing.

    Examples:
        >>> from pyhealth.datasets import PolypDataset
        >>> dataset = PolypDataset(root="/data/DATA/POLYP")  # doctest: +SKIP
        >>> samples = dataset.set_task()  # doctest: +SKIP
    """

    default_dataset_name = "polyp"
    default_table = "polyp"

    def __init__(
        self,
        root: str,
        subsets: Sequence[str] = POLYP_TEST_SETS,
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | None = None,
        metadata_dir: str | None = None,
        num_workers: int = 1,
        dev: bool = False,
        refresh_metadata: bool = False,
    ) -> None:
        if isinstance(subsets, str):
            subsets = (subsets,)
        known = (*POLYP_TEST_SETS, "train")
        unknown = [s for s in subsets if s not in known]
        if unknown or not subsets:
            raise ValueError(f"subsets must be a non-empty subset of {known}, got {list(subsets)}.")
        self.subsets = tuple(s for s in known if s in subsets)
        self.source_root = Path(root).expanduser().resolve()
        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_FILE
        source_key = f"root={self.source_root}|subsets={','.join(self.subsets)}"
        marker = self.metadata_dir / "polyp-source.txt"
        with FileLock(str(self.metadata_path) + ".lock"):
            current = marker.read_text().strip() if marker.exists() else None
            if refresh_metadata or not self.metadata_path.exists() or current != source_key:
                self.prepare_metadata()
                marker.write_text(source_key)
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "polyp.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def prepare_metadata(self) -> None:
        """Write the metadata CSV, one row per image, to ``metadata_dir``.

        Raises:
            FileNotFoundError: If an annotation file or an image is missing.
        """
        rows = []
        for subset in self.subsets:
            if subset == "train":
                ann_path = self.source_root / "annotations" / "Polyp_train.json"
                image_dir = self.source_root / "train" / "images"
            else:
                ann_path = self.source_root / "annotations" / f"{subset}_val.json"
                image_dir = self.source_root / "val" / subset / "images"
            if not ann_path.exists():
                raise FileNotFoundError(f"{ann_path} is missing; see the PolypDataset docs.")
            with open(ann_path) as f:
                ann = json.load(f)
            boxes: dict[int, list[list[float]]] = {img["id"]: [] for img in ann["images"]}
            for a in ann["annotations"]:
                boxes[a["image_id"]].append([float(v) for v in a["bbox"]])
            for img in sorted(ann["images"], key=lambda i: i["id"]):
                path = image_dir / img["file_name"]
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} is missing. The test images come from PraNet's "
                        "TestDataset.zip; see the PolypDataset docs."
                    )
                rows.append({
                    "patient_id": f"{subset}/{Path(img['file_name']).stem}",
                    "subset": subset,
                    "path": str(path),
                    "width": int(img["width"]),
                    "height": int(img["height"]),
                    "boxes": json.dumps(boxes[img["id"]]),
                })
        pd.DataFrame(rows).to_csv(self.metadata_path, index=False)
        logger.info("Wrote %d polyp images to %s", len(rows), self.metadata_path)

    @property
    def default_task(self):
        """The default task, :class:`~pyhealth.tasks.PolypDetection`."""
        from ..tasks import PolypDetection

        return PolypDetection()
