"""LNDb lung nodule CT dataset.

The loader indexes the merged radiologist findings of the LNDb training set
(``trainNodules_gt.csv``) into one event per finding, with the official
cross-validation fold of its CT, and ships a small MetaImage (``.mhd`` /
``.raw``) reader so the CTs load without SimpleITK.
"""

from __future__ import annotations

import hashlib
import logging
import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .imaging.io import _resolve_metadata_dir

logger = logging.getLogger(__name__)

_GT_CSV = "trainNodules_gt.csv"
_FOLDS_CSV = "trainFolds.csv"
_METADATA_CSV = "lndb-metadata-pyhealth.csv"
_MARKER = "lndb-metadata-source.txt"

#: Volume (mm^3) that LNDb stores for nodules < 3 mm, which radiologists
#: marked by their centre only: a sphere of 3 mm diameter.
SMALL_NODULE_VOLUME = 4.0 / 3.0 * math.pi * 1.5**3

_MET_DTYPES = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_LONG": np.int64,
    "MET_ULONG": np.uint64,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}


@dataclass
class MetaImage:
    """A CT volume with the geometry needed to map world to voxel coordinates.

    Attributes:
        array: Voxel array in ``(z, y, x)`` order.
        origin: World position of voxel ``(0, 0, 0)`` in ``(x, y, z)`` mm.
        spacing: Voxel size in ``(x, y, z)`` mm.
        direction: ``3 x 3`` direction cosine matrix (columns are the image
            axes), as returned by SimpleITK's ``GetDirection``.

    Examples:
        >>> import numpy as np
        >>> image = MetaImage(
        ...     array=np.zeros((4, 5, 6), dtype=np.int16),
        ...     origin=np.array([-10.0, -10.0, 0.0]),
        ...     spacing=np.array([0.5, 0.5, 2.0]),
        ...     direction=np.eye(3),
        ... )
        >>> image.world_to_voxel([-9.0, -9.5, 4.0]).tolist()
        [2, 1, 2]
    """

    array: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray

    def world_to_voxel(self, xyz) -> np.ndarray:
        """Return the nearest voxel index ``(z, y, x)`` of a world point.

        Args:
            xyz: World coordinate ``(x, y, z)`` in mm, as in the LNDb CSVs.

        Returns:
            Integer array ``(z, y, x)``. It may lie outside the volume.

        Examples:
            >>> import numpy as np
            >>> image = MetaImage(np.zeros((2, 2, 2)), np.zeros(3),
            ...                   np.ones(3), np.eye(3))
            >>> image.world_to_voxel([1.0, 0.0, 0.0]).tolist()
            [0, 0, 1]
        """
        offset = np.asarray(xyz, dtype=float) - self.origin
        index_xyz = np.linalg.solve(self.direction * self.spacing, offset)
        return np.rint(index_xyz[::-1]).astype(int)


def _parse_mhd_header(path: Path) -> dict[str, str]:
    """Parse ``key = value`` lines of a MetaImage header."""
    header: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def _read_mhd_numpy(path: Path) -> MetaImage:
    """Read a 3D MetaImage with numpy only."""
    header = _parse_mhd_header(path)
    dims = [int(v) for v in header["DimSize"].split()]
    if len(dims) != 3 or int(header.get("NDims", 3)) != 3:
        raise ValueError(f"Only 3D MetaImage volumes are supported: {path}")
    element_type = header.get("ElementType", "")
    if element_type not in _MET_DTYPES:
        raise ValueError(f"Unsupported ElementType {element_type!r}: {path}")
    if int(header.get("ElementNumberOfChannels", 1)) != 1:
        raise ValueError(f"Multi-channel MetaImage is not supported: {path}")
    data_file = header.get("ElementDataFile", "")
    if data_file in ("", "LOCAL", "LIST") or "%" in data_file:
        raise ValueError(
            f"ElementDataFile={data_file!r} is not supported by the numpy "
            f"reader; install SimpleITK to read {path}."
        )
    msb = header.get(
        "BinaryDataByteOrderMSB", header.get("ElementByteOrderMSB", "False")
    )
    dtype = np.dtype(_MET_DTYPES[element_type]).newbyteorder(
        ">" if msb.lower() == "true" else "<"
    )
    raw_path = path.with_name(data_file)
    shape = tuple(reversed(dims))
    if header.get("CompressedData", "False").lower() == "true":
        buffer = zlib.decompress(raw_path.read_bytes())
        array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    else:
        array = np.memmap(raw_path, dtype=dtype, mode="r", shape=shape)
    origin = header.get("Offset", header.get("Origin", "0 0 0"))
    matrix = header.get("TransformMatrix", "1 0 0 0 1 0 0 0 1")
    return MetaImage(
        array=array,
        origin=np.array([float(v) for v in origin.split()]),
        spacing=np.array(
            [float(v) for v in header.get("ElementSpacing", "1 1 1").split()]
        ),
        # MetaIO stores one row per image axis; ITK's direction matrix has
        # one column per axis, hence the transpose.
        direction=np.array([float(v) for v in matrix.split()]).reshape(3, 3).T,
    )


def read_mhd(path: str | Path, backend: str = "auto") -> MetaImage:
    """Read an LNDb CT stored as a MetaImage ``.mhd`` header plus ``.raw`` data.

    Args:
        path: Path to the ``.mhd`` header.
        backend: ``"numpy"`` uses the built-in reader for single-file data,
            uncompressed (as the LNDb CTs are; memory mapped) or zlib
            compressed (``CompressedData = True``). ``"sitk"`` uses SimpleITK, which must be
            installed. ``"auto"`` (default) uses SimpleITK when it is
            importable and numpy otherwise.

    Returns:
        A :class:`MetaImage` whose ``array`` is in ``(z, y, x)`` order.

    Raises:
        ImportError: If ``backend="sitk"`` and SimpleITK is missing.
        ValueError: For an unknown backend or an unsupported header.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> import numpy as np
        >>> folder = Path(tempfile.mkdtemp())
        >>> np.arange(24, dtype=np.int16).tofile(folder / "ct.raw")
        >>> _ = (folder / "ct.mhd").write_text(
        ...     "NDims = 3\\nDimSize = 4 3 2\\nElementType = MET_SHORT\\n"
        ...     "ElementSpacing = 0.7 0.7 1.5\\nOffset = -5 -5 -10\\n"
        ...     "ElementDataFile = ct.raw\\n"
        ... )
        >>> image = read_mhd(folder / "ct.mhd", backend="numpy")
        >>> image.array.shape
        (2, 3, 4)
        >>> int(image.array[1, 0, 0])
        12
    """
    path = Path(path)
    if backend not in ("auto", "numpy", "sitk"):
        raise ValueError("backend must be 'auto', 'numpy' or 'sitk'.")
    if backend in ("auto", "sitk"):
        try:
            import SimpleITK as sitk
        except ImportError as exc:
            if backend == "sitk":
                raise ImportError(
                    "read_mhd(backend='sitk') requires SimpleITK. Install it "
                    "with `pip install SimpleITK`, or use backend='numpy'."
                ) from exc
        else:
            image = sitk.ReadImage(str(path))
            return MetaImage(
                array=sitk.GetArrayViewFromImage(image).copy(),
                origin=np.array(image.GetOrigin(), dtype=float),
                spacing=np.array(image.GetSpacing(), dtype=float),
                direction=np.array(image.GetDirection(), dtype=float).reshape(
                    3, 3
                ),
            )
    return _read_mhd_numpy(path)


def equivalent_diameter(volume_mm3: float) -> float:
    """Diameter (mm) of a sphere with the given volume (mm^3).

    Examples:
        >>> round(equivalent_diameter(SMALL_NODULE_VOLUME), 3)
        3.0
    """
    return 2.0 * (3.0 * float(volume_mm3) / (4.0 * math.pi)) ** (1.0 / 3.0)


def _finding_class(nodule: int, volume: float) -> int:
    """Map a merged LNDb finding to 0 non-nodule, 1 nodule <3 mm, 2 >=3 mm."""
    if int(nodule) == 0:
        return 0
    return 1 if equivalent_diameter(volume) <= 3.0 + 1e-3 else 2


class LNDbDataset(BaseDataset):
    """LNDb lung nodule database: chest CTs with radiologist findings.

    LNDb holds 294 chest CTs from the Centro Hospitalar e Universitario de
    Sao Joao (Porto, Portugal), 236 of them in the public training set. Each
    training CT was read by one to three radiologists, who marked every
    finding as a nodule (segmented when >= 3 mm, centre only when < 3 mm) or
    a non-nodule. This class indexes the merged findings of
    ``trainNodules_gt.csv``, one event per finding, in a ``findings`` table:

    - ``patient_id``: ``LNDb-XXXX``. Each patient has one CT, so splitting
      by patient is splitting by scan.
    - ``image_path``: the ``LNDb-XXXX.mhd`` header, or empty when the CT is
      not on disk (tasks skip those findings).
    - ``x``, ``y``, ``z``: world coordinates in mm.
    - ``nodule``, ``volume``, ``eq_diameter``, ``texture``, ``agreement``
      (number of radiologists who marked it), ``finding_id``.
    - ``finding_class``: 0 non-nodule, 1 nodule < 3 mm, 2 nodule >= 3 mm.
      Nodules < 3 mm are stored with the volume of a 3 mm sphere
      (:data:`SMALL_NODULE_VOLUME`, 14.137 mm^3), so a nodule is class 1 when
      its equivalent diameter is at most 3 mm.
    - ``fold``: the official cross-validation fold (0-3) from
      ``trainFolds.csv``, or -1 when that file is absent.

    The test set's labels are not public, so only the training set is
    indexed.

    Getting the data:
        - Source: Zenodo record https://doi.org/10.5281/zenodo.8348419
          (https://zenodo.org/records/8348419); challenge page
          https://lndb.grand-challenge.org/.
        - Access: open, no account, registration or agreement needed.
        - Files (training set): ``trainset_csv.zip`` (78 KB: holds
          ``trainNodules_gt.csv``, ``trainNodules.csv``, ``trainFolds.csv``,
          ``trainCTs.csv`` and ``trainFleischner.csv``) and the CTs in
          ``data0.rar`` to ``data5.rar`` (3.8-4.2 GB each, 24.3 GB in all,
          about 40 CTs per archive). ``masks.rar`` (22 MB, per-radiologist
          nodule masks) and the ``test*`` files are not used.
        - Download, e.g.::

              curl -L -o trainset_csv.zip "https://zenodo.org/records/8348419/files/trainset_csv.zip?download=1"
              for i in 0 1 2 3 4 5; do
                  curl -L -o data$i.rar "https://zenodo.org/records/8348419/files/data$i.rar?download=1"
              done

          One archive is enough to try the tasks: findings whose CT is not
          on disk are skipped.
        - Unpack: ``unzip trainset_csv.zip`` and ``unrar x dataN.rar`` (or
          ``bsdtar -xf dataN.rar``). Python cannot read RAR, so there is no
          ``download=True``.
        - ``root`` layout (both the CSVs and the CTs are searched for
          recursively, so the extracted folder names do not matter)::

              lndb/                          # root
                  trainNodules_gt.csv        # required
                  trainFolds.csv             # optional: official folds
                  ...
                  data0/
                      LNDb-0001.mhd
                      LNDb-0001.raw          # uncompressed int16 CT, ~170 MB
                      ...

        - Licence: ambiguous. The Zenodo record's licence field says CC BY
          4.0, but its description says the data are released under CC
          BY-NC-ND 4.0 (non-commercial, no derivatives). Follow the stricter
          NC-ND terms unless the LNDb authors tell you otherwise. Cite
          Pedrosa et al. (2019) and the challenge paper (Pedrosa et al.,
          2021).

    The metadata CSV is written to ``metadata_dir`` (by default
    ``<cache_dir>/metadata/lndb``), never into ``root``. It is rebuilt when
    ``root`` or the set of CTs on disk changes.

    Citation:
        Pedrosa, J., Aresta, G., Ferreira, C., et al. LNDb: A Lung Nodule
        Database on Computed Tomography. arXiv:1911.08434, 2019.
        Pedrosa, J., Aresta, G., Ferreira, C., et al. LNDb challenge on
        automatic lung cancer patient management. Medical Image Analysis
        70, 102027, 2021.

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).
    Deviations from CLIMB: CLIMB's documentation describes the same three
    classes as ``finding_class``, but its code instead builds two-way
    multilabel ``[small, big]`` targets over sliding windows, and its QA
    split only asks about nodules >= 3 mm. This loader keeps one finding per
    sample, so findings are never split across windows, and
    :class:`~pyhealth.tasks.LNDbFindingClassification` offers the three
    documented classes or the binary nodule >= 3 mm question. CLIMB splits on
    the first 75% of scan ids; here the official four folds are exposed as
    ``fold`` instead.

    Args:
        root: Folder holding the LNDb CSVs and CTs (searched recursively).
        dataset_name: Optional dataset name. Defaults to ``"lndb"``.
        config_path: Optional table configuration path.
        cache_dir: Directory for PyHealth caches.
        metadata_dir: Explicit directory for the metadata CSV.
        num_workers: Number of workers for dataset processing.
        dev: Whether to limit the dataset to 1000 patients.
        refresh_metadata: Rebuild the metadata CSV even if it is current.

    Examples:
        >>> from pyhealth.datasets import LNDbDataset
        >>> from pyhealth.tasks import LNDbFindingClassification
        >>> dataset = LNDbDataset(root="/data/lndb")  # doctest: +SKIP
        >>> dataset.stats()  # doctest: +SKIP
        >>> samples = dataset.set_task(LNDbFindingClassification())  # doctest: +SKIP
    """

    default_dataset_name = "lndb"
    default_table = "findings"

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
        if not self.source_root.is_dir():
            raise FileNotFoundError(
                f"LNDb root not found: {self.source_root}. Download "
                "trainset_csv.zip and data*.rar from "
                "https://doi.org/10.5281/zenodo.8348419."
            )
        self.gt_path = self._find(_GT_CSV, required=True)
        self.folds_path = self._find(_FOLDS_CSV, required=False)
        self.scan_paths = {
            path.stem: path for path in sorted(self.source_root.rglob("LNDb-*.mhd"))
        }

        name = dataset_name or self.default_dataset_name
        self.metadata_dir = (
            Path(metadata_dir).expanduser()
            if metadata_dir is not None
            else _resolve_metadata_dir(cache_dir, name)
        )
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.metadata_dir / _METADATA_CSV
        marker = self.metadata_dir / _MARKER
        source_key = self._source_key()
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
            config_path = str(Path(__file__).parent / "configs" / "lndb.yaml")
        super().__init__(
            root=str(self.metadata_dir),
            tables=[self.default_table],
            dataset_name=name,
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def _find(self, filename: str, required: bool) -> Path | None:
        """Return the first ``filename`` under ``root``, searched recursively."""
        matches = sorted(self.source_root.rglob(filename))
        if matches:
            return matches[0]
        if required:
            raise FileNotFoundError(
                f"{filename} not found under {self.source_root}. Extract "
                "trainset_csv.zip from https://doi.org/10.5281/zenodo.8348419 "
                "into root."
            )
        return None

    def _source_key(self) -> str:
        """Hash of the inputs the metadata CSV depends on."""
        parts = [str(self.source_root), str(self.gt_path), str(self.folds_path)]
        for path in (self.gt_path, self.folds_path):
            if path is not None:
                stat = path.stat()
                parts.append(f"{stat.st_size}:{stat.st_mtime_ns}")
        parts.extend(str(path) for path in self.scan_paths.values())
        return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

    def _load_folds(self) -> dict[int, int]:
        """Map LNDbID to its official fold (columns ``Fold0`` .. ``Fold3``)."""
        if self.folds_path is None:
            logger.warning(
                "%s not found under %s; fold is -1 for every finding.",
                _FOLDS_CSV,
                self.source_root,
            )
            return {}
        folds: dict[int, int] = {}
        frame = pd.read_csv(self.folds_path)
        for fold, column in enumerate(frame.columns):
            for value in frame[column].dropna():
                folds[int(value)] = fold
        return folds

    def prepare_metadata(self) -> None:
        """Write one row per merged finding to the metadata CSV.

        Examples:
            >>> dataset = LNDbDataset(root="/data/lndb")  # doctest: +SKIP
            >>> dataset.prepare_metadata()  # doctest: +SKIP
        """
        folds = self._load_folds()
        frame = pd.read_csv(self.gt_path)
        rows = []
        for row in frame.itertuples(index=False):
            lndb_id = int(row.LNDbID)
            scan_name = f"LNDb-{lndb_id:04d}"
            scan = self.scan_paths.get(scan_name)
            volume = float(row.Volume)
            texture = getattr(row, "Text", float("nan"))
            rows.append(
                {
                    "patient_id": scan_name,
                    "record_id": f"{scan_name}_{int(row.FindingID)}",
                    "image_path": "" if scan is None else str(scan),
                    "finding_id": int(row.FindingID),
                    "x": float(row.x),
                    "y": float(row.y),
                    "z": float(row.z),
                    "agreement": int(row.AgrLevel),
                    "nodule": int(row.Nodule),
                    "volume": volume,
                    "eq_diameter": equivalent_diameter(volume),
                    "finding_class": _finding_class(row.Nodule, volume),
                    "texture": "" if pd.isna(texture) else float(texture),
                    "fold": folds.get(lndb_id, -1),
                }
            )
        columns = [
            "patient_id", "record_id", "image_path", "finding_id", "x", "y",
            "z", "agreement", "nodule", "volume", "eq_diameter",
            "finding_class", "texture", "fold",
        ]
        pd.DataFrame(rows, columns=columns).to_csv(self.metadata_path, index=False)
        on_disk = sum(1 for row in rows if row["image_path"])
        logger.info(
            "Wrote %d LNDb findings (%d with a CT on disk, %d CTs found) to %s",
            len(rows),
            on_disk,
            len(self.scan_paths),
            self.metadata_path,
        )

    @property
    def default_task(self):
        """Return the default three-class finding classification task.

        Examples:
            >>> LNDbDataset(root="/data/lndb").default_task  # doctest: +SKIP
        """
        from ..tasks.lndb_classification import LNDbFindingClassification

        return LNDbFindingClassification()
