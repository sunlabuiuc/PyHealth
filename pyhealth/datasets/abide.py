"""
PyHealth dataset for ABIDE I resting-state fMRI (Preprocessed Connectomes Project).

Dataset link:
    http://preprocessed-connectomes-project.org/abide/

Dataset papers: (please cite if you use this dataset)
    A. Di Martino, C.-G. Yan, Q. Li, et al. "The autism brain imaging data
    exchange: towards a large-scale evaluation of the intrinsic brain
    architecture in autism." Molecular Psychiatry 19, 659-667 (2014).
    https://doi.org/10.1038/mp.2013.78

    C. Craddock, Y. Benhajali, C. Chu, et al. "The Neuro Bureau Preprocessing
    Initiative: open sharing of preprocessed neuroimaging data and
    derivatives." Frontiers in Neuroinformatics (2013).
    https://doi.org/10.3389/conf.fninf.2013.09.00041

    H. Cui, W. Dai, Y. Zhu, et al. "BrainGB: A Benchmark for Brain Network
    Analysis with Graph Neural Networks." IEEE Transactions on Medical
    Imaging 42(2), 493-506 (2023). https://doi.org/10.1109/TMI.2022.3218745

Description:
    Implements ``ABIDEDataset``: one metadata row per scan, linking the ABIDE
    phenotypic file to the PCP ROI time-series file of a chosen pipeline,
    strategy and atlas. ``download=True`` fetches them from the open
    ``fcp-indi`` S3 bucket. Connectivity matrices are computed by the task.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from filelock import FileLock

from .base_dataset import BaseDataset
from .configs.config import load_yaml_config
from .ptbxl import _write_csv_atomic, _write_yaml_atomic, root_cache_key
from .utils import MODULE_CACHE_PATH

if TYPE_CHECKING:
    from pyhealth.tasks.abide_classification import ABIDEAutismClassification

logger = logging.getLogger(__name__)

ABIDE_S3_URL = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
ABIDE_PHENOTYPIC_CSV = "Phenotypic_V1_0b_preprocessed1.csv"

ABIDE_PIPELINES = ("cpac", "ccs", "dparsf", "niak")
ABIDE_STRATEGIES = ("filt_noglobal", "filt_global", "nofilt_noglobal", "nofilt_global")
ABIDE_ATLASES = ("cc200", "cc400", "aal", "ho", "ez", "tt", "dosenbach160")

METADATA_COLUMNS = (
    "patient_id",
    "record_id",
    "site",
    "dx_group",
    "age",
    "sex",
    "roi_file",
)

_DEFAULT_METADATA_CACHE = Path(MODULE_CACHE_PATH) / "abide"


def abide_roi_relpath(file_id: str, pipeline: str, strategy: str, atlas: str) -> str:
    """Return the S3-relative path of one scan's ROI time-series file.

    Args:
        file_id (str): ``FILE_ID`` from the phenotypic file, e.g.
            ``"Pitt_0050003"``.
        pipeline (str): PCP pipeline, e.g. ``"cpac"``.
        strategy (str): Noise strategy, e.g. ``"filt_noglobal"``.
        atlas (str): Atlas, e.g. ``"cc200"``.

    Returns:
        str: The path below ``ABIDE_Initiative/``; the same layout is used
        under ``root`` when downloading.

    Examples:
        >>> abide_roi_relpath("Pitt_0050003", "cpac", "filt_noglobal", "cc200")
        'Outputs/cpac/filt_noglobal/rois_cc200/Pitt_0050003_rois_cc200.1D'
    """
    return (
        f"Outputs/{pipeline}/{strategy}/rois_{atlas}/{file_id}_rois_{atlas}.1D"
    )


class ABIDEDataset(BaseDataset):
    """ABIDE I resting-state fMRI ROI time series, preprocessed by the PCP.

    The Autism Brain Imaging Data Exchange I pools resting-state fMRI from 17
    international sites (20 site codes): 1,112 subjects, 1,035 of them with
    PCP-preprocessed functional data (505 autism, 530 typical controls).
    Each scan becomes one event of table ``scans`` holding its site,
    diagnosis, age, sex and the path to its ROI time-series file
    (``<FILE_ID>_rois_<atlas>.1D``, time x ROI). The task builds the
    connectivity matrix from it, following the BrainGB recipe (Pearson
    correlation of the Craddock-200 ROI series from C-PAC, band-pass
    filtered, no global signal regression).

    Getting the data:
        - Source: ABIDE Preprocessed, Preprocessed Connectomes Project,
          http://preprocessed-connectomes-project.org/abide/ (download
          guide: http://preprocessed-connectomes-project.org/abide/download.html).
          The files sit in the public S3 bucket
          ``s3://fcp-indi/data/Projects/ABIDE_Initiative/``, also served over
          HTTPS at ``https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/``.
        - Access: open. The PCP derivatives on S3 need no account, sign-up
          or agreement (the raw ABIDE images on NITRC ask for a NITRC
          registration, but they are not used here).
        - Files: the phenotypic summary
          ``Phenotypic_V1_0b_preprocessed1.csv`` (about 450 KB) and one ROI
          time-series file per scan,
          ``Outputs/<pipeline>/<strategy>/rois_<atlas>/<FILE_ID>_rois_<atlas>.1D``
          (about 0.1-1 MB each; the C-PAC ``filt_noglobal`` ``rois_cc200``
          folder holds 1,102 files, about 430 MB).
        - Download, option 1: pass ``download=True``. It fetches the
          phenotypic file and the ROI files of the chosen ``pipeline`` /
          ``strategy`` / ``atlas`` into ``root`` in the S3 layout
          (``max_subjects`` limits it to the first scans).
        - Download, option 2, with the AWS CLI (no credentials needed)::

              aws s3 cp --no-sign-request s3://fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv <root>/
              aws s3 sync --no-sign-request s3://fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/ <root>/Outputs/cpac/filt_noglobal/rois_cc200/

          or ``wget`` each file from the HTTPS URL. Nothing needs unpacking.
        - ``root`` layout::

              <root>/
                  Phenotypic_V1_0b_preprocessed1.csv
                  Outputs/
                      cpac/                     # pipeline
                          filt_noglobal/        # strategy
                              rois_cc200/       # rois_<atlas>
                                  Pitt_0050003_rois_cc200.1D
                                  ...

          The file at the chosen selection's path is always used first.
          ROI files kept elsewhere under ``root`` (e.g. a flat
          ``rois_cc200/`` folder) are used only for scans that lack it, and
          files under ``Outputs/<other pipeline>/<other strategy>/`` are
          ignored, so several selections can share one ``root``.

    Licence: the ABIDE data are shared under Creative Commons
    Attribution-NonCommercial-ShareAlike 3.0 (CC BY-NC-SA 3.0): non-commercial
    research use only, with attribution, and derivatives must use the same
    licence. INDI also asks that the analysed datasets are named and their
    funding acknowledged. Cite Di Martino et al. (2014), Craddock et al.
    (2013) and, for the connectivity recipe, BrainGB (Cui et al., 2023).

    Missing values: scans without ``AGE_AT_SCAN`` get ``age = -1.0``; filter
    or impute it before using age as a feature.

    Benchmarked in CLIMB (Dai et al., 2025, https://github.com/DDVD233/CLIMB).

    Deviations from CLIMB:
        - CLIMB loads BrainGB's ``ABIDE.npy`` (1,009 subjects), which needs a
          request to BrainGB and is a pickle (``allow_pickle=True``, unsafe
          to load). Here the matrices are rebuilt from the open PCP outputs,
          so the cohort is every scan with a ROI file (1,035 for C-PAC
          cc200) unless you filter it; the exact BrainGB subject list is not
          published.
        - CLIMB's ``train`` flag is ignored by its loader, so its train and
          validation sets are both the full 1,009 subjects. There is no
          official split; use :class:`~pyhealth.datasets.PatientSplit` with a
          seed (one scan per subject, so it does not leak). ``site`` is kept
          for site-stratified or leave-site-out evaluation.

    Args:
        root (str): Folder holding (or receiving) the phenotypic file and ROI
            files.
        dataset_name (str | None): Optional name prefix; the selection and a
            content hash of the metadata are appended.
        config_path (str | Path | None): Optional YAML config; defaults to
            ``configs/abide.yaml``.
        pipeline (str): PCP pipeline. Default ``"cpac"``.
        strategy (str): Noise strategy. Default ``"filt_noglobal"``.
        atlas (str): ROI atlas. Default ``"cc200"``.
        download (bool): Download the phenotypic file and missing ROI files
            from S3. Default ``False``.
        max_subjects (int | None): With ``download=True``, fetch only the
            first ``max_subjects`` scans (phenotypic-file order), for a quick
            try. Default ``None`` (all).
        metadata_cache_dir (str | Path | None): Where the derived metadata
            CSV and YAML go. Defaults to ``~/.cache/pyhealth/datasets/abide``.
        **kwargs: Forwarded to :class:`BaseDataset`.

    Raises:
        FileNotFoundError: If the phenotypic file is missing (and
            ``download`` is ``False``) or no ROI file is found.
        ValueError: If ``pipeline``, ``strategy`` or ``atlas`` is unknown,
            or a scan has several candidate ROI files outside the S3 layout.

    Examples:
        >>> from pyhealth.datasets import ABIDEDataset
        >>> ds = ABIDEDataset(root="/data/abide", download=True, max_subjects=20)  # doctest: +SKIP
        >>> ds.stats()  # doctest: +SKIP
        >>> samples = ds.set_task()  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | Path | None = None,
        pipeline: str = "cpac",
        strategy: str = "filt_noglobal",
        atlas: str = "cc200",
        download: bool = False,
        max_subjects: int | None = None,
        metadata_cache_dir: str | Path | None = None,
        **kwargs,
    ) -> None:
        for name, value, allowed in (
            ("pipeline", pipeline, ABIDE_PIPELINES),
            ("strategy", strategy, ABIDE_STRATEGIES),
            ("atlas", atlas, ABIDE_ATLASES),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
        self.data_root = Path(root)
        self.pipeline = pipeline
        self.strategy = strategy
        self.atlas = atlas
        self.download = download
        self.max_subjects = max_subjects
        self.metadata_cache_dir = Path(
            metadata_cache_dir if metadata_cache_dir is not None else _DEFAULT_METADATA_CACHE
        )
        selection = f"{pipeline}-{strategy}-{atlas}"
        self.metadata_file_name = (
            f"abide-pyhealth-{selection}-{root_cache_key(self.data_root)}.csv"
        )
        package_config = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).parent / "configs" / "abide.yaml"
        )
        if download:
            self._download()
        self.prepare_metadata()
        resolved_config = self._write_resolved_config(package_config)
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        content_key = hashlib.sha1(csv_path.read_bytes()).hexdigest()[:10]
        super().__init__(
            root=str(self.data_root),
            tables=["scans"],
            dataset_name=f"{dataset_name or 'abide'}_{selection}_{content_key}",
            config_path=str(resolved_config),
            **kwargs,
        )

    @property
    def phenotypic_path(self) -> Path:
        """Path to ``Phenotypic_V1_0b_preprocessed1.csv`` under ``root``."""
        return self.data_root / ABIDE_PHENOTYPIC_CSV

    def _read_phenotypic(self) -> pd.DataFrame:
        if not self.phenotypic_path.is_file():
            raise FileNotFoundError(
                f"Missing {self.phenotypic_path}. Download it from "
                f"{ABIDE_S3_URL}/{ABIDE_PHENOTYPIC_CSV} or pass download=True."
            )
        df = pd.read_csv(self.phenotypic_path)
        df = df[df["FILE_ID"].astype(str) != "no_filename"]
        return df.reset_index(drop=True)

    def _download(self) -> None:
        """Fetch the phenotypic file and the selected ROI files from S3."""
        from pyhealth.utils import download_file

        self.data_root.mkdir(parents=True, exist_ok=True)
        if not self.phenotypic_path.is_file():
            download_file(f"{ABIDE_S3_URL}/{ABIDE_PHENOTYPIC_CSV}", str(self.phenotypic_path))
        file_ids = self._read_phenotypic()["FILE_ID"].astype(str).tolist()
        if self.max_subjects is not None:
            file_ids = file_ids[: int(self.max_subjects)]
        fetched = 0
        for file_id in file_ids:
            rel = abide_roi_relpath(file_id, self.pipeline, self.strategy, self.atlas)
            dest = self.data_root / rel
            if dest.is_file():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                download_file(f"{ABIDE_S3_URL}/{rel}", str(dest))
                fetched += 1
            except OSError as exc:  # urllib errors subclass OSError
                logger.warning("Could not download %s: %s", rel, exc)
        logger.info("Downloaded %d ABIDE ROI files into %s", fetched, self.data_root)

    def load_table(self, table_name: str):
        """Read the derived metadata CSV from ``metadata_cache_dir``.

        Args:
            table_name (str): Table name from the YAML config.

        Returns:
            The dataframe produced by :meth:`BaseDataset.load_table`.
        """
        original_root = self.root
        self.root = str(self.metadata_cache_dir)
        try:
            return super().load_table(table_name)
        finally:
            self.root = original_root

    def _write_resolved_config(self, package_config: Path) -> Path:
        config = load_yaml_config(str(package_config))
        config.tables["scans"].file_path = self.metadata_file_name
        out_path = self.metadata_cache_dir / self.metadata_file_name.replace(
            ".csv", ".yaml"
        )
        with FileLock(str(out_path) + ".lock"):
            _write_yaml_atomic(config.model_dump(), out_path)
        return out_path

    def _find_flat_roi_files(self, suffix: str) -> dict[str, list[Path]]:
        """Map ``FILE_ID`` to ``*<suffix>`` files outside the S3 layout.

        Files under ``Outputs/<pipeline>/<strategy>/`` belong to a known
        selection, so they are left out: only the exact path of the chosen
        selection is used from that layout.
        """
        found: dict[str, list[Path]] = {}
        for p in sorted(self.data_root.rglob(f"*{suffix}")):
            parts = p.relative_to(self.data_root).parts
            if (
                len(parts) >= 4
                and parts[0] == "Outputs"
                and parts[1] in ABIDE_PIPELINES
                and parts[2] in ABIDE_STRATEGIES
            ):
                continue
            found.setdefault(p.name[: -len(suffix)], []).append(p)
        return found

    def prepare_metadata(self) -> None:
        """Join the phenotypic file with the ROI files found under ``root``.

        Each scan's file is looked up first at the S3 path of the chosen
        selection, ``Outputs/<pipeline>/<strategy>/rois_<atlas>/``. Only if
        that file is absent is a ``<FILE_ID>_rois_<atlas>.1D`` file
        elsewhere under ``root`` (outside ``Outputs/<pipeline>/<strategy>/``)
        used, and several such files for one scan raise an error. ROI files
        of another pipeline or strategy are never picked up. Scans without a
        ROI file are skipped (and counted in a log message). A missing
        ``AGE_AT_SCAN`` is stored as ``-1.0``.

        Raises:
            ValueError: If a scan has no file at the selected S3 path and
                several candidate files elsewhere under ``root``.

        Examples:
            >>> ds = ABIDEDataset(root="/data/abide")  # doctest: +SKIP
            >>> ds.prepare_metadata()  # doctest: +SKIP
        """
        pheno = self._read_phenotypic()
        suffix = f"_rois_{self.atlas}.1D"
        others = self._find_flat_roi_files(suffix)
        rows = []
        for rec in pheno.to_dict(orient="records"):
            file_id = str(rec["FILE_ID"])
            path = self.data_root / abide_roi_relpath(
                file_id, self.pipeline, self.strategy, self.atlas
            )
            if not path.is_file():
                candidates = others.get(file_id, [])
                if not candidates:
                    continue
                if len(candidates) > 1:
                    raise ValueError(
                        f"Several {file_id}{suffix} files under {self.data_root} "
                        f"and none at the {self.pipeline}/{self.strategy} S3 "
                        f"path: {[str(c) for c in candidates]}. Keep one, or "
                        "use the Outputs/<pipeline>/<strategy>/ layout."
                    )
                path = candidates[0]
            age = rec.get("AGE_AT_SCAN")
            rows.append(
                {
                    "patient_id": str(int(rec["SUB_ID"])),
                    "record_id": file_id,
                    "site": str(rec["SITE_ID"]),
                    "dx_group": int(rec["DX_GROUP"]),
                    "age": float(age) if pd.notna(age) else -1.0,
                    "sex": {1: "M", 2: "F"}.get(int(rec["SEX"]), "Unknown"),
                    "roi_file": str(path.resolve()),
                }
            )
        if not rows:
            raise FileNotFoundError(
                f"No *{suffix} files under {self.data_root} match the phenotypic "
                "file. Pass download=True or check pipeline/strategy/atlas."
            )
        missing = len(pheno) - len(rows)
        if missing:
            logger.info(
                "%d of %d ABIDE scans have no %s file under root; skipped",
                missing,
                len(pheno),
                suffix,
            )
        df = pd.DataFrame(rows, columns=list(METADATA_COLUMNS))
        csv_path = self.metadata_cache_dir / self.metadata_file_name
        self.metadata_cache_dir.mkdir(parents=True, exist_ok=True)
        with FileLock(str(csv_path) + ".lock"):
            _write_csv_atomic(df, csv_path)
        logger.info("Wrote %d ABIDE scans to %s", len(df), csv_path)

    @property
    def default_task(self) -> ABIDEAutismClassification:
        """The binary autism vs. control connectivity task.

        Returns:
            ABIDEAutismClassification: Default task instance.
        """
        from pyhealth.tasks.abide_classification import ABIDEAutismClassification

        return ABIDEAutismClassification()
