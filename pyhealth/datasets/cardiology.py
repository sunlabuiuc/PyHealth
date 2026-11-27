"""12-lead ECGs of the PhysioNet/CinC Challenge 2020.

Description: Loads the Challenge 2020 training set (PhysioNet
    ``challenge-2020`` 1.0.2), which pools six ECG sources. One metadata row
    is written per recording, from its WFDB header. Windowing and labels live
    in :class:`~pyhealth.tasks.CardiologyDetection`.

References:
    - Perez Alday, E. A. et al. Classification of 12-lead ECGs: the
      PhysioNet/Computing in Cardiology Challenge 2020. Physiological
      Measurement 41(12), 2020. https://doi.org/10.1088/1361-6579/abc960
    - https://physionet.org/content/challenge-2020/1.0.2/
"""

import logging
from pathlib import Path

import pandas as pd

from pyhealth.tasks import CardiologyDetection

from .base_dataset import BaseDataset
from .metadata_csv import (
    metadata_csv_exists,
    resolve_metadata_root,
    write_metadata_csv,
)

logger = logging.getLogger(__name__)

#: Source folders of ``training/``, in the order of ``chosen_dataset``, and
#: the table each one is loaded as.
SOURCES: dict[str, str] = {
    "cpsc_2018": "cpsc_2018",
    "cpsc_2018_extra": "cpsc_2018_extra",
    "georgia": "georgia",
    "ptb": "ptb",
    "ptb-xl": "ptb_xl",
    "st_petersburg_incart": "st_petersburg_incart",
}

_SEX = {"f": "Female", "female": "Female", "m": "Male", "male": "Male"}
_COLUMNS = [
    "patient_id",
    "record_id",
    "source",
    "signal_file",
    "fs",
    "n_samples",
    "n_leads",
    "age",
    "sex",
    "dx",
]


def _csv_name(table: str) -> str:
    return f"cardiology-{table}-pyhealth.csv"


def parse_header(header_path: str | Path) -> dict:
    """Reads one WFDB header of the Challenge 2020 data.

    The first line gives the record name, the number of leads, the sampling
    rate and the number of samples; the ``# Age:``, ``# Sex:`` and ``# Dx:``
    comments give the demographics and the SNOMED CT diagnosis codes.

    Args:
        header_path: Path to a ``.hea`` file.

    Returns:
        A dict with ``record_id``, ``n_leads``, ``fs``, ``n_samples``, ``age``
        (empty when unknown), ``sex`` (``"Female"``, ``"Male"`` or empty; the
        sources write both ``F`` and ``Female``) and ``dx`` (the codes joined
        by ``","``).

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from pyhealth.datasets.cardiology import parse_header
        >>> path = Path(tempfile.mkdtemp()) / "I0001.hea"
        >>> text = "I0001 12 257 462600\\n# Age: 65\\n# Sex: F\\n# Dx: 53741008\\n"
        >>> _ = path.write_text(text)
        >>> header = parse_header(path)
        >>> header["sex"], header["fs"], header["dx"]
        ('Female', 257.0, '53741008')
    """
    lines = Path(header_path).read_text().splitlines()
    fields = lines[0].split()
    comments = {}
    for line in lines[1:]:
        if line.startswith("#") and ":" in line:
            key, value = line[1:].split(":", 1)
            comments[key.strip().lower()] = value.strip()
    age = comments.get("age", "")
    return {
        # Some copies of the Georgia headers write "E00001.mat" here.
        "record_id": Path(fields[0]).stem,
        "n_leads": int(fields[1]),
        # WFDB allows "fs/counter_freq"; only fs is used.
        "fs": float(fields[2].split("/")[0]),
        "n_samples": int(fields[3]),
        "age": "" if age.lower() in ("", "nan", "unknown") else age,
        "sex": _SEX.get(comments.get("sex", "").lower(), ""),
        "dx": ",".join(
            code.strip() for code in comments.get("dx", "").split(",") if code.strip()
        ),
    }


class CardiologyDataset(BaseDataset):
    """12-lead ECGs of the PhysioNet/CinC Challenge 2020 (six sources).

    The training set of the Challenge (PhysioNet ``challenge-2020``, version
    1.0.2) holds 43,101 recordings from six sources. Each source folder is
    loaded as its own table:

    - ``cpsc_2018``: 6,877 recordings, 500 Hz
    - ``cpsc_2018_extra``: 3,453 recordings, 500 Hz
    - ``georgia``: 10,344 recordings, 500 Hz
    - ``ptb``: 516 recordings, 1000 Hz
    - ``ptb-xl`` (table ``ptb_xl``): 21,837 recordings, 500 Hz
    - ``st_petersburg_incart``: 74 recordings, 257 Hz

    Each recording comes with age, sex and one or more SNOMED CT diagnosis
    codes. The Challenge gives no patient identifiers, so every recording is
    its own patient (``patient_id`` is the record name, such as ``A0001``).
    The ``ptb-xl`` source is the Challenge's renamed copy of PTB-XL;
    :class:`~pyhealth.datasets.PTBXLDataset` reads the original PTB-XL
    release, which has patient ids.

    ``prepare_metadata`` writes ``cardiology-<table>-pyhealth.csv`` (one row
    per recording) next to the raw files, or under
    ``~/.cache/pyhealth/cardiology/<hash of root>`` when the dataset directory
    is not writable. Only the headers are read, so this is fast. An existing
    CSV is reused as it is, so recordings added later are not indexed. To
    re-index, delete the CSV (in ``root`` or in the cache directory) and
    construct the dataset again. The CSV location is decided by the first
    selected source's table, and every selected table is read from that
    same directory.

    Changes from the PyHealth 1.x ``CardiologyDataset``: every recording is
    one patient (1.x grouped each folder of up to 1,000 recordings into one
    "patient"), and the windowing moved to
    :class:`~pyhealth.tasks.CardiologyDetection`, which reads each
    recording's sampling rate instead of assuming 500 Hz.

    Getting the data:
        - Source: https://physionet.org/content/challenge-2020/1.0.2/
        - Access: open, no account needed (CC BY 4.0; 4.5 GB zipped).
        - Download: ``wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/``
        - ``root``: the ``training/`` folder of that download, which holds the
          per-source folders (``cpsc_2018``, ``georgia``, ...). Missing
          sources are skipped, with a warning if they were asked for.

    Args:
        root: The ``training/`` folder.
        sources: Source folders to load, for example ``["georgia", "ptb-xl"]``.
            Defaults to every source found under ``root``.
        chosen_dataset: The 1.x way to pick sources: six 0/1 flags in the
            order cpsc_2018, cpsc_2018_extra, georgia, ptb, ptb-xl,
            st_petersburg_incart. Use ``sources`` or this, not both.
        dataset_name: Name stored on the dataset. Defaults to ``"cardiology"``.
        config_path: YAML config. Defaults to ``configs/cardiology.yaml``.
        **kwargs: Forwarded to :class:`~pyhealth.datasets.BaseDataset`
            (``dev``, ``num_workers`` and ``cache_dir``).

    Raises:
        ValueError: If both ``sources`` and ``chosen_dataset`` are given, a
            source is unknown, ``chosen_dataset`` isn't six 0/1 flags, or no
            selected source is found under ``root``.

    Examples:
        >>> from pyhealth.datasets import CardiologyDataset
        >>> from pyhealth.tasks import CardiologyDetection
        >>> dataset = CardiologyDataset(
        ...     root="/srv/local/data/physionet.org/files/challenge-2020/1.0.2/training",
        ...     sources=["cpsc_2018", "georgia"],
        ... )
        >>> dataset.stats()
        >>> samples = dataset.set_task(CardiologyDetection("AR"))
        >>> samples[0]["signal"].shape
        torch.Size([12, 5000])
    """

    def __init__(
        self,
        root: str,
        sources: list[str] | None = None,
        chosen_dataset: list[int] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "cardiology.yaml"

        self.root = root
        self.sources = self._select_sources(root, sources, chosen_dataset)
        self.prepare_metadata()
        tables = [SOURCES[source] for source in self.sources]

        super().__init__(
            root=resolve_metadata_root(root, "cardiology", _csv_name(tables[0])),
            tables=tables,
            dataset_name=dataset_name or "cardiology",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def _select_sources(
        root: str, sources: list[str] | None, chosen_dataset: list[int] | None
    ) -> list[str]:
        """Resolves ``sources`` or ``chosen_dataset`` to source folders."""
        names = list(SOURCES)
        if sources is not None and chosen_dataset is not None:
            raise ValueError("Pass sources or chosen_dataset, not both.")
        if chosen_dataset is not None:
            if len(chosen_dataset) != len(names) or set(chosen_dataset) - {0, 1}:
                raise ValueError(
                    f"chosen_dataset must be {len(names)} flags of 0 or 1 for "
                    f"{names}; got {chosen_dataset}."
                )
            sources = [name for name, flag in zip(names, chosen_dataset) if flag]
        if sources is not None:
            unknown = sorted(set(sources) - set(names))
            if unknown:
                raise ValueError(f"Unknown sources {unknown}; choose from {names}.")
        selected = [name for name in names if sources is None or name in sources]
        found = [name for name in selected if (Path(root) / name).is_dir()]
        if sources is not None:
            for name in selected:
                if name not in found:
                    logger.warning("Source %s not found under %s; skipped.", name, root)
        if not found:
            raise ValueError(f"None of the sources {selected} were found under {root}.")
        return found

    def prepare_metadata(self) -> None:
        """Writes ``cardiology-<table>-pyhealth.csv`` for each selected source.

        Each CSV has one row per recording with ``patient_id`` (the record
        name), ``record_id``, ``source``, ``signal_file`` (the ``.mat``
        file), ``fs``, ``n_samples``, ``n_leads``, ``age``, ``sex`` and
        ``dx``. An existing CSV, in ``root`` or in the cache, is left
        untouched; delete it to re-index.
        """
        root = Path(self.root)
        for source in self.sources:
            csv_name = _csv_name(SOURCES[source])
            if metadata_csv_exists(root, "cardiology", csv_name):
                continue
            rows = []
            for header in sorted((root / source).glob("*/*.hea")):
                row = parse_header(header)
                row.update(
                    patient_id=row["record_id"],
                    source=source,
                    signal_file=str(header.with_suffix(".mat").resolve()),
                )
                rows.append(row)
            if not rows:
                logger.warning("No recordings found under %s", root / source)
            df = pd.DataFrame(rows, columns=_COLUMNS)
            write_metadata_csv(df, root, "cardiology", csv_name)

    @property
    def default_task(self) -> CardiologyDetection:
        """Returns arrhythmia detection, the task of the 1.x example.

        This lets ``dataset.set_task()`` run with no argument, as
        :class:`~pyhealth.datasets.CHBMITDataset` does. Pass
        ``CardiologyDetection(category)`` to detect another category.

        Returns:
            :class:`~pyhealth.tasks.CardiologyDetection` for category ``"AR"``.
        """
        return CardiologyDetection("AR")
