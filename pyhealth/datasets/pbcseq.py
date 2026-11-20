"""Mayo Clinic primary biliary cholangitis (PBC) sequential dataset."""

import logging
from pathlib import Path

import dask.dataframe as dd
import narwhals as nw
import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

VISIT_ANCHOR_DATE = pd.Timestamp("1974-01-01")
"""Synthetic calendar date of every patient's enrollment (day 0)."""

REQUIRED_COLUMNS = (
    "id",
    "day",
    "futime",
    "status",
    "trt",
    "age",
    "sex",
    "ascites",
    "hepato",
    "spiders",
    "edema",
    "bili",
    "chol",
    "albumin",
    "alk_phos",
    "ast",
    "platelet",
    "protime",
    "stage",
)


class PBCSeqDataset(BaseDataset):
    """Mayo Clinic PBC trial with repeated lab visits (``pbcseq``).

    312 patients from the Mayo Clinic randomized trial of D-penicillamine in
    primary biliary cholangitis (1974-1984), with 1,945 visits at baseline,
    about 6 months, 1 year, and then yearly. Each visit records labs
    (bilirubin, albumin, AST, prothrombin time, cholesterol, alkaline
    phosphatase, platelets), clinical signs, and histologic stage. Each patient
    has one outcome: ``futime`` days from enrollment to death (``status`` 2),
    liver transplant (1), or censoring (0).

    The data ships with the R ``survival`` package as ``pbcseq``. Export it to
    ``pbcseq.csv`` in ``root`` with R
    (``write.csv(survival::pbcseq, "pbcseq.csv", row.names = FALSE)``) or with
    Python (``pyreadr.read_r("pbc.rda")["pbcseq"].to_csv("pbcseq.csv",
    index=False)``). The R column ``alk.phos`` is renamed to ``alk_phos``.

    ``pbcseq`` stores visit times as days since enrollment, but PyHealth needs
    calendar timestamps, so each visit is dated
    :data:`VISIT_ANCHOR_DATE` + ``day``. The dates are synthetic: every
    patient shares the same anchor, so they only encode time since enrollment.
    The original ``day`` is kept as an attribute.

    All visits form one event table, ``pbcseq``. ``futime``, ``status``,
    ``trt``, ``age``, and ``sex`` repeat on every visit of a patient.

    This dataset supports the reproduction of the Model for Allograft Survival
    (Gao et al., MLHC 2024), whose four labs available here are bilirubin,
    albumin, AST, and prothrombin time (an INR proxy). See
    :class:`~pyhealth.tasks.LongitudinalLiverSurvival`.

    Citation:
        Murtaugh, P. A., Dickson, E. R., Van Dam, G. M., et al. "Primary
        biliary cirrhosis: Prediction of short-term survival based on repeated
        patient visits." Hepatology 20(1), 1994.

    Args:
        root: Directory containing ``pbcseq.csv``.
        tables: Tables to load. ``pbcseq`` is always loaded; leave empty.
        dataset_name: Dataset name. Defaults to ``"pbcseq"``.
        config_path: Path to a custom YAML config. Defaults to the bundled
            ``configs/pbcseq.yaml``.
        **kwargs: Passed to :class:`~pyhealth.datasets.BaseDataset`, for
            example ``cache_dir``, ``num_workers``, or ``dev``.

    Examples:
        >>> from pyhealth.datasets import PBCSeqDataset
        >>> from pyhealth.tasks import LongitudinalLiverSurvival
        >>> dataset = PBCSeqDataset(root="/path/to/pbcseq")  # doctest: +SKIP
        >>> samples = dataset.set_task(LongitudinalLiverSurvival())  # doctest: +SKIP
    """

    def __init__(
        self,
        root: str,
        tables: list[str] | None = None,
        dataset_name: str | None = None,
        config_path: str | None = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "pbcseq.yaml")
        tables = list(tables or [])
        if "pbcseq" not in tables:
            tables.insert(0, "pbcseq")
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "pbcseq",
            config_path=config_path,
            **kwargs,
        )

    def preprocess_pbcseq(self, frame: nw.LazyFrame) -> nw.LazyFrame:
        """Renames R columns and dates each visit from its day offset.

        Args:
            frame: The raw ``pbcseq`` table, with lowercased column names and
                string values.

        Returns:
            The table with ``alk_phos`` and a ``visit_date`` column
            (``YYYY-MM-DD``).

        Raises:
            ValueError: If a required column is missing.
        """
        df = frame.to_native()
        if "alk.phos" in df.columns:
            df = df.rename(columns={"alk.phos": "alk_phos"})
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"pbcseq.csv is missing columns {missing}.")
        days = dd.to_numeric(df["day"], errors="coerce")
        visit_date = VISIT_ANCHOR_DATE + dd.to_timedelta(days, unit="D")
        df = df.assign(visit_date=visit_date.dt.strftime("%Y-%m-%d"))
        return nw.from_native(df)
