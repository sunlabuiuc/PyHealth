import logging
from pathlib import Path
from typing import List, Optional
import pandas as pd
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MIMIC3CirculatoryFailureDataset(BaseDataset):
    """MIMIC-III dataset for circulatory failure early-warning prediction.

    This dataset is designed for a FAMEWS-inspired reproduction setting on
    MIMIC-III. It will support cohort construction, event parsing, and
    time-series feature extraction for circulatory failure prediction within
    a future prediction window.

    Args:
        root: Root directory of the MIMIC-III dataset.
        tables: Additional tables to load beyond the default cohort tables.
        dataset_name: Name of the dataset instance.
        config_path: Path to the dataset config YAML file.
        **kwargs: Additional keyword arguments passed to BaseDataset.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Initializes the MIMIC-III circulatory failure dataset."""
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3_cf.yaml"

        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3_cf",
            config_path=str(config_path),
            **kwargs,
        )

    def load_cohort(self):
        """Load patients + admissions + icustays."""

        import pandas as pd
        from pathlib import Path

        root = Path(self.root)

        patients_df = pd.read_csv(root / "PATIENTS.csv.gz")
        admissions_df = pd.read_csv(root / "ADMISSIONS.csv.gz")
        icustays_df = pd.read_csv(root / "ICUSTAYS.csv.gz")

        df = patients_df.merge(admissions_df, on="SUBJECT_ID")
        df = df.merge(icustays_df, on=["SUBJECT_ID", "HADM_ID"])

        patients = []

        for _, row in df.iterrows():
            patients.append(
                {
                    "patient_id": row["SUBJECT_ID"],
                    "gender": row["GENDER"],
                    "hadm_id": row["HADM_ID"],
                    "icustay_id": row["ICUSTAY_ID"],
                    "admittime": row["ADMITTIME"],
                    "intime": row["INTIME"],
                    "outtime": row["OUTTIME"],
                }
            )

        return patients

    def load_patients(self):
        """Backward-compatible wrapper for current development."""
        return self.load_cohort()

    def build_failure_labels(self):
        """Build first failure time per ICU stay (MAP < 65) using chunked reads."""

        import pandas as pd
        from pathlib import Path

        root = Path(self.root)

        # load cohort once
        cohort = pd.DataFrame(self.load_cohort())
        cohort["intime"] = pd.to_datetime(cohort["intime"])
        cohort["outtime"] = pd.to_datetime(cohort["outtime"])

        results = []

        chunks = pd.read_csv(
            root / "CHARTEVENTS.csv.gz",
            usecols=[
                "SUBJECT_ID",
                "HADM_ID",
                "ICUSTAY_ID",
                "ITEMID",
                "CHARTTIME",
                "VALUENUM",
            ],
            chunksize=50000,
        )

        for chunk in chunks:
            # filter MAP only
            chunk = chunk[chunk["ITEMID"] == 220052].copy()
            if chunk.empty:
                continue

            chunk["CHARTTIME"] = pd.to_datetime(
                chunk["CHARTTIME"],
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce",
            )

            merged = chunk.merge(
                cohort,
                left_on="ICUSTAY_ID",
                right_on="icustay_id",
            )

            if merged.empty:
                continue

            filtered = merged[
                (merged["CHARTTIME"] >= merged["intime"])
                & (merged["CHARTTIME"] <= merged["outtime"])
            ].copy()

            if filtered.empty:
                continue

            filtered["failure_label"] = (filtered["VALUENUM"] < 65).astype(int)

            failure_events = filtered[filtered["failure_label"] == 1]
            if failure_events.empty:
                continue

            first_failure_chunk = (
                failure_events.groupby("ICUSTAY_ID")["CHARTTIME"]
                .min()
                .reset_index()
                .rename(columns={"CHARTTIME": "first_failure_time"})
            )

            results.append(first_failure_chunk)

        if not results:
            return pd.DataFrame(columns=["ICUSTAY_ID", "first_failure_time"])

        first_failure = pd.concat(results, ignore_index=True)

        # keep earliest failure time per ICU stay across all chunks
        first_failure = (
            first_failure.groupby("ICUSTAY_ID")["first_failure_time"]
            .min()
            .reset_index()
        )

        return first_failure

        # filter MAP
        map_df = chartevents[chartevents["ITEMID"] == 220052]

        # convert time
        map_df["CHARTTIME"] = pd.to_datetime(map_df["CHARTTIME"])

        # load cohort
        cohort = pd.DataFrame(self.load_cohort())
        cohort["intime"] = pd.to_datetime(cohort["intime"])
        cohort["outtime"] = pd.to_datetime(cohort["outtime"])

        # merge
        merged = map_df.merge(
            cohort,
            left_on="ICUSTAY_ID",
            right_on="icustay_id",
        )

        # filter ICU period
        filtered = merged[
            (merged["CHARTTIME"] >= merged["intime"])
            & (merged["CHARTTIME"] <= merged["outtime"])
        ].copy()

        # label
        filtered["failure_label"] = (filtered["VALUENUM"] < 65).astype(int)

        # first failure time
        failure_events = filtered[filtered["failure_label"] == 1]

        first_failure = (
            failure_events.groupby("ICUSTAY_ID")["CHARTTIME"]
            .min()
            .reset_index()
            .rename(columns={"CHARTTIME": "first_failure_time"})
        )

        return first_failure

    def get_patient_by_icustay_id(self, icustay_id: int):
        """Build one task-ready patient dict for a given ICU stay."""

        import pandas as pd
        from pathlib import Path

        root = Path(self.root)

        # 1) load cohort
        cohort_df = pd.DataFrame(self.load_cohort())
        cohort_df["intime"] = pd.to_datetime(cohort_df["intime"])
        cohort_df["outtime"] = pd.to_datetime(cohort_df["outtime"])

        row = cohort_df[cohort_df["icustay_id"] == icustay_id]
        if row.empty:
            return None
        row = row.iloc[0]

        # 2) load failure labels
        first_failure = self.build_failure_labels()
        failure_row = first_failure[first_failure["ICUSTAY_ID"] == icustay_id]

        first_failure_time = None
        if not failure_row.empty:
            first_failure_time = failure_row.iloc[0]["first_failure_time"]

        # 3) load MAP time series for this ICU stay
        chartevents = pd.read_csv(
            root / "CHARTEVENTS.csv.gz",
            usecols=["ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
        )

        ts = chartevents[chartevents["ITEMID"] == 220052].copy()
        ts["CHARTTIME"] = pd.to_datetime(
            ts["CHARTTIME"],
            format="%Y-%m-%d %H:%M:%S",
            errors="coerce",
        )
        ts = ts[ts["ICUSTAY_ID"] == icustay_id].copy()

        ts = ts[
            (ts["CHARTTIME"] >= row["intime"]) &
            (ts["CHARTTIME"] <= row["outtime"])
        ].copy()

        ts = ts.sort_values("CHARTTIME")

        time_series = []
        for _, ts_row in ts.iterrows():
            if pd.isna(ts_row["VALUENUM"]):
                continue
            time_series.append(
                {
                    "charttime": ts_row["CHARTTIME"],
                    "map": float(ts_row["VALUENUM"]),
                }
            )

        patient = {
            "patient_id": int(row["patient_id"]),
            "icustay_id": int(row["icustay_id"]),
            "gender": row["gender"],
            "intime": row["intime"],
            "outtime": row["outtime"],
            "time_series": time_series,
            "first_failure_time": first_failure_time,
        }

        return patient