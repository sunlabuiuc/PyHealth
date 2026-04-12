import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from pyhealth.datasets import BaseDataset
from pyhealth.tasks.shhs_sleep_staging import SleepStagingSHHS

logger = logging.getLogger(__name__)

_VISIT_ECG_RATES = {"shhs1": 125, "shhs2": 256}


class SHHSDataset(BaseDataset):
    """Dataset for the Sleep Heart Health Study (SHHS).

    Dataset is available at https://sleepdata.org/datasets/shhs

    The Sleep Heart Health Study (SHHS) is a multi-center cohort study implemented
    by the National Heart Lung & Blood Institute to determine the cardiovascular and
    other consequences of sleep-disordered breathing. It tests whether sleep-related
    breathing is associated with an increased risk of coronary heart disease, stroke,
    all cause mortality, and hypertension.

    In all, 6,441 men and women aged 40 years and older were enrolled between
    November 1, 1995 and January 31, 1998 to take part in SHHS Visit 1.

    During exam cycle 3 (January 2001- June 2003), a second
    polysomnogram (SHHS Visit 2) was obtained in 3,295 of the participants.

    CVD Outcomes data were monitored and adjudicated by parent cohorts between baseline
    and 2011. More than 130 manuscripts have been published investigating predictors
    and outcomes of sleep disorders.

    This loader expects the standard NSRR directory layout:

        root/
            polysomnography/
                edfs/shhs1/  # EDF signal files
                edfs/shhs2/  # EDF signal files
                annotations-events-profusion/shhs1/  # Profusion XML
                annotations-events-profusion/shhs2/  # Profusion XML
            datasets/
                shhs-harmonized-dataset-0.21.0.csv

    Args:
        root: Root directory containing the SHHS download.
        dataset_name: Optional name override.
        config_path: Optional path to a custom YAML config.
        cache_dir: Optional directory for caching processed data.
        num_workers: Number of worker processes for parallel operations.
        dev: Whether to run in dev mode (limits to 1000 patients).

    Examples:
        >>> from pyhealth.datasets import SHHSDataset
        >>> dataset = SHHSDataset(root="/path/to/shhs")
        >>> dataset.stats()
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
            config_path = str(Path(__file__).parent / "configs" / "shhs.yaml")

        metadata_path = Path(root) / "shhs-metadata.csv"
        if not metadata_path.exists():
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["shhs_sleep"],
            dataset_name=dataset_name or "shhs_sleep",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @property
    def default_task(self) -> SleepStagingSHHS:
        return SleepStagingSHHS()

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Build shhs-metadata.csv by joining EDF paths with demographics.

        Only recordings that have both an EDF and a matching Profusion XML
        annotation file are included.
        """
        poly_root = Path(root) / "polysomnography"

        records: list[dict] = []
        for visit in ("shhs1", "shhs2"):
            edf_dir = poly_root / "edfs" / visit
            ann_dir = poly_root / "annotations-events-profusion" / visit

            if not edf_dir.is_dir():
                logger.info("EDF directory not found, skipping: %s", edf_dir)
                continue

            ecg_rate = _VISIT_ECG_RATES[visit]

            for edf_file in sorted(edf_dir.iterdir()):
                if edf_file.suffix != ".edf":
                    continue

                # filename pattern: shhs1-200001.edf
                nsrrid = edf_file.stem.split("-")[1]

                ann_file = ann_dir / f"{edf_file.stem}-profusion.xml"
                if not ann_file.exists():
                    logger.debug("No annotation for %s, skipping", edf_file.name)
                    continue

                records.append(
                    {
                        "patient_id": nsrrid,
                        "visitnumber": 1 if visit == "shhs1" else 2,
                        "signal_file": str(edf_file),
                        "annotation_file": str(ann_file),
                        "ecg_sample_rate": ecg_rate,
                    }
                )

        if not records:
            raise FileNotFoundError(
                f"No matched EDF/XML pairs found under {poly_root}"
            )

        metadata = pd.DataFrame(records)

        # Merge demographics from harmonized CSV
        harmonized = _find_harmonized_csv(root)
        if harmonized is not None:
            demo = pd.read_csv(
                harmonized,
                usecols=[
                    "nsrrid",
                    "visitnumber",
                    "nsrr_age",
                    "nsrr_sex",
                    "nsrr_bmi",
                    "nsrr_ahi_hp3r_aasm15",
                ],
                dtype={"nsrrid": str},
            )
            demo = demo.rename(
                columns={
                    "nsrrid": "patient_id",
                    "nsrr_age": "age",
                    "nsrr_sex": "sex",
                    "nsrr_bmi": "bmi",
                    "nsrr_ahi_hp3r_aasm15": "ahi",
                }
            )
            metadata = metadata.merge(
                demo, on=["patient_id", "visitnumber"], how="left"
            )
        else:
            logger.warning("Harmonized CSV not found; demographics will be empty")
            for col in ("age", "sex", "bmi", "ahi"):
                metadata[col] = None

        output_path = Path(root) / "shhs-metadata.csv"
        metadata.to_csv(output_path, index=False)
        logger.info("Wrote %d records to %s", len(metadata), output_path)


def _find_harmonized_csv(root: str) -> Optional[str]:
    """Locate the harmonized dataset CSV under root/datasets/."""
    datasets_dir = Path(root) / "datasets"
    if not datasets_dir.is_dir():
        return None
    for f in sorted(datasets_dir.iterdir(), reverse=True):
        if f.name.startswith("shhs-harmonized-dataset") and f.suffix == ".csv":
            return str(f)
    return None
