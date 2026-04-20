import logging
import os
import subprocess
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

from pyhealth.datasets.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PhysioNet2012Dataset(BaseDataset):
    """PhysioNet/CinC Challenge 2012 Dataset.

    Automatically downloads the set-a and set-b training sets and parses them into the
    standardized PyHealth CSV format (events.csv + outcomes.csv).

    Dataset is available at:
    https://physionet.org/content/challenge-2012/1.0.0/

    Citations:
    ----------
    If you use this dataset, please cite:
    Silva I, Moody G, Scott DJ, Celi LA, Mark RG. Predicting in-hospital mortality
    of ICU patients: The PhysioNet/Computing in Cardiology Challenge 2012.
    Computing in Cardiology. 2012;39:245-248.

    Args:
        root: Root directory where the dataset will be stored.
        dataset_name: Optional name of the dataset. Defaults to "physionet2012".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.
        dev: Whether to run in dev mode (limits to a small subset of patients).

    Attributes:
        root: Root directory of the dataset.
        dataset_name: Name of the dataset.
        config_path: Path to configuration file.

    Expected Files & Structure (after first run):
    -------------------------------------------
    {root}/
    ├── events.csv # time-series vital-sign / lab parameters
    ├── outcomes.csv # in-hospital_death label per recordid
    ├── set-a/ # raw TXT files (kept for reproducibility)
    └── set-b/ # raw TXT files (kept for reproducibility)

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> dataset = PhysioNet2012Dataset(root="/path/to/physionet2012")
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "physionet2012.yaml"

        # Create the standardized CSVs if they do not already exist
        events_path = os.path.join(root, "events.csv")
        outcomes_path = os.path.join(root, "outcomes.csv")
        if not (os.path.exists(events_path) and os.path.exists(outcomes_path)):
            logger.info(
                "Preparing PhysioNet2012 metadata (downloading + parsing set-a and set-b)..."
            )
            self.prepare_metadata(root)

        # Use tables from kwargs if provided (for tests), otherwise use default
        tables = kwargs.pop("tables", None)
        if tables is None:
            tables = ["events", "outcomes"]

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "physionet2012",
            config_path=config_path,
            dev=dev,
            **kwargs,
        )

    def prepare_metadata(self, root: str) -> None:
        """Downloads set-a.zip, set-b.zip, Outcomes-a.txt, and Outcomes-b.txt (if needed) and
        converts them into the two standardized CSVs expected by the config.

        This method is idempotent: if events.csv and outcomes.csv already
        exist, it returns immediately.
        """
        events_path = os.path.join(root, "events.csv")
        outcomes_path = os.path.join(root, "outcomes.csv")

        # Early exit if CSVs are already present
        if os.path.exists(events_path) and os.path.exists(outcomes_path):
            return

        os.makedirs(root, exist_ok=True)
        set_a_dir = os.path.join(root, "set-a")
        set_b_dir = os.path.join(root, "set-b")

        # Download raw data only if the set-a or set-b directories are missing
        if not os.path.exists(set_a_dir) or not os.path.exists(set_b_dir):
            zip_a_path = os.path.join(root, "set-a.zip")
            zip_b_path = os.path.join(root, "set-b.zip")
            outcomes_a_txt_path = os.path.join(root, "Outcomes-a.txt")
            outcomes_b_txt_path = os.path.join(root, "Outcomes-b.txt")

            logger.info(f"Downloading set-a.zip to {zip_a_path} via AWS S3")
            subprocess.run([
                "aws", "s3", "cp", "--no-sign-request", 
                "s3://physionet-open/challenge-2012/1.0.0/set-a.zip", 
                zip_a_path
            ], check=True)

            logger.info(f"Downloading set-b.zip to {zip_b_path} via AWS S3")
            subprocess.run([
                "aws", "s3", "cp", "--no-sign-request", 
                "s3://physionet-open/challenge-2012/1.0.0/set-b.zip", 
                zip_b_path
            ], check=True)

            logger.info(f"Downloading Outcomes-a.txt to {outcomes_a_txt_path} via AWS S3")
            subprocess.run([
                "aws", "s3", "cp", "--no-sign-request", 
                "s3://physionet-open/challenge-2012/1.0.0/Outcomes-a.txt", 
                outcomes_a_txt_path
            ], check=True)

            logger.info(f"Downloading Outcomes-b.txt to {outcomes_b_txt_path} via AWS S3")
            subprocess.run([
                "aws", "s3", "cp", "--no-sign-request", 
                "s3://physionet-open/challenge-2012/1.0.0/Outcomes-b.txt", 
                outcomes_b_txt_path
            ], check=True)

            logger.info("Extracting set-a.zip ...")
            with zipfile.ZipFile(zip_a_path, "r") as zip_ref:
                zip_ref.extractall(path=root)

            logger.info("Extracting set-b.zip ...")
            with zipfile.ZipFile(zip_b_path, "r") as zip_ref:
                zip_ref.extractall(path=root)

        # ------------------------------------------------------------------
        # 1. Outcomes
        # ------------------------------------------------------------------
        logger.info("Parsing Outcomes-a.txt and Outcomes-b.txt -> outcomes.csv")
        outcomes_a = pd.read_csv(os.path.join(root, "Outcomes-a.txt"))
        outcomes_b = pd.read_csv(os.path.join(root, "Outcomes-b.txt"))
        outcomes = pd.concat([outcomes_a, outcomes_b], ignore_index=True)
        outcomes.rename(
            columns={
                "RecordID": "recordid",
                "In-hospital_death": "in-hospital_death",
            },
            inplace=True,
        )
        outcomes.to_csv(outcomes_path, index=False)

        # ------------------------------------------------------------------
        # 2. Events
        # ------------------------------------------------------------------
        logger.info("Parsing set-a/*.txt and set-b/*.txt -> events.csv")
        events =[]
        base_time = datetime(2012, 1, 1)
        
        txt_files_a =[os.path.join(set_a_dir, f) for f in os.listdir(set_a_dir) if f.endswith(".txt")]
        txt_files_b =[os.path.join(set_b_dir, f) for f in os.listdir(set_b_dir) if f.endswith(".txt")]
        txt_files = txt_files_a + txt_files_b

        for file_path in tqdm(txt_files, desc="Parsing TXT files"):
            recordid = os.path.basename(file_path).split(".")[0]
            df = pd.read_csv(file_path)
            df["recordid"] = recordid

            def parse_time(t_str: str) -> datetime:
                h, m = map(int, t_str.split(":"))
                return base_time + timedelta(hours=h, minutes=m)

            df["time"] = df["Time"].apply(parse_time)
            df.rename(
                columns={"Parameter": "parameter", "Value": "value"},
                inplace=True,
            )
            events.append(df[["recordid", "time", "parameter", "value"]])

        events_df = pd.concat(events, ignore_index=True)
        events_df.to_csv(events_path, index=False)

        logger.info(
            f"PhysioNet2012 metadata prepared: {len(events_df)} events, "
            f"{len(outcomes)} outcomes"
        )