import os
import tarfile
import urllib.request
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from pyhealth.datasets.base_dataset import BaseDataset


class PhysioNet2012Dataset(BaseDataset):
    """PhysioNet/CinC Challenge 2012 Dataset.

    Automatically downloads set-a and parses it into PyHealth 2.0 CSV format.

    Args:
        root (str): The root directory to store the dataset.
        tables (Optional[List[str]]): List of tables to process. Defaults to None.
        dev (bool): Whether to enable dev mode. Defaults to False.
        **kwargs: Additional keyword arguments for BaseDataset.
    """

    def __init__(
        self,
        root: str = "/tmp/physionet2012",
        tables: Optional[List[str]] = None,
        dev: bool = False,
        **kwargs
    ) -> None:
        if tables is None:
            tables = ["events", "outcomes"]
        self.dev = dev
        self.download(root)
        config_path = os.path.join(
            os.path.dirname(__file__), "configs", "physionet2012.yaml"
        )
        super().__init__(
            dataset_name="PhysioNet2012",
            root=root,
            tables=tables,
            config_path=config_path,
            dev=dev,
            **kwargs
        )

    def download(self, root: str) -> None:
        """Downloads and extracts the dataset if missing.

        Args:
            root (str): The destination root directory.
        """
        events_path = os.path.join(root, "events.csv")
        outcomes_path = os.path.join(root, "outcomes.csv")
        if os.path.exists(events_path) and os.path.exists(outcomes_path):
            return
        
        set_a_dir = os.path.join(root, "set-a")
        if not os.path.exists(set_a_dir):
            os.makedirs(root, exist_ok=True)
            tar_path = os.path.join(root, "set-a.tar.gz")
            urllib.request.urlretrieve(
                "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz",
                tar_path,
            )
            urllib.request.urlretrieve(
                "https://physionet.org/files/challenge-2012/1.0.0/Outcomes-a.txt",
                os.path.join(root, "Outcomes-a.txt"),
            )
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=root)

        outcomes = pd.read_csv(os.path.join(root, "Outcomes-a.txt"))
        outcomes.rename(
            columns={"RecordID": "recordid", "In-hospital_death": "in-hospital_death"},
            inplace=True,
        )
        outcomes.to_csv(outcomes_path, index=False)

        events =[]
        base_time = datetime(2012, 1, 1)
        files =[f for f in os.listdir(set_a_dir) if f.endswith(".txt")]
        
        for file in tqdm(files, desc="Parsing TXT files"):
            recordid = file.split(".")[0]
            df = pd.read_csv(os.path.join(set_a_dir, file))
            df["recordid"] = recordid

            def parse_time(t_str: str) -> datetime:
                h, m = map(int, t_str.split(":"))
                return base_time + timedelta(hours=h, minutes=m)

            df["time"] = df["Time"].apply(parse_time)
            df.rename(
                columns={"Parameter": "parameter", "Value": "value"}, 
                inplace=True
            )
            events.append(df[["recordid", "time", "parameter", "value"]])

        events_df = pd.concat(events, ignore_index=True)
        events_df.to_csv(events_path, index=False)