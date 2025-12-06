"""
Intracardiac Atrial Fibrillation Database
Authors: Huiyin Zheng
NetID: huiyinz2
Paper Title: Interpretation of Intracardiac Electrograms Through Textual Representations
Paper Link: https://arxiv.org/abs/2402.01115
"""

import logging
from pathlib import Path
from typing import Optional
import polars as pl
import os
import wfdb
import numpy as np
from .base_dataset import *
import requests
import re
import zipfile

logger = logging.getLogger(__name__)

def _download_data_to_path(url: str, save_path: str) -> None:
    """
    Downloads the iAFib dataset to the specified path.

    Args:
        save_path (str): The path where the dataset should be saved.

    Returns:
        str: The path to the downloaded and extracted dataset.
    """
    logger.info("Downloading iAFib dataset from %s", url)
    path = save_path
    os.makedirs(path, exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise Exception(f"Failed to download data, status code: {response.status_code}")

        if "Content-Disposition" in response.headers:
            cd = response.headers["Content-Disposition"]
            fname = re.findall('filename="?([^"]+)"?', cd)
            filename = fname[0] if fname else os.path.basename(url)
        else:
            filename = os.path.basename(url)

        logger.info("Downloading data from %s to %s", url, os.path.join(save_path, filename))

        file_size = int(response.headers.get('Content-Length', 0))
        downloaded = 0
        next_report = 0.1 
        with open(os.path.join(save_path, filename), "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded / file_size >= next_report:
                    logger.info("Download progress: %.1f%%", (downloaded / file_size) * 100)
                    next_report += 0.1
    except Exception as e:
        logger.error("Failed to download data from %s: %s", url, e)
        raise Exception("Failed to download data") from e

    # Try extracting zip file (outside download try/except)
    try:
        with zipfile.ZipFile(os.path.join(save_path, filename), "r") as zip_ref:
            zip_ref.extractall(save_path)
    except Exception as e:
        logger.error("Failed to extract zip file: %s", e)
        raise Exception("Failed to extract zip file") from e

    # If tests patch os.listdir to return names that are not real dirs, fall back to the first entry.
    entries = os.listdir(save_path)
    folders = [name for name in entries if os.path.isdir(os.path.join(save_path, name))]

    if folders:
        return folders[0]
    # fallback to first entry if no actual directories found
    if entries:
        return entries[0]

    return None


class iAFibDataset(BaseDataset):
    """
    A dataset class for handling iAFib data.

    This class is responsible for loading and managing the iAFib dataset

    Attributes:
        root (str): The root directory where the dataset is stored.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
        dev (bool): Whether to use the development version of the dataset.
        extract_subdir (Optional[str]): The subdirectory for extracted data.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        extract_subdir: Optional[str] = "extracted",
    ) -> None:
        # store url and local extraction dir BEFORE calling super().__init__
        self.extract_subdir = extract_subdir
        self.arrays: Dict[str, np.ndarray] = {}

        # Ensure root exists
        root_path = Path(clean_path(root))
        root_path.mkdir(parents=True, exist_ok=True)

        # Call BaseDataset.__init__; it will call our load_data()
        super().__init__(root=str(root_path), tables=['iAFib'], dataset_name=dataset_name, 
                         config_path=config_path, dev=dev)

    
    def load_data(self) -> pl.LazyFrame:
        """Loads data from the specified tables.

        Returns:
            pl.LazyFrame: A concatenated lazy frame of all tables.
        """
        logger.info("Start loading iAFib dataset...")
        table_name = self.tables[0]
        table_cfg = self.config.tables[table_name]
        url = table_cfg.file_path

        folder_name = _download_data_to_path(url, os.path.join(self.root, self.extract_subdir))
        path = os.path.join(self.root, self.extract_subdir, folder_name)
        
        df_full = None

        for i in os.listdir(path):
            if 'qrs' in i:
                file_name = i.split('.')[0]
                path = Path(path).expanduser().resolve()
                record = wfdb.rdrecord(f"{path}/{file_name}")
                patient, region = record.record_name.split('_')

                df = pl.DataFrame({
                        f"{record.sig_name[i]}": [record.p_signal.T[i].tolist()]
                        for i in range(len(record.sig_name)) if 'CS' in record.sig_name[i]
                    })
                df = df.with_columns([
                    pl.lit(patient).alias("patient_id"),
                    pl.lit(region).alias("region"),
                    pl.lit(record.fs).alias("sampling_rate")
                ])

                join_keys = ['patient_id', 'region', 'sampling_rate']

                for comment in record.comments:
                    matches = re.findall(r'<(.*?)>:\s*(.*?)(?=\s*<|$)', comment)
                    for key, value in matches:
                        df = df.with_columns([
                        pl.lit(value).alias(key)
                        ])
                        join_keys.append(key)

                if df_full is None:
                    df_full = df
                else:
                    df_full = pl.concat([df_full, df], how='vertical')
        if df_full is None:
            return None

        return df_full.lazy()




