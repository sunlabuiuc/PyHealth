import logging
from typing import List, Optional

from .base_dataset import BaseDataset

import pandas as pd
import re
import os
import pydicom

logger = logging.getLogger(__name__)


class MIMICCxr(BaseDataset):
    """
        This class is responsible for handling MIMIC-CXR (version 2.1.0) dataset.

        MIMIC-CXR v2.1.0 contains:
            - A set of 10 folders (p10 - p19), each with ~6,500 sub-folders.
                Sub-folders are named according to the patient identifier, and contain free-text reports and
                DICOM files for all studies for that patient
            - cxr-record-list.csv.gz - a compressed file providing the link between an image,
                its corresponding study identifier, and its corresponding patient identifier
            - cxr-study-list.csv.gz - a compressed file providing a link between anonymous study and patient identifiers
            - cxr-provider-list.csv.gz - a compressed file providing the ordering, attending, and resident provider
                associated with the given radiology study
            - mimic-cxr-reports.tar.gz - for convenience, all free-text reports have been compressed in a single archive
                file

        Attributes:
            root (str): The root directory where dataset files are stored.
            tables (List[str]): List of csv file name (including file extension).
            dataset_name (Optional[str]): The name of the dataset.
            config_path (Optional[str]): The path to the configuration file.
        """

    def __init__(
            self,
            root: str,
            tables: List[str],
            dataset_name: Optional[str] = None,
            config_path: Optional[str] = None,
    ):
        self.view_positions = self.extract_view_positions()

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic_cxr",
            config_path=config_path,
        )

    def create_dataframe(self, file_name):
        df = pd.read_csv(f"{self.root}/{file_name}")
        df.fillna('', inplace=True)
        return df

    @staticmethod
    def ap_pattern():
        return re.compile(
            r"(AP\s*chest|Portable.*AP|anteroposterior|single frontal view|portable (semi-upright|semi-erect))|AP VIEW|AP ONLY",
            re.IGNORECASE)

    def extract_view_positions(self):
        view_positions = {}
        # Iterate through each DICOM file to extract ViewPosition metadata
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            for filename in filenames:
                if filename.lower().endswith('.dcm'):
                    dcm_path = os.path.join(dirpath, filename)
                    dcm_data = pydicom.dcmread(dcm_path, stop_before_pixels=True)

                    # Check if ViewPosition metadata is present
                    view_position = getattr(dcm_data, 'ViewPosition', 'Unknown')
                    view_positions[filename] = view_position

        return view_positions
