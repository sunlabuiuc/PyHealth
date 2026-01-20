import logging
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from pyhealth.datasets.sample_dataset import SampleDataset
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.processors.image_processor import ImageProcessor
from pyhealth.tasks.base_task import BaseTask
from pyhealth.datasets import BaseDataset
from pyhealth.tasks import PH2MelanomaClassification

logger = logging.getLogger(__name__)

class PH2Dataset(BaseDataset):
    """
    Base image dataset for the PH2 Database (Dermatology).
    The PH2 dataset contains 200 dermoscopic images of melanocytic lesions,
    including the categories common nevus, atypical nevus, and melanoma.

    The original dataset and its images and labels can be found at the following link:
    https://www.kaggle.com/datasets/spacesurfer/ph2-dataset?resource=download 

    The dataset should be formatted as follows:

    PH2 root folder (passed as `root` argument) should contain:
    
    1. Metadata file:
       - Either 'PH2_dataset.xlsx' (official Excel from the download) 
         or 'PH2_dataset.csv' (converted version).
       - The Excel file should contain the columns:
         'Image Name', 'Common Nevus', 'Atypical Nevus', 'Melanoma', etc.
       - The first 12 rows of the Excel file will be skipped during processing.

    2. Images folder:
       - Directory named 'PH2_Dataset_images'
       - Each patient/image folder has the structure:
         PH2_Dataset_images/
           IMD001/
             IMD001_Dermoscopic_Image/
               IMD001.bmp
           IMD002/
             IMD002_Dermoscopic_Image/
               IMD002.bmp
           ...
       - Each image file name matches the folder/patient ID with a '.bmp' extension.

    3. Output after processing:
       - A CSV file 'ph2_metadata_pyhealth.csv' is automatically created by this dataset class
         which maps each image path to its corresponding label (common_nevus, atypical_nevus, melanoma).

    Notes:
    - The dataset will automatically generate labels based on the checkboxes in the Excel file.
    - Only images with valid labels and existing image files are included.
    - This dataset is compatible with the PH2MelanomaClassification task in PyHealth.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "ph2.yaml"

        # Check if we need to process the raw excel file into a CSV for PyHealth
        if not os.path.exists(os.path.join(root, "ph2_metadata_pyhealth.csv")):
            self.prepare_metadata(root)

        default_tables = ["ph2_data"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "ph2",
            config_path=str(config_path),
        )
        return

    def prepare_metadata(self, root: str) -> None:
        """Process PH2 Excel file and map image paths."""
        logger.info("Processing PH2 metadata...")
        
        # Standard filename in PH2 download is 'PH2_dataset.xlsx'
        excel_path = os.path.join(root, "PH2_dataset.xlsx")
        if not os.path.exists(excel_path):
            # Fallback if user converted it to CSV already
            excel_path = os.path.join(root, "PH2_dataset.csv")
            if not os.path.exists(excel_path):
                raise FileNotFoundError(f"Could not find PH2_dataset.xlsx in {root}")
            df = pd.read_csv(excel_path)
        else:
            # Skip the first few header rows present in PH2 excel
            df = pd.read_excel(excel_path, header=12) 

        # Rename columns
        df = df.rename(columns={
            "Image Name": "image_id",
            "Common Nevus": "common_nevus",
            "Atypical Nevus": "atypical_nevus",
            "Melanoma": "melanoma"
        })
        
        # Generate labels
        def get_diagnosis(row):
            if row.get("Melanoma") == "X": return "melanoma"
            if row.get("Atypical Nevus") == "X": return "atypical_nevus"
            if row.get("Common Nevus") == "X": return "common_nevus"
            return "Unknown"
            
        df["diagnosis"] = df.apply(get_diagnosis, axis=1)

        # Map image paths
        # PH2 Structure: root/PH2_Dataset_images/IMD003/IMD003_Dermoscopic_Image/IMD003.bmp
        image_root = os.path.join(root, "PH2_Dataset_images")
        
        def get_path(img_id):
            # Construct the path
            possible_path = os.path.join(
                image_root, 
                img_id, 
                f"{img_id}_Dermoscopic_Image", 
                f"{img_id}.bmp"
            )
            if os.path.exists(possible_path):
                return possible_path
            return None

        df["path"] = df["image_id"].apply(get_path)
        
        # Drop missing files
        df = df.dropna(subset=["path"])

        # Save processed CSV
        output_path = os.path.join(root, "ph2_metadata_pyhealth.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed PH2 metadata to {output_path}")
        return

    def set_task(
        self,
        task: Optional[BaseTask] = None,
        num_workers: int = 1,
        cache_dir: Optional[str] = None,
        cache_format: str = "parquet",
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> SampleDataset:
        
        if input_processors is None or "image" not in input_processors:
            image_processor = ImageProcessor(
                image_size=224, 
                mode="RGB", 
            )
            if input_processors is None:
                input_processors = {}
            input_processors["image"] = image_processor

        return super().set_task(
            task, num_workers, cache_dir, cache_format, 
            input_processors, output_processors
        )

    @property
    def default_task(self) -> BaseTask:
        return PH2MelanomaClassification()