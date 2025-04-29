import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class GDSCDataset(BaseDataset):
    """
    A dataset class for handling GDSC (Genomics of Drug Sensitivity in Cancer) data.

    This class is responsible for loading and managing the GDSC dataset, 
    which includes drug and cell line sensitivity data.

    Attributes:
        root (str): The root directory where the dataset is stored.
        dataset_name (Optional[str]): The name of the dataset.
        config_path (Optional[str]): The path to the configuration file.
        gdsc_data (pd.DataFrame): The loaded GDSC dataset.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = "gdsc",
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Initializes the GDSCDataset with the specified parameters.

        Args:
            root (str): The root directory where the dataset is stored.
            dataset_name (Optional[str]): The name of the dataset.
            config_path (Optional[str]): The path to the configuration file.
        """
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "gdsc.yaml"
        
        self.root = root
        self.dataset_name = dataset_name
        self.config_path = config_path
        self.gdsc_data = None
        
        self.load_data()

    def load_data(self) -> None:
        """
        Loads the GDSC dataset from the CSV file into a pandas DataFrame.

        The CSV file is expected to be in the specified root directory and 
        should contain drug and cell line sensitivity data.
        """
        gdsc_file_path = Path(self.root) / "gdsc.csv"
        if not gdsc_file_path.exists():
            logger.error(f"GDSC file {gdsc_file_path} does not exist!")
            raise FileNotFoundError(f"GDSC file {gdsc_file_path} not found.")

        try:
            self.gdsc_data = pd.read_csv(gdsc_file_path)
            logger.info(f"Successfully loaded GDSC data from {gdsc_file_path}")
        except Exception as e:
            logger.error(f"Error loading GDSC data: {e}")
            raise

    def get_drug_sensitivity(self, cell_line_id: str) -> Optional[pd.Series]:
        """
        Returns the drug sensitivity profile for a specific cell line.

        Args:
            cell_line_id (str): The ID of the cell line for which to fetch drug sensitivity data.

        Returns:
            pd.Series: The sensitivity data for the specified cell line.
        """
        if self.gdsc_data is None:
            logger.error("GDSC data is not loaded.")
            return None
        
        cell_line_data = self.gdsc_data[self.gdsc_data.iloc[:, 0] == cell_line_id]
        
        if cell_line_data.empty:
            logger.warning(f"Cell line {cell_line_id} not found in the dataset.")
            return None
        
        return cell_line_data.iloc[:, 1:].squeeze()

    def get_cell_lines(self) -> List[str]:
        """
        Returns a list of unique cell line IDs available in the dataset.

        Returns:
            List[str]: A list of cell line IDs.
        """
        if self.gdsc_data is None:
            logger.error("GDSC data is not loaded.")
            return []
        
        return self.gdsc_data.iloc[:, 0].unique().tolist()

    def get_drugs(self) -> List[str]:
        """
        Returns a list of drug IDs available in the dataset.

        Returns:
            List[str]: A list of drug IDs.
        """
        if self.gdsc_data is None:
            logger.error("GDSC data is not loaded.")
            return []
        
        return self.gdsc_data.columns[1:].tolist()

    def save(self, file_name: str) -> None:
        """
        Saves the GDSC dataset to a CSV file.

        Args:
            file_name (str): The name of the file to save the dataset to.
        """
        if self.gdsc_data is None:
            logger.error("No data to save.")
            return
        
        save_path = Path(self.root) / file_name
        try:
            self.gdsc_data.to_csv(save_path, index=False)
            logger.info(f"Dataset saved to {save_path}")
        except Exception as e:
            logger.error(f"Error saving GDSC data: {e}")
            raise
