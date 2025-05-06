import logging
import warnings
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np
from pyhealth.datasets import MIMIC3Dataset
import ast
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CheXpertMappedDataset(BaseDataset):
    """
    A dataset class for mapping CheXpert data to ICD-9 codes through known ICD-10 codes.

    Based on the Paper: Analysis of Integrating ChatGPT into Secure Hospital Networks
    Since the paper's dataset is not publicly available, we use CheXpert as a placeholder.
    The mapping is done using a sample label to ICD-10 mapping, which can be extended as needed.

    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mimic3.yaml"
        default_tables = ["patients", "admissions", "icustays"]
        tables = default_tables + tables
        if "prescriptions" in tables:
            warnings.warn(
                "Events from prescriptions table only have date timestamp (no specific time). "
                "This may affect temporal ordering of events.",
                UserWarning,
            )
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3",
            config_path=config_path,
            **kwargs
        )
        return

    def map_data(self):
        base_data = pd.read_csv('pyhealth/datasets/utils/chexpert_data.csv')
        base_data.head()
        
        # 1. Load CheXpert label file
        ## Take a random subset of 10,000 records for the pyhealth mapping
        pyhealth_data = base_data.sample(n=10000, random_state=42)
        pyhealth_data.head()

        ### Generate a list of labels and automate the mapping
        LABEL_COLUMNS = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation'
        ]

        mimic3base = MIMIC3Dataset(
            root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        )

        # 2. Label to ICD-10 mapping
        label_to_icd = {
            'Cardiomegaly': 'I51.7',
            'Edema': 'R60.9',
            'Consolidation': 'J98.4',
            'Atelectasis': 'J98.11',
            'Pleural Effusion': 'J90',
            'Pneumonia': 'J18.9',
            'Pneumothorax': 'J93.9',
            'Fracture': 'S22.3',
            'Lung Lesion': 'D14.3',
            'Enlarged Cardiomediastinum': 'I51.7',
        }

        # 3. Function to convert CheXpert row to ICD codes
        def get_icd_codes_from_labels(row):
            icd_codes = []
            for label, icd in label_to_icd.items():
                if pd.notnull(row[label]) and row[label] == 1.0:
                    icd_codes.append(icd)
            return icd_codes

        # 4. Process dataset
        pyhealth_data['ICD_Codes'] = pyhealth_data.apply(get_icd_codes_from_labels, axis=1)

        # 5. Output example structure
        processed_df = pyhealth_data[['Path', 'ICD_Codes']]
        processed_df['Path'] = processed_df['Path'].apply(lambda x: '/content/CheXpert-v1.0-small/' + x)

        # Save the processed dataframe
        processed_df.to_csv('chexpert_icd_mapped.csv', index=False)

        new_data= pd.read_csv('/content/chexpert_icd_mapped.csv')
        print("new mapped data", new_data.head())

        # Ensure ICD_Codes is a list, not a string
        if isinstance(new_data["ICD_Codes"].iloc[0], str):
            new_data["ICD_Codes"] = new_data["ICD_Codes"].apply(ast.literal_eval)

        # Load the ICD-10 → ICD-9 mapping file
        mapping_df = pd.read_csv('pyhealth/datasets/utils/icd10cmtoicd9gem.csv')

        # Build the mapping dictionary
        icd10_to_icd9_dict = mapping_df.groupby("icd10cm")["icd9cm"].apply(list).to_dict()

        # Define the mapping function
        def map_icd10_to_icd9(icd10_list):
            icd9_codes = set()
            for code in icd10_list:
                normalized = code.replace(".", "").upper()
                if normalized in icd10_to_icd9_dict:
                    icd9_codes.update(icd10_to_icd9_dict[normalized])
            return list(icd9_codes)

        # Apply the function to map ICD-10 to ICD-9
        new_data["ICD9_Mapped"] = new_data["ICD_Codes"].apply(map_icd10_to_icd9)

        # Preview the result
        print(new_data[["Path", "ICD_Codes", "ICD9_Mapped"]].head())

        return(new_data.to_csv('final_pyhealth_mapped.csv', index=False))
