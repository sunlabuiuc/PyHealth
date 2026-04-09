import pandas as pd
from pathlib import Path

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class PTBXLDataset(BaseDataset):
    """Base dataset for the PTB-XL ECG dataset

    PTB-XL is a publically available electrocardiography dataset. Contains 21837 samples from 18885 patients, all approximately 10 seconds in duration.

    Dataset is available here: https://www.kaggle.com/datasets/physionet/ptbxl-electrocardiography-database
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            tables=["ptbxl"],
            dataset_name="ptbxl",
            config_path=None,
            **kwargs,
        ) 

    def load_data(self) -> dd.DataFrame:
        """Returns a dataframe with each individual row corresponding to each .hea/.mat file combination in the PTB-XL dataset.

        Returns:
          dd.DataFrame: Dataframe with one row per record
        """
        root_path = Path(self.root)
        files = root_path.glob("*.hea")

        if not files:
            raise FileNotFoundError(f"No .hea files found in {self.root}. Are you sure you have the right directory?")

        logger.info(f"Found {len(files)} .hea files")

        rows = []
        for hea_file in files:
            age = None
            sex = None
            dx = []
            
            with open(hea_file, "r") as f:
                for line in f:
                    line = line.strip()

                    # Parse individual lines
                    if line.startswith("#Age:"):
                        try:
                            age = int(line.split(":")[1].strip())
                        except ValueError:
                            age = None
                    elif line.startswith("#Sex:"):
                        sex = line.split(":")[1].strip()
                    elif line.startswith("#Dx:"):
                        dx = [x.strip() for x in line.split(":")[1].split(",")]

              # May need a line here for the dx abbreviations

              rows.append({
                  "patient_id":     hea_file.stem,
                  "event_type":     "ptbxl",
                  "timestamp":      pd.NaT,
                  "ptbxl/mat":      str(root_path / f"{hea_file.stem}.mat"),
                  "ptbxl/age":      age,
                  "ptbxl/sex":      sex,
                  "ptbxl/dx_codes": ".".join(dx)
              })

        df = pd.Dataframe(rows)
        # TBD - Will need train and test splits

        logger.info(f"Parsed {len(df)} records.")
        return dd.from_pandas(df, npartitions=1)
        
      
        
