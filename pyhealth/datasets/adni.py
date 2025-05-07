import os
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class ADNIDataset(BaseDataset):
    """
    Data download instructions for ADNI datasets can be found here: 
    https://github.com/abyssmu/CS598-Hadi-Womack/tree/main?tab=readme-ov-file#downloading-the-data

    A dataset class for handling preprocessed ADNI MRI scans (.npz).

    This dataset reads .npz metadata from filenames, e.g.,
    'MCI-001-IMG123-75.0-F.npz', and builds a patient event table.
    Each `.npz` file should be named as: group-patientid-imageid-age-sex.npz

    Example of what table will look like after loading into table:
    ┌────────────┬──────────┬─────────────────────────────────┬───────┬───────┬──────┬─────┐
    │ patient_id ┆ image_id ┆ path                            ┆ group ┆ label ┆ age  ┆ sex │
    │ ---        ┆ ---      ┆ ---                             ┆ ---   ┆ ---   ┆ ---  ┆ --- │
    │ str        ┆ str      ┆ str                             ┆ str   ┆ i64   ┆ f64  ┆ str │
    ╞════════════╪══════════╪═════════════════════════════════╪═══════╪═══════╪══════╪═════╡
    │ 002_S_1070 ┆ I121071  ┆ /Users/nathanhadi/Documents/Sp… ┆ MCI   ┆ 1     ┆ 75.0 ┆ M   │
    │ 029_S_1218 ┆ I172267  ┆ /Users/nathanhadi/Documents/Sp… ┆ MCI   ┆ 1     ┆ 87.0 ┆ F   │
    │ 130_S_0285 ┆ I39118   ┆ /Users/nathanhadi/Documents/Sp… ┆ MCI   ┆ 1     ┆ 67.0 ┆ M   │
    │ 016_S_1121 ┆ I96238   ┆ /Users/nathanhadi/Documents/Sp… ┆ MCI   ┆ 1     ┆ 58.0 ┆ F   │
    │ 141_S_1004 ┆ I85687   ┆ /Users/nathanhadi/Documents/Sp… ┆ MCI   ┆ 1     ┆ 75.0 ┆ F   │
    └────────────┴──────────┴─────────────────────────────────┴───────┴───────┴──────┴─────┘

    Attributes:
        root (str): Root directory where dataset is stored (the .npz files).
        tables (List[str]): A list of tables to be included in the dataset. default is ['adni_table'].
        dataset_name (Optional[str]): Name of the dataset (default: 'adni').
        config_path (Optional[str]): The path to the configuration file.
    """
    def __init__(
        self,
        root: str,
        tables: List[str] = [],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "adni.yaml"

        default_tables = ["adni_table"]
        tables = default_tables + tables

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "adni",
            config_path=config_path,
            **kwargs
        )

    def load_table(self, table_name: str) -> pl.LazyFrame:
        """
        Table-specific preprocess function which will be called by BaseDataset.load_table().
        
        Loads the ADNI metadata table by parsing `.npz` filenames into structured rows.

        This method scans the root directory for `.npz` files with filenames formatted as:
        <group>-<patient_id>-<image_id>-<age>-<sex>.npz

        It extracts metadata from the filenames, including diagnostic group, patient ID,
        image ID, age, sex, and assigns a numeric label (0 = CN, 1 = MCI, 2 = AD).

        Args:
            table_name (str): The name of the table to load.

        Returns:
            pl.LazyFrame: The processed dataframe
        """
        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        rows = []
        for fname in os.listdir(self.root):
            if not fname.endswith(".npz"):
                continue

            parts = fname.replace(".npz", "").split("-")
            if len(parts) < 3:
                logger.warning(f"Skipping malformed filename: {fname}")
                continue

            group = parts[0]
            patient_id = parts[1]
            image_id = parts[2]
            age = float(parts[3]) if len(parts) >= 4 and parts[3].replace('.', '', 1).isdigit() else None
            sex = parts[4] if len(parts) >= 5 else None
            label = {"CN": 0, "MCI": 1, "AD": 2}.get(group)

            if label is None:
                logger.warning(f"Skipping unrecognized group in filename: {fname}")
                continue

            rows.append({
                "patient_id": patient_id,
                "image_id": image_id,
                "path": str(Path(self.root) / fname),
                "group": group,
                "label": label,
                "age": age,
                "sex": sex
            })

        if not rows:
            logger.warning(f"No valid .npz files found in {self.root}")
            return pl.DataFrame().lazy()

        return pl.DataFrame(rows).lazy()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dataset = ADNIDataset(
        root="/Users/nathanhadi/Documents/Spring2025/CS598/MCI_processed_data"
    )

    dataset.stats()

    df = dataset.load_table("adni_table").collect()
    print(df.head(5))