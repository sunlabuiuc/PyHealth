"""
Name: Ryan Kupiec 
NetID: rkupiec2
Paper Title: Time-Aware Transformer-based Network for Clinical Notes
Series Prediction
Paper link: https://proceedings.mlr.press/v126/zhang20c/zhang20c.pdf
Description: Merging of the NOTEEVENTS and ADMISSIONS datasets to create the mortality dataset required to run any of the code in the paper.
"""

from pathlib import Path
import polars as pl
from typing import List, Optional
from .base_dataset import BaseDataset

class MortalityNotesDataset(BaseDataset):
    """
    Mortality Notes dataset class for MIMIC‑III:  
    - joins admissions → noteevents  
    - drops same‑day notes  
    - removes discharge summaries  
    - fills missing charttimes with end‑of‑day  

     Attributes:
        root (str): The root directory where the dataset is stored.
        dev (bool): Whether to pull a subset of the data.
        config_path (Optional[str]): The path to the configuration file.
        dataset_name (Optional[str]): The name of the dataset.
    
    Examples:
        ds = MortalityNotesDataset(
            root="~/Path/to/folder/with/csvs",
            dev=False
        )
    """
    def __init__(
        self,
        root: str,
        dev: bool = False,
        config_path: Optional[str] = None,
        dataset_name: str = "mortality_notes",
    ):
        '''
        Initialize the MortalityNotesDataset dataset.
        Args:
            root (str): The root directory where the dataset is stored.
            dev (bool): Whether to pull a subset of the data.
            config_path (Optional[str]): The path to the configuration file.
            dataset_name (Optional[str]): The name of the dataset.
        '''
    
        config_path = Path(__file__).parent / "configs" / "mortality_notes.yaml"
        super().__init__(
            root=root,
            tables=["noteevents", "admissions"],
            dataset_name=dataset_name,
            config_path=config_path,
            dev=dev
        )
        return

    def preprocess_admissions(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = df.with_columns([
            pl.col("hadm_id").cast(pl.Int32),
            pl.col("dischtime").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date()
        ])
        return df
    
    def preprocess_noteevents(self, df: pl.LazyFrame) -> pl.LazyFrame:
        df = (
            df
            .filter(pl.col("hadm_id").is_not_null())
            .filter(pl.col("category") != "Discharge summary")
        )

        df = df.with_columns([
            pl.col("hadm_id").cast(pl.Int32),
            (pl.when(pl.col("charttime").is_null())
               .then(pl.col("chartdate") + pl.lit(" 23:59:59"))
               .otherwise(pl.col("charttime"))
             ).alias("charttime_fixed"),
        ]).drop("charttime").rename({"charttime_fixed":"charttime"})

        df = df.with_columns([
            pl.col("charttime").str.to_datetime("%Y-%m-%d %H:%M:%S").dt.date(),
            pl.col("chartdate").str.to_date("%Y-%m-%d")
        ])

        return df

    def get_final_notes(self) -> pl.DataFrame:
        df = self.collected_global_event_df
        
        admissions = df.filter(pl.col("event_type") == "admissions").rename({
            "admissions/hadm_id": "hadm_id"
        })
        notes = df.filter(pl.col("event_type") == "noteevents").rename({
            "noteevents/hadm_id": "hadm_id"
        })

        df = (
            notes
            .join(admissions.select(["patient_id", "hadm_id", "admissions/hospital_expire_flag", "admissions/dischtime"]), on=["patient_id", "hadm_id"], how="left")
        )
        df = df.with_columns(
          (pl.col("admissions/dischtime_right") - pl.col("noteevents/charttime"))
            .dt
            .total_days()
            .alias("days_before_discharge")
        )
        df = df.filter(pl.col("days_before_discharge") > 0)
        
        df = df.select([
            pl.col("hadm_id").alias("Adm_ID"),
            pl.col("noteevents/row_id").cast(pl.Int32).alias("note_id"),
            pl.col("noteevents/text").alias("text"),
            pl.col("noteevents/chartdate").alias("chartdate"),
            pl.col("noteevents/charttime").alias("charttime"),
            pl.col("admissions/hospital_expire_flag_right").cast(pl.Int32).alias("Label")
        ])
        
        return df

if __name__ == "__main__":
    ds = MortalityNotesDataset(
        root="~/Downloads/Test",
        dev=False
    )
    final = ds.get_final_notes().to_pandas()
    vc = final.drop_duplicates(subset=['Adm_ID'])["Label"].value_counts()
    print(vc)