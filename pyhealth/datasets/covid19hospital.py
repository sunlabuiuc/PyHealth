"""
Author: Ben Crumbacher
NetID: bdc6
Paper title: Approximate Bayesian Computation for an Explicit-Duration
            Hidden Markov Model of COVID-19 Hospital Trajectories
Paper link: https://www.michaelchughes.com/papers/VisaniEtAl_arXiv_2021.pdf

Description:
This module implements a dataset class for COVID-19 hospitalization data used in the
paper referenced above. The dataset utilizes the COVID Tracking Project's data to analyze
hospital trajectories during the pandemic. It provides functionality to load, filter,
and process the data for various states and time periods, supporting the creation of
explicit-duration hidden Markov models (ED-HMM) to understand patient trajectories 
through different care states (general ward, ICU with/without ventilation, discharge,
or death).

Data source: 
https://api.covidtracking.com/v1/states/daily.csv

Example usage:

dataset = Covid19Hospital(
    root="./pyhealth/datasets",
    tables=["covid_tracking"],
    us_state="MA",
    start_date="20201201",
    end_training_date="20210101",
    end_date="20210201",
    dev=False
)
"""

import logging
import os
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)

class Covid19Hospital(BaseDataset):
    """Dataset class for COVID-19 hospital data using only COVID tracking data."""

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        us_state: str = "MA",
        start_date: str = "20201111",
        end_training_date: str = "20210111",
        end_date: str = "20210211",
        **kwargs,
    ):
        """
        Initializes the Covid19Hospital dataset with COVID tracking data.

        Args:
            root (str): The root directory or URL where the COVID tracking data is stored.
            tables (List[str]): List of table names to load.
            dataset_name (Optional[str]): Name of the dataset. Defaults to class name.
            config_path (Optional[str]): Path to the configuration YAML file.
            dev (bool): Whether to run in dev mode (limits to 1000 patients).
            us_state (str): 2 letter abbreviation for state of interest.
            start_date (str): YYYYMMDD format of the start date.
            end_training_date (str): YYYYMMDD format of the end training date.
            end_date (str): YYYYMMDD format of the end date.
        """
        self._us_state = us_state
        self._start_date = start_date
        self._end_training_date = end_training_date
        self._end_date = end_date
        self.data = pd.DataFrame([])  # Raw data will be stored here
        self.filtered_data = pd.DataFrame([])  # Filtered data will be stored here

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "covid_tracking.yaml"
        
        # Initialize the base class, which will call our load_data() method
        super().__init__(root, tables, dataset_name, config_path, dev, **kwargs)

    def get_covidtracking_data(self):
        """Fetch and process COVID tracking data directly from the URL."""
        if not self.root.startswith("http"):
            logger.error("Root must be a URL when not downloading a CSV file.")
            return pd.DataFrame([])

        logger.info("Fetching COVID tracking data directly from the URL...")
        try:
            # Read the CSV data directly from the URL
            data = pd.read_csv(self.root)
            
            # Debug to check data loading
            logger.info(f"Fetched {len(data)} rows from COVID tracking data.")
            if len(data) < 10:
                logger.warning(f"Very few rows loaded: {len(data)}. Data sample: {data.head().to_dict()}")
            
            return data
        except Exception as e:
            logger.error(f"Error fetching COVID tracking data: {e}")
            return pd.DataFrame([])

    def get_filtered_data(self):
        """Filter data by state and date range."""
        if self.data.empty:
            logger.warning("No data available to filter")
            return pd.DataFrame([])
            
        # Convert dates if needed
        if 'date' in self.data.columns and not pd.api.types.is_integer_dtype(self.data['date']):
            try:
                # Debug the date format
                sample_dates = self.data['date'].head(5).tolist()
                logger.info(f"Sample dates before conversion: {sample_dates}")
                
                self.data['date'] = pd.to_datetime(self.data['date']).dt.strftime('%Y%m%d').astype(int)
                
                logger.info(f"Date conversion completed. Range: {self.data['date'].min()} to {self.data['date'].max()}")
            except Exception as e:
                logger.error(f"Error converting dates: {e}")
                return pd.DataFrame([])
        
        # Create selection column
        try:
            # Make sure state column exists
            if 'state' not in self.data.columns:
                logger.error("'state' column not found in data")
                return pd.DataFrame([])
                
            # Check if the specified state exists in the data
            if self._us_state not in self.data['state'].unique():
                logger.error(f"State '{self._us_state}' not found in data. Available states: {self.data['state'].unique()}")
                return pd.DataFrame([])
                
            # Debug the date range filter
            logger.info(f"Filtering for state={self._us_state}, date range: {self._start_date} to {self._end_date}")
            
            self.data['selected_row'] = self.data.apply(
                lambda x: 1 if (x['state'] == self._us_state and 
                               (int(self._start_date) <= x['date'] <= int(self._end_date))) 
                else 0, axis=1
            )
            
            filtered = self.data[self.data['selected_row'] == 1].sort_values(['date'], ascending=True)
            logger.info(f"Filtered {len(filtered)} rows for state {self._us_state}")
            
            # If no data was filtered, log available dates for debugging
            if len(filtered) == 0:
                state_data = self.data[self.data['state'] == self._us_state]
                if len(state_data) > 0:
                    logger.warning(f"No data found in date range. Available dates for {self._us_state}: {state_data['date'].min()} to {state_data['date'].max()}")
                else:
                    logger.warning(f"No data found for state {self._us_state}")
                    
            return filtered
        except Exception as e:
            logger.error(f"Error filtering data: {e}")
            return pd.DataFrame([])

    def get_icu_counts(self):
        """Gets the ICU counts from the COVID tracking data.
        
        This function extracts the number of patients currently in ICU from the
        global event dataframe, handling missing values by filling with NaN.
        
        Returns:
            numpy.ndarray: An array of ICU counts over time, with NaN values for
                missing data points. Empty array if no data is available.
                
        Example:
            Used in ED-HMM model fitting to track ICU occupancy trends.
        """
        df = self.collected_global_event_df
        if df.is_empty():
            return np.array([])
            
        return (
            df.filter(pl.col("event_type") == "covid_tracking")
              .select(pl.col("covid_tracking/inIcuCurrently"))
              .fill_null(float('nan'))
              .to_numpy()
        )

    def get_death_counts(self):
        """Gets the daily death counts from the COVID tracking data.
        
        This function extracts the daily death increase numbers from the 
        global event dataframe, handling missing values by filling with NaN.
        
        Returns:
            numpy.ndarray: An array of daily death counts over time, with NaN values
                for missing data points. Empty array if no data is available.
                
        Example:
            Used in mortality analysis and ED-HMM terminal state modeling.
        """
        df = self.collected_global_event_df
        if df.is_empty():
            return np.array([])
            
        return (
            df.filter(pl.col("event_type") == "covid_tracking")
              .select(pl.col("covid_tracking/deathIncrease"))
              .fill_null(float('nan'))
              .to_numpy()
        )

    def stats(self):
        """Print statistics about the dataset, extending the base class implementation."""
        # Call the parent class's stats method first to get the basic stats
        super().stats()
        
        # Add COVID-specific stats
        df = self.collected_global_event_df
        print(f"State: {self._us_state}")
        print(f"Date range: {self._start_date} to {self._end_date}")
        print(f"Columns: {list(df.columns)}")
        
        # Additional useful statistics
        if 'in_icu' in df.columns:
            icu_data = df['in_icu'].to_numpy()
            valid_icu = icu_data[~np.isnan(icu_data)]
            if len(valid_icu) > 0:
                print(f"ICU stats - Min: {np.min(valid_icu)}, Max: {np.max(valid_icu)}, Avg: {np.mean(valid_icu):.1f}")
                
        if 'deaths' in df.columns:
            deaths_data = df['deaths'].to_numpy()
            valid_deaths = deaths_data[~np.isnan(deaths_data)]
            if len(valid_deaths) > 0:
                print(f"Deaths stats - Min: {np.min(valid_deaths)}, Max: {np.max(valid_deaths)}, " 
                      f"Total: {np.sum(valid_deaths)}, Avg: {np.mean(valid_deaths):.1f}")