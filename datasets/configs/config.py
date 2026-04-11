"""Configuration module for PyHealth datasets.

This module provides classes and functions for loading and validating dataset
configurations in YAML format. It uses Pydantic models to ensure type safety
and validation of configuration files.
"""

from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class JoinConfig(BaseModel):
    """Configuration for joining tables in a dataset.

    Attributes:
        file_path (str): Path to the file containing the table to join.
            Relative to the dataset root directory.
        on (str): Column name to join on.
        how (str): Type of join to perform (e.g., 'left', 'right', 'inner',
            'outer').
        columns (List[str]): List of columns to include from the joined table.
    """
    file_path: str
    on: str
    how: str
    columns: List[str]


class TableConfig(BaseModel):
    """Configuration for a single table in a dataset.

    This class represents the configuration for a single table in a dataset.
    It includes the file path to the table, the column name for patient
    identifiers, the column name(s) for timestamps (if applicable), the format
    of the timestamp (if applicable), a list of attribute columns, and any join
    configurations. The join operations specified will be applied before the
    table is processed into the event DataFrame.

    Attributes:
        file_path (str): Path to the table file. Relative to the dataset root
            directory.
        patient_id (Optional[str]): Column name containing patient identifiers.
            If set to `null`, row index will be used as patient_id.
        timestamp (Optional[Union[str, List[str]]]): One or more column names to be
            used to construct the timestamp. If a list is provided, the columns will
            be concatenated in order before parsing using the provided format.
        timestamp_format (Optional[str]): Format of the (possibly concatenated) timestamp.
        attributes (List[str]): List of column names to include as attributes.
        join (List[JoinConfig]): List of join configurations for this table.
    """
    file_path: str
    patient_id: Optional[str] = None
    timestamp: Optional[Union[str, List[str]]] = None
    timestamp_format: Optional[str] = None
    attributes: List[str]
    join: List[JoinConfig] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    """Root configuration class for a dataset.

    This class represents the entire dataset configuration. It includes the
    version of the dataset and a dictionary of table configurations. Each key
    in the dictionary is a table name, and its value is a TableConfig object
    that specifies the configuration for that table.
    
    Attributes:
        version (str): The version of the dataset.
        tables (Dict[str, TableConfig]): A dictionary where each key is a table
            name and each value is a TableConfig object representing the
            configuration for that table.
    """
    version: str
    tables: Dict[str, TableConfig]

    @field_validator("tables", mode="before")
    @classmethod
    def convert_to_table_config(cls, v: Any) -> Dict[str, TableConfig]:
        return {k: TableConfig(**value) for k, value in v.items()}


def load_yaml_config(file_path: str) -> DatasetConfig:
    """Load and validate a dataset configuration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.

    Returns:
        DatasetConfig: Validated dataset configuration object.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the YAML file is malformed.
        pydantic.ValidationError: If the configuration does not match the
            expected schema.
    """
    with open(file_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return DatasetConfig.model_validate(raw_config)
