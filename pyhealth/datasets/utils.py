import hashlib
import os
import pickle
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any, Union, Set

import torch
import numpy as np
import pandas as pd
from dateutil.parser import parse as dateutil_parse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory

MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "datasets")
create_directory(MODULE_CACHE_PATH)

logger = logging.getLogger(__name__)

# basic tables which are a part of the defined datasets
DATASET_BASIC_TABLES = {
    "MIMIC3Dataset": {"PATIENTS", "ADMISSIONS"},
    "MIMIC4Dataset": {"patients", "admission"},
}


def hash_str(s):
    return hashlib.md5(s.encode()).hexdigest()


def strptime(s: str) -> Optional[datetime]:
    """Helper function which parses a string to datetime object.

    Args:
        s: str, string to be parsed.

    Returns:
        Optional[datetime], parsed datetime object. If s is nan, return None.
    """
    # return None if s is nan
    if s != s:
        return None
    return dateutil_parse(s)

def padyear(year: str, month='1', day='1') -> str:
    """Pad a date time year of format 'YYYY' to format 'YYYY-MM-DD'
    
    Args: 
        year: str, year to be padded. Must be non-zero value.
        month: str, month string to be used as padding. Must be in [1, 12]
        day: str, day string to be used as padding. Must be in [1, 31]
        
    Returns:
        padded_date: str, padded year.
    
    """
    return f"{year}-{month}-{day}"

def flatten_list(l: List) -> List:
    """Flattens a list of list.

    Args:
        l: List, the list of list to be flattened.

    Returns:
        List, the flattened list.

    Examples:
        >>> flatten_list([[1], [2, 3], [4]])
        [1, 2, 3, 4]R
        >>> flatten_list([[1], [[2], 3], [4]])
        [1, [2], 3, 4]
    """
    assert isinstance(l, list), "l must be a list."
    return sum(l, [])


def list_nested_levels(l: List) -> Tuple[int]:
    """Gets all the different nested levels of a list.

    Args:
        l: the list to be checked.

    Returns:
        All the different nested levels of the list.

    Examples:
        >>> list_nested_levels([])
        (1,)
        >>> list_nested_levels([1, 2, 3])
        (1,)
        >>> list_nested_levels([[]])
        (2,)
        >>> list_nested_levels([[1, 2, 3], [4, 5, 6]])
        (2,)
        >>> list_nested_levels([1, [2, 3], 4])
        (1, 2)
        >>> list_nested_levels([[1, [2, 3], 4]])
        (2, 3)
    """
    if not isinstance(l, list):
        return tuple([0])
    if not l:
        return tuple([1])
    levels = []
    for i in l:
        levels.extend(list_nested_levels(i))
    levels = [i + 1 for i in levels]
    return tuple(set(levels))


def is_homo_list(l: List) -> bool:
    """Checks if a list is homogeneous.

    Args:
        l: the list to be checked.

    Returns:
        bool, True if the list is homogeneous, False otherwise.

    Examples:
        >>> is_homo_list([1, 2, 3])
        True
        >>> is_homo_list([])
        True
        >>> is_homo_list([1, 2, "3"])
        False
        >>> is_homo_list([1, 2, 3, [4, 5, 6]])
        False
    """
    if not l:
        return True

    # if the value vector is a mix of float and int, convert all to float
    l = [float(i) if type(i) == int else i for i in l]
    return all(isinstance(i, type(l[0])) for i in l)


def collate_fn_dict(batch: List[dict]) -> dict:
    """Collates a batch of data into a dictionary of lists.

    Args:
        batch: List of dictionaries, where each dictionary represents a data sample.

    Returns:
        A dictionary where each key corresponds to a list of values from the batch.
    """
    return {key: [d[key] for d in batch] for key in batch[0]}


def collate_fn_dict_with_padding(batch: List[dict]) -> dict:
    """Collates a batch of data into a dictionary with padding for tensor values.

    Args:
        batch: List of dictionaries, where each dictionary represents a data sample.

    Returns:
        A dictionary where each key corresponds to a list of values from the batch.
        Tensor values are padded to the same shape.
    """
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        if isinstance(values[0], torch.Tensor):
            # Check if shapes are the same
            shapes = [v.shape for v in values]
            if all(shape == shapes[0] for shape in shapes):
                # Same shape, just stack
                collated[key] = torch.stack(values)
            else:
                # Variable shapes, pad
                if values[0].dim() == 0:
                    # Scalars, treat as stackable
                    collated[key] = torch.stack(values)
                elif values[0].dim() >= 1:
                    collated[key] = pad_sequence(values, batch_first=True, padding_value=0)
                else:
                    raise ValueError(f"Unsupported tensor shape: {values[0].shape}")
        else:
            # Non-tensor data: keep as list
            collated[key] = values

    return collated


def get_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Creates a DataLoader for a given dataset.

    Args:
        dataset: The dataset to load data from.
        batch_size: The number of samples per batch.
        shuffle: Whether to shuffle the data at every epoch.

    Returns:
        A DataLoader instance for the dataset.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn_dict_with_padding,
    )
    return dataloader


def validate_dataset_schema(dataset, required_tables: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Validates the schema of a PyHealth dataset against expected structure.
    
    This function checks if a dataset has the required tables and validates the 
    structure of each table against expected schema.
    
    Args:
        dataset: A PyHealth BaseDataset subclass instance.
        required_tables: Optional[Set[str]], set of required table names.
            If None, uses the basic tables defined for the dataset class.
            
    Returns:
        Dict[str, Any]: Validation results containing:
            - 'valid': bool, overall validation result
            - 'missing_tables': List[str], tables that are missing
            - 'table_validation': Dict[str, Dict], validation results for each table
            
    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(root='/path/to/mimic3')
        >>> results = validate_dataset_schema(dataset)
        >>> print(results['valid'])
    """
    results = {
        'valid': True,
        'missing_tables': [],
        'table_validation': {}
    }
    
    # Get dataset class name
    dataset_class_name = dataset.__class__.__name__
    
    # Determine required tables
    if required_tables is None:
        if dataset_class_name in DATASET_BASIC_TABLES:
            required_tables = DATASET_BASIC_TABLES[dataset_class_name]
        else:
            logger.warning(f"No basic tables defined for {dataset_class_name}. Skipping table validation.")
            return results
    
    # Check if dataset has tables attribute
    if not hasattr(dataset, 'tables'):
        results['valid'] = False
        results['error'] = "Dataset does not have 'tables' attribute"
        return results
    
    # Check for missing tables
    dataset_tables = set(dataset.tables)
    if required_tables:
        missing_tables = required_tables - dataset_tables
        if missing_tables:
            results['valid'] = False
            results['missing_tables'] = list(missing_tables)
    
    # Validate each table's schema if possible
    if hasattr(dataset, 'global_event_df'):
        try:
            # Get schema information from the lazy frame
            schema = dataset.global_event_df.collect_schema()
            column_names = schema.names()
            
            # Check for essential columns
            essential_columns = {'patient_id'}
            missing_essential = essential_columns - set(column_names)
            
            if missing_essential:
                results['valid'] = False
                results['missing_essential_columns'] = list(missing_essential)
                
            # Add schema info to results
            results['schema'] = {
                'columns': column_names,
                'dtypes': {name: str(schema.field(name).dtype) for name in column_names}
            }
            
        except Exception as e:
            results['valid'] = False
            results['schema_error'] = str(e)
    
    return results


def assess_dataset_quality(dataset, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """Assesses the quality of a PyHealth dataset.
    
    This function analyzes a dataset to identify potential quality issues such as
    missing values, outliers, and class imbalance.
    
    Args:
        dataset: A PyHealth BaseDataset subclass instance.
        sample_size: Optional[int], number of samples to analyze.
            If None, uses all available samples.
            
    Returns:
        Dict[str, Any]: Quality assessment results containing:
            - 'sample_count': int, number of samples analyzed
            - 'missing_values': Dict, statistics about missing values
            - 'patient_distribution': Dict, statistics about patient distribution
            - 'visit_distribution': Dict, statistics about visit distribution (if applicable)
            - 'class_balance': Dict, statistics about class balance (if applicable)
            
    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(root='/path/to/mimic3')
        >>> dataset.set_task('mortality_prediction')
        >>> results = assess_dataset_quality(dataset)
        >>> print(results['missing_values'])
    """
    results = {
        'sample_count': 0,
        'missing_values': {},
        'patient_distribution': {},
    }
    
    # Check if dataset has samples
    if not hasattr(dataset, 'samples') or not dataset.samples:
        results['error'] = "Dataset has no samples. Call set_task() first."
        return results
    
    # Get samples to analyze
    samples = dataset.samples
    if sample_size is not None and sample_size < len(samples):
        import random
        samples = random.sample(samples, sample_size)
    
    results['sample_count'] = len(samples)
    
    # Analyze missing values
    missing_counts = {}
    for sample in samples:
        for key, value in sample.items():
            if key not in missing_counts:
                missing_counts[key] = 0
            
            # Check for missing values in different data types
            if value is None:
                missing_counts[key] += 1
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                # For lists/tuples, count elements that are None or NaN
                if isinstance(value[0], (list, tuple)):
                    # Nested lists
                    missing_counts[key] += sum(
                        1 for sublist in value if any(
                            v is None or (isinstance(v, float) and np.isnan(v)) 
                            for v in sublist
                        )
                    )
                else:
                    # Flat lists
                    missing_counts[key] += sum(
                        1 for v in value if v is None or (isinstance(v, float) and np.isnan(v))
                    )
    
    # Calculate missing value percentages
    results['missing_values'] = {
        key: {
            'count': count,
            'percentage': (count / len(samples)) * 100
        }
        for key, count in missing_counts.items() if count > 0
    }
    
    # Analyze patient distribution
    if hasattr(dataset, 'patient_to_index') and dataset.patient_to_index:
        patient_counts = {
            patient_id: len(indices) 
            for patient_id, indices in dataset.patient_to_index.items()
        }
        
        results['patient_distribution'] = {
            'total_patients': len(patient_counts),
            'min_samples_per_patient': min(patient_counts.values()),
            'max_samples_per_patient': max(patient_counts.values()),
            'avg_samples_per_patient': sum(patient_counts.values()) / len(patient_counts),
        }
    
    # Analyze visit distribution if available
    if hasattr(dataset, 'visit_to_index') and dataset.visit_to_index:
        visit_counts = {
            visit_id: len(indices) 
            for visit_id, indices in dataset.visit_to_index.items()
        }
        
        results['visit_distribution'] = {
            'total_visits': len(visit_counts),
            'min_samples_per_visit': min(visit_counts.values()),
            'max_samples_per_visit': max(visit_counts.values()),
            'avg_samples_per_visit': sum(visit_counts.values()) / len(visit_counts),
        }
    
    # Analyze class balance for classification tasks
    if all('y' in sample for sample in samples):
        from collections import Counter
        
        # Get labels
        labels = [sample['y'] for sample in samples]
        
        # Handle different label types
        if all(isinstance(label, (int, float, str, bool)) for label in labels):
            # Single label classification
            label_counts = Counter(labels)
            
            results['class_balance'] = {
                'label_counts': dict(label_counts),
                'label_percentages': {
                    label: (count / len(labels)) * 100
                    for label, count in label_counts.items()
                },
                'imbalance_ratio': max(label_counts.values()) / min(label_counts.values()) if label_counts else 0
            }
        elif all(isinstance(label, (list, tuple, np.ndarray)) for label in labels):
            # Multi-label or sequence classification
            # Analyze first to determine if binary or multi-class
            flat_labels = []
            for label in labels:
                if isinstance(label, np.ndarray):
                    label = label.tolist()
                flat_labels.extend(label)
            
            unique_values = set(flat_labels)
            if unique_values == {0, 1} or unique_values == {0.0, 1.0}:
                # Binary labels, likely multi-label classification
                # Count positives for each position
                if all(len(label) == len(labels[0]) for label in labels):
                    # Same length labels
                    label_array = np.array(labels)
                    pos_counts = label_array.sum(axis=0)
                    
                    results['class_balance'] = {
                        'task_type': 'multi_label',
                        'label_dimensions': len(labels[0]),
                        'positive_counts': pos_counts.tolist(),
                        'positive_percentages': ((pos_counts / len(labels)) * 100).tolist(),
                    }
    
    return results


def analyze_temporal_dataset(dataset, time_col: str = 'timestamp') -> Dict[str, Any]:
    """Analyzes temporal characteristics of a PyHealth dataset.
    
    This function examines temporal patterns in the dataset such as
    time spans, visit frequencies, and temporal gaps.
    
    Args:
        dataset: A PyHealth BaseDataset subclass instance.
        time_col: str, name of the timestamp column in the dataset.
            
    Returns:
        Dict[str, Any]: Temporal analysis results containing:
            - 'time_span': Dict, overall time span of the dataset
            - 'patient_time_spans': Dict, time spans per patient
            - 'visit_intervals': Dict, statistics about intervals between visits
            - 'temporal_density': Dict, density of events over time
            
    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> dataset = MIMIC3Dataset(root='/path/to/mimic3')
        >>> results = analyze_temporal_dataset(dataset)
        >>> print(results['time_span'])
    """
    results = {
        'has_temporal_data': False,
    }
    
    # Check if dataset has global_event_df
    if not hasattr(dataset, 'global_event_df'):
        results['error'] = "Dataset does not have global_event_df attribute"
        return results
    
    try:
        # Check if time column exists
        df = dataset.collected_global_event_df
        
        if time_col not in df.columns:
            # Try to find alternative time columns
            time_columns = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
            if time_columns:
                time_col = time_columns[0]
                logger.info(f"Using {time_col} as timestamp column")
            else:
                results['error'] = f"No timestamp column found. Expected '{time_col}'"
                return results
        
        results['has_temporal_data'] = True
        
        # Convert to pandas for easier datetime handling
        pdf = df.to_pandas()
        
        # Try to convert timestamp column to datetime
        try:
            if not pd.api.types.is_datetime64_any_dtype(pdf[time_col]):
                pdf[time_col] = pd.to_datetime(pdf[time_col], errors='coerce')
        except Exception as e:
            results['timestamp_conversion_error'] = str(e)
            return results
        
        # Overall time span
        min_time = pdf[time_col].min()
        max_time = pdf[time_col].max()
        
        if pd.notna(min_time) and pd.notna(max_time):
            time_span = max_time - min_time
            results['time_span'] = {
                'start': min_time.isoformat(),
                'end': max_time.isoformat(),
                'days': time_span.days,
                'total_seconds': time_span.total_seconds(),
            }
        
        # Patient-level time spans
        if 'patient_id' in pdf.columns:
            patient_spans = {}
            for patient_id, group in pdf.groupby('patient_id'):
                if pd.notna(group[time_col]).any():
                    p_min_time = group[time_col].min()
                    p_max_time = group[time_col].max()
                    if pd.notna(p_min_time) and pd.notna(p_max_time):
                        p_time_span = p_max_time - p_min_time
                        patient_spans[patient_id] = {
                            'days': p_time_span.days,
                            'total_seconds': p_time_span.total_seconds(),
                        }
            
            if patient_spans:
                span_days = [span['days'] for span in patient_spans.values()]
                results['patient_time_spans'] = {
                    'min_days': min(span_days),
                    'max_days': max(span_days),
                    'avg_days': sum(span_days) / len(span_days),
                    'median_days': sorted(span_days)[len(span_days) // 2],
                }
        
        # Visit intervals if visit_id exists
        if 'visit_id' in pdf.columns:
            # Group by patient and sort by timestamp
            visit_intervals = []
            
            for patient_id, patient_group in pdf.groupby('patient_id'):
                # Get unique visits with their timestamps
                visit_times = patient_group.groupby('visit_id')[time_col].min().sort_values()
                
                if len(visit_times) > 1:
                    # Calculate intervals between consecutive visits
                    intervals = visit_times.diff().dropna()
                    visit_intervals.extend([interval.total_seconds() / (24 * 3600) for interval in intervals])
            
            if visit_intervals:
                results['visit_intervals'] = {
                    'min_days': min(visit_intervals),
                    'max_days': max(visit_intervals),
                    'avg_days': sum(visit_intervals) / len(visit_intervals),
                    'median_days': sorted(visit_intervals)[len(visit_intervals) // 2],
                    'count': len(visit_intervals),
                }
        
        return results
        
    except Exception as e:
        results['error'] = str(e)
        return results


if __name__ == "__main__":
    print(list_nested_levels([1, 2, 3]))
    print(list_nested_levels([1, [2], 3]))
    print(is_homo_list([1, 2, 3]))
    print(is_homo_list([1, 2, "3"]))
