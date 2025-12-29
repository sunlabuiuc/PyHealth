import hashlib
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
import litdata
from dateutil.parser import parse as dateutil_parse
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from pyhealth import BASE_CACHE_PATH
from pyhealth.utils import create_directory

MODULE_CACHE_PATH = os.path.join(BASE_CACHE_PATH, "datasets")
create_directory(MODULE_CACHE_PATH)


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


def padyear(year: str, month="1", day="1") -> str:
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


def _is_time_value_tuple(
    value: Any,
    allow_additional_components: bool = True,
) -> bool:
    """Detects StageNet-style temporal tuples.

    Args:
        value: Candidate value to inspect.
        allow_additional_components: Whether tuples with >2 elements should be
            considered valid. This accommodates future extensions where extra
            metadata (e.g., feature types) may be appended.

    Returns:
        bool: True if the value looks like a temporal tuple, False otherwise.
    """

    if not isinstance(value, tuple) or len(value) < 2:
        return False

    time_component, value_component = value[0], value[1]
    time_ok = time_component is None or isinstance(time_component, list)
    values_ok = isinstance(value_component, list)

    if not (time_ok and values_ok):
        return False

    if not allow_additional_components and len(value) > 2:
        return False

    return True


def _convert_for_cache(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Serializes temporal tuples for parquet/pickle friendly caching.

    The conversion stays backwards compatible with older cache structures while
    capturing any extra tuple components that may be introduced later on.

    Args:
        sample: Dictionary representing a single data sample.

    Returns:
        Dict[str, Any]: A new dictionary with temporal tuples replaced by
            cache-friendly dictionaries containing metadata and raw components.
    """

    converted: Dict[str, Any] = {}
    for key, value in sample.items():
        if _is_time_value_tuple(value):
            tuple_components = list(value)

            cached_representation: Dict[str, Any] = {
                "__stagenet_cache__": True,
                "time": tuple_components[0],
                "values": tuple_components[1],
            }

            extras = tuple_components[2:]
            if extras:
                cached_representation["extras"] = list(extras)
            cached_representation["components"] = tuple_components

            converted[key] = cached_representation
        else:
            converted[key] = value

    return converted


def _restore_from_cache(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Restore temporal tuples previously serialized for caching."""

    restored: Dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, dict) and value.get("__stagenet_cache__"):
            if "components" in value and isinstance(value["components"], list):
                components = list(value["components"])
            elif "components" in value and isinstance(value["components"], tuple):
                components = list(value["components"])
            else:
                components = [value.get("time"), value.get("values")]
                extras = value.get("extras")
                if isinstance(extras, list):
                    components.extend(extras)
                elif extras is not None:
                    components.append(extras)

            restored[key] = tuple(components)
        else:
            restored[key] = value

    return restored


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
        Tuples of (time, values) from temporal processors are collated separately.
    """
    collated = {}
    keys = batch[0].keys()

    for key in keys:
        values = [sample[key] for sample in batch]

        # Check if this is a temporal feature tuple (time, values)
        if isinstance(values[0], tuple) and len(values[0]) == 2:
            # Handle (time, values) tuples from processors
            time_tensors = [v[0] for v in values]
            value_tensors = [v[1] for v in values]

            # Collate values
            if value_tensors[0].dim() == 0:
                # Scalars
                collated_values = torch.stack(value_tensors)
            elif all(v.shape == value_tensors[0].shape for v in value_tensors):
                # All same shape
                collated_values = torch.stack(value_tensors)
            else:
                # Variable shapes, use pad_sequence
                collated_values = pad_sequence(
                    value_tensors, batch_first=True, padding_value=0
                )

            # Collate times (if present)
            collated_times = None
            # Check if ALL samples have time (not just some)
            if all(t is not None for t in time_tensors):
                time_tensors_all = [t for t in time_tensors if t is not None]
                if all(t.shape == time_tensors_all[0].shape for t in time_tensors_all):
                    collated_times = torch.stack(time_tensors_all)
                else:
                    collated_times = pad_sequence(
                        time_tensors_all, batch_first=True, padding_value=0
                    )

            # Return as tuple (time, values)
            collated[key] = (collated_times, collated_values)

        elif isinstance(values[0], torch.Tensor):
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
                    collated[key] = pad_sequence(
                        values, batch_first=True, padding_value=0
                    )
                else:
                    raise ValueError(f"Unsupported tensor shape: {values[0].shape}")
        else:
            # Non-tensor data: keep as list
            collated[key] = values

    return collated


def get_dataloader(
    dataset: litdata.StreamingDataset, batch_size: int, shuffle: bool = False
) -> DataLoader:
    """Creates a DataLoader for a given dataset.

    Args:
        dataset: The dataset to load data from.
        batch_size: The number of samples per batch.
        shuffle: Whether to shuffle the data at every epoch.

    Returns:
        A DataLoader instance for the dataset.
    """
    dataset.set_shuffle(shuffle)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_dict_with_padding,
    )

    return dataloader


def save_processors(sample_dataset, output_dir: str) -> Dict[str, str]:
    """Save input and output processors to pickle files.

    This function saves the fitted processors from a SampleDataset to disk,
    allowing them to be reused in future runs for consistent feature encoding.

    Args:
        sample_dataset: SampleDataset with fitted processors
        output_dir (str): Directory to save processor files

    Returns:
        Dict[str, str]: Paths where processors were saved with keys
            'input_processors' and 'output_processors'

    Example:
        >>> from pyhealth.datasets import save_processors
        >>> sample_dataset = base_dataset.set_task(task)
        >>> paths = save_processors(sample_dataset, "./output/processors")
        >>> print(paths["input_processors"])
        ./output/processors/input_processors.pkl
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save input processors
    input_processors_path = output_path / "input_processors.pkl"
    with open(input_processors_path, "wb") as f:
        pickle.dump(sample_dataset.input_processors, f)
    paths["input_processors"] = str(input_processors_path)
    print(f"✓ Saved input processors to {input_processors_path}")

    # Save output processors
    output_processors_path = output_path / "output_processors.pkl"
    with open(output_processors_path, "wb") as f:
        pickle.dump(sample_dataset.output_processors, f)
    paths["output_processors"] = str(output_processors_path)
    print(f"✓ Saved output processors to {output_processors_path}")

    return paths


def load_processors(processor_dir: str) -> Tuple[Dict, Dict]:
    """Load input and output processors from pickle files.

    This function loads previously saved processors from disk, allowing
    consistent feature encoding across different runs without refitting.

    Args:
        processor_dir (str): Directory containing processor pickle files

    Returns:
        Tuple[Dict, Dict]: (input_processors, output_processors)

    Raises:
        FileNotFoundError: If processor files are not found

    Example:
        >>> from pyhealth.datasets import load_processors
        >>> input_procs, output_procs = load_processors("./output/processors")
        >>> sample_dataset = base_dataset.set_task(
        ...     task,
        ...     input_processors=input_procs,
        ...     output_processors=output_procs
        ... )
    """
    from pathlib import Path

    processor_path = Path(processor_dir)

    input_processors_path = processor_path / "input_processors.pkl"
    output_processors_path = processor_path / "output_processors.pkl"

    if not input_processors_path.exists():
        raise FileNotFoundError(
            f"Input processors not found at {input_processors_path}"
        )
    if not output_processors_path.exists():
        raise FileNotFoundError(
            f"Output processors not found at {output_processors_path}"
        )

    with open(input_processors_path, "rb") as f:
        input_processors = pickle.load(f)
    print(f"✓ Loaded input processors from {input_processors_path}")

    with open(output_processors_path, "rb") as f:
        output_processors = pickle.load(f)
    print(f"✓ Loaded output processors from {output_processors_path}")

    return input_processors, output_processors


if __name__ == "__main__":
    print(list_nested_levels([1, 2, 3]))
    print(list_nested_levels([1, [2], 3]))
    print(list_nested_levels([[1, [2], [[3]]]]))
    print(is_homo_list([1, 2, 3]))
    print(is_homo_list([1, 2, [3]]))
    print(is_homo_list([1, 2.0]))
