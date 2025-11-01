"""Utility functions for tasks and processors."""

import pickle
from pathlib import Path
from typing import Dict, Tuple


def save_processors(sample_dataset, output_dir: str) -> Dict[str, str]:
    """Save input and output processors to pickle files.

    Args:
        sample_dataset: SampleDataset with fitted processors
        output_dir (str): Directory to save processor files

    Returns:
        Dict[str, str]: Paths where processors were saved

    Example:
        >>> from pyhealth.tasks.utils import save_processors
        >>> sample_dataset = base_dataset.set_task(task)
        >>> paths = save_processors(sample_dataset, "./output/processors")
        >>> print(paths["input_processors"])
        ./output/processors/input_processors.pkl
    """
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

    Args:
        processor_dir (str): Directory containing processor pickle files

    Returns:
        Tuple[Dict, Dict]: (input_processors, output_processors)

    Raises:
        FileNotFoundError: If processor files are not found

    Example:
        >>> from pyhealth.tasks.utils import load_processors
        >>> input_procs, output_procs = load_processors("./output/processors")
        >>> sample_dataset = base_dataset.set_task(
        ...     task,
        ...     input_processors=input_procs,
        ...     output_processors=output_procs
        ... )
    """
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
