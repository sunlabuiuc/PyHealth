# Example: Model Uncertainty Analysis with Custom Data Wrapper for PyHealth

This example demonstrates how custom preprocessed sequential data (e.g., for EHR analysis) can be integrated with the PyHealth library.

## Files Included:

1.  **`pyhealth_custom_dataset_wrapper.py`**:
    * Defines `CustomSequentialEHRDataPyHealth`, a Python class inheriting from `pyhealth.datasets.SampleEHRDataset`.
    * This class serves as a wrapper, taking lists of preprocessed sequence tensors and label tensors as input.
    * It structures this data into the `samples` format expected by `SampleEHRDataset` and uses a `task_fn` to prepare individual samples.
    * This demonstrates a method for making custom data formats compatible with PyHealth's data handling system.

2.  **`uncertainty_wrapper_example.ipynb`**:
    * A Jupyter notebook showcasing the usage of `CustomSequentialEHRDataPyHealth`.
    * It first generates minimal synthetic sequential data (for illustrative purposes).
    * It then instantiates the `CustomSequentialEHRDataPyHealth` wrapper with this data.
    * Finally, it demonstrates accessing a sample from the PyHealth-compatible dataset.

## Purpose

The primary goal is to provide a simple example of integrating external, preprocessed sequential data into the PyHealth ecosystem using `SampleEHRDataset`. This wrapped dataset could then potentially be used with other PyHealth functionalities or in custom analysis pipelines (like model uncertainty studies) while maintaining compatibility with PyHealth data structures.