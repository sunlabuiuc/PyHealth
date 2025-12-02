# MIMIC-CXR Reports Dataset (PyHealth Extension)

This document provides implementation details and usage instructions for the MIMIC-CXR Radiology Reports dataset added to PyHealth as part of this contribution.
The dataset supports loading, parsing, and accessing structured text reports from the MIMIC-CXR dataset, enabling downstream modeling.

------------------------------------------------------------------------

## Files Introduced

pyhealth/
  ├── datasets/
        ├── mimic_cxr_reports.py                   # Dataset implementation
        ├── configs/
        │       └── mimic_cxr_reports.yaml         # Dataset configuration (Pydantic validated)
        ├── __init__.py                            # Updated the Dataset class relative import
  ├── tests/
        ├── test_mimic_cxr_reports.py              # Test script for dataset loader
  ├── docs/
        ├── README_mimic_cxr_reports.md            # Documentation


------------------------------------------------------------------------

## Contribution Overview

This contribution adds end-to-end support for:

-   Parsing **MIMIC-CXR radiology reports** stored as plain text
    (`.txt`) files.
-   A new dataset class:
    -   **`MIMICCXRReportsDataset`**
-   A new dataset configuration file:
    -   **`mimic_cxr_reports.yaml`**
-   Automatic directory traversal for:
    -   patient folders (`pXX`)
    -   study subfolders (`sXXXXXX`)
-   Structured sample or dataset generation for downstream NLP tasks.

------------------------------------------------------------------------

## Dataset Overview

This dataset extracts structured information from MIMIC-CXR report text files, including:

patient_id : Patient directory identifier
study_id : Study directory identifier
report_text : Full text of the report
findings : Extracted FINDINGS section
impression : Extracted IMPRESSION section
path : Full filesystem path to the report

The dataset is:
	•	Environment-agnostic
	•	Non-destructive (read-only)
	•	Fully integrated with PyHealth’s BaseDataset interface
	•	Supports dev mode with sample limiting

------------------------------------------------------------------------

## Directory Structure Expected

Your MIMIC-CXR dataset must follow this structure:

    mimic-cxr-reports/
     ├── files/
     │   ├── p10/
     │   │    ├── p10XXXXXX/
     │   │    │     ├── sXXXXXXXX.txt
     │   │    │     ├── other sXXXXXXXX files...
     │   ├── p11/
     │   │    ├── p11XXXXXX/
     │   │    │     ├── sXXXXXXXX.txt
     │   │    │     ├── other sXXXXXXXX files...

Ensure you have downloaded the dataset (mimic-cxr-reports.zip file) from the following URL to your desired location: 

https://physionet.org/content/mimic-cxr/2.1.0/mimic-cxr-reports.zip


------------------------------------------------------------------------

## Implementation Summary

This class: `MIMICCXRReportsDataset`

    •   Resolves the dataset YAML config portably
	•	Automatically extracts mimic-cxr-reports.zip if needed
	•	Collects patient/study-level .txt reports
	•	Extracts FINDINGS and IMPRESSION sections via regex
	•	Stores samples in structured dictionaries
	•	Provides the standard PyHealth API:

------------------------------------------------------------------------

## Example Usage

``` python
from pyhealth.datasets import MIMICCXRReportsDataset

root_path = "/path/to/mimic-cxr/zipfile" # change the location to point to the path where the mimic-cxr-reports.zip file is placed

ds = MIMICCXRReportsDataset(
    root=root_path,
    patients=["p10", "p11"],
    dev_mode=True,
    limit=5,
)

samples = ds.get_samples()
print(samples[0])
```

------------------------------------------------------------------------

## Development Installation

From the root of your fork, run the following command:

``` pip install -e . ```

This enables live editing without reinstalling.

To test the sample dataset loader in development mode:

1. Update the **root_path** variable in the **test_mimic_cxr_reports.py** file to point to the path where the mimic-cxr-reports.zip file is placed
2. Run the following command: ``` python tests/test_mimic_cxr_reports.py ```

------------------------------------------------------------------------

## License Note

This dataset wrapper assumes users already have approved & credentialed access to MIMIC-CXR from PhysioNet.
No data files are redistributed by this code.
