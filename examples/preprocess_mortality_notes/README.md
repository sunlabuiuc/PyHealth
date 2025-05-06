# MIMIC-III In-Hospital Mortality Prediction: Note Preprocessing Example

This directory contains an example script (`preprocess_mortality_notes.py`) demonstrating how to preprocess MIMIC-III clinical notes data for the task of in-hospital mortality prediction, using the `pyhealth` library.

## Purpose

The raw MIMIC-III dataset distributes information across multiple tables (e.g., `PATIENTS`, `ADMISSIONS`, `NOTEEVENTS`). Many machine learning models, especially those processing sequential data like clinical notes (e.g., FTL-Trans, LSTMs, Transformers), require input data to be structured in a specific format, often grouping notes by admission and associating them with patient identifiers and task-specific labels.

This script bridges the gap between the raw MIMIC-III tables, loaded efficiently using `pyhealth.datasets.MIMIC3Dataset`, and a structured, task-ready format. It performs the following steps:

1.  **Loads Data:** Uses `pyhealth.datasets.MIMIC3Dataset` to load `PATIENTS`, `ADMISSIONS`, and `NOTEEVENTS` tables.
2.  **Joins Information:** Combines patient (`SUBJECT_ID`) and admission (`HADM_ID`) information with clinical notes (`TEXT`) and the in-hospital mortality label (`HOSPITAL_EXPIRE_FLAG`).
3.  **Cleans Text:** Performs basic cleaning on the clinical note text, including removing de-identification placeholders (`[**...**]`), normalizing whitespace, handling basic punctuation, and converting to lowercase.
4.  **Handles Timestamps:** Uses the `charttime` provided by `NOTEEVENTS` (with missing times imputed by `MIMIC3Dataset` using `chartdate`).
5.  **Structures Output:** Groups the processed notes by admission (`visit_id`) and sorts them chronologically.
6.  **Saves Data:** Outputs the data into a JSON Lines (`.jsonl`) file, where each line represents a single hospital admission.

This script was inspired by the preprocessing requirements of models like FTL-Trans but provides a general pipeline for preparing MIMIC-III notes for sequence-based mortality prediction.

## Output Format

The script generates a `.jsonl` file where each line is a JSON object corresponding to one hospital admission (`visit_id`). The structure of each JSON object is:

```json
{
  "patient_id": <int>,          // SUBJECT_ID from MIMIC-III
  "visit_id": <int>,            // HADM_ID from MIMIC-III
  "notes_sequence": [           // List of notes, sorted chronologically
    [<str_iso_timestamp | null>, <str_cleaned_text>],
    [<str_iso_timestamp | null>, <str_cleaned_text>],
    ...
  ],
  "label": <int>                // 0 or 1 (HOSPITAL_EXPIRE_FLAG)
}
```

notes_sequence: A list containing tuples for each note within the admission. Each tuple contains the note's timestamp (as an ISO 8601 formatted string, or null if unavailable) and the cleaned note text.


## Usage

1. Prerequisites:
- Python 3.8+
- pyhealth: Install via pip install pyhealth
- Access to MIMIC-III Clinical Database v1.4 (Requires credentialed access from PhysioNet). Ensure the .csv.gz files are available locally.

2. Run the script:

```bash
python preprocess_mortality_notes.py \
    --mimic_root /path/to/your/mimic-iii-clinical-database-1.4 \
    --output_file ./processed_mimic_notes_mortality.jsonl
```

Replace /path/to/your/mimic-iii-clinical-database-1.4 with the actual path to your MIMIC-III dataset files and ./processed_mimic_notes_mortality.jsonl with your desired output file path.

## Next Steps

The generated .jsonl file can be used as input for creating a pyhealth.datasets.SampleDataset, which can then be used to train and evaluate various models within the PyHealth framework for the task of in-hospital mortality prediction based on clinical notes. You would typically parse this JSONL file in the load_data method of a custom dataset class inheriting from SampleDataset.
