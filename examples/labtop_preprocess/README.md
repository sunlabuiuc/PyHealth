# Example: LabTOP-style Preprocessing for MIMIC-IV

This example focuses only on generating LabTOP-style token sequences and does not modify existing PyHealth datasets.It demonstrates how to preprocess **MIMIC-IV ICU data** into **LabTOP-style token sequences** using `Bio_ClinicalBERT` along with custom time and event-type tokens. It provides a clean, reproducible pipeline for transforming raw MIMIC-IV events into tokenized sequences suitable for clinical language models.

## Data Access & Download

MIMIC-IV is a restricted-access dataset. To download the data, follow these steps:

1.  **Register on PhysioNet:** Create an account at [https://physionet.org](https://physionet.org).
2.  **Complete CITI Training:** Complete the required "Data or Specimens Only Research" course on the CITI Program website and link your completion report to your PhysioNet profile.
3.  **Sign the DUA:** Request access to the [MIMIC-IV project page](https://physionet.org/content/mimiciv/) and sign the Data Use Agreement.
4.  **Download Data:** Once approved, you can download the data directly from the project page or using the `physionet-credentialed-users` Google Cloud bucket.

   > **Note:** Ensure you download **MIMIC-IV v2.0** or later.

## Required Datasets

The script requires the following MIMIC-IV (v2.0+) CSV files to be present in the `data_dir`:

- **Core:** `patients.csv.gz`, `admissions.csv.gz`, `icustays.csv.gz`
- **Events:** `labevents.csv.gz`, `inputevents.csv.gz`, `outputevents.csv.gz`, `procedureevents.csv.gz`
- **Medications:** `emar.csv.gz`, `emar_detail.csv.gz`
- **Dictionaries:** `d_labitems.csv.gz`, `d_items.csv.gz`

## Features & Logic

- **ICU Stay Filtering:**
  - Selects stays with valid `intime` and `outtime`.
  - Restricts events to the **first 72 hours** of the ICU stay.
- **Event Unification:**
  - **Lab Events:** Filters for the top 200 most frequent `itemid`s.
  - **Procedure Events:** Filters for the top 20 most frequent `itemid`s.
  - **Medications (EMAR):** Uses `emar` and `emar_detail` for administration and dosage info.
  - **Input/Output:** Includes all valid fluid inputs and outputs.
- **Time Tokenization:**
  - Absolute timestamps are converted to relative tokens from ICU admission: `[DAYd] [WEEKDAY] [HHh] [MMm]`.
  - Minutes are bucketed into 10-minute bins (0, 10, 20, 30, 40, 50).
- **Value Encoding:**
  - Numeric values are tokenized at the digit level (e.g., `1.23` → `1`, `.`, `2`, `3`) to improve numerical understanding by the LLM.
- **Custom Tokenizer:**
  - Extends `emilyalsentzer/Bio_ClinicalBERT` with special tokens for time, event types (`labevent`, `inputevent`, etc.), and `[EOE]` (End of Event).

## Usage
Minimal Usage:

```bash
python labtop_preprocess.py --data_dir /path/to/mimic --out_dir ./processed
```

Full Usage:

```bash
python labtop_preprocess.py \
  --data_dir /path/to/mimic_iv \
  --out_dir ./processed_data \
  --stay_limit 100 \
  --inspect
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_dir` | Required | Path to the directory containing MIMIC-IV CSV files. |
| `--out_dir` | `processed_data` | Directory where processed `.pkl` files and the tokenizer will be saved. |
| `--tokenizer_name` | `emilyalsentzer/Bio_ClinicalBERT` | HuggingFace model name to use as the base tokenizer. |
| `--max_len` | `1024` | Maximum sequence length for tokenized stays. |
| `--stay_limit` | `None` | Limit the number of stays to process (useful for testing/debugging). |
| `--shard_size` | `50000` | Number of sequences per temporary shard file. |
| `--inspect` | `False` | If set, prints raw textual representations of events for a few stays. |

## Output Structure

The script generates the following in `out_dir`:

1.  **`train.pkl`, `val.pkl`, `test.pkl`**:
    - Lists of dictionaries, each containing:
      - `stay_id`: The ICU stay identifier.
      - `input_ids`: Token IDs for the full sequence (demographics + events).
      - `type_ids`: Token type IDs (distinguishing context vs. target values).

2.  **`val_eval.pkl`, `test_eval.pkl`**:
    - Used for evaluating lab value prediction. Each item contains:
      - `prompt_ids`: Tokens up to the lab value (context).
      - `label_ids`: Tokens for the target lab value.
      - `valuenum`: Original numeric value.
      - `itemid`: Lab item identifier.

3.  **`tokenizer/`**:
    - A saved HuggingFace tokenizer directory containing the base tokenizer with all added special tokens.

## Requirements

- **MIMIC-IV Data:** Users must obtain MIMIC-IV data independently via PhysioNet credentialing.
- **Python Dependencies:** `pyhealth`, `transformers`, `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`.

## Academic Context

This project was developed as part of **CS598: Deep Learning for Healthcare** at UIUC. It aims to improve reproducibility and transparency in AI4H research by providing reusable components for the PyHealth ecosystem.
