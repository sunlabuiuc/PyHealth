# Claude Code Instructions: TCGA RNA-seq Dataset + Task
# Person 1 — BulkRNABert PyHealth Contribution

---

## Your role and context

You are helping implement a PyHealth contribution for a Deep Learning for Healthcare
course final project at UIUC. The project reproduces BulkRNABert (Gélard et al., 2024),
a BERT-style transformer pretrained on bulk RNA-seq data for cancer type classification
and survival prediction. See the attached paper PDF for full details.

We are working in a group development repo (not a PyHealth fork). The files you
write here will be copied into a fork of https://github.com/sunlabuiuc/PyHealth before
PR submission. You must still match PyHealth's patterns exactly — see the reference
files provided.

The final PR will target: https://github.com/sunlabuiuc/PyHealth (main branch)

---

## What you are building

Person 1 is responsible for these files:

| Action   | File                                                                    |
|----------|-------------------------------------------------------------------------|
| CREATE   | `pyhealth/datasets/tcga_rnaseq.py`                                      |
| CREATE   | `pyhealth/datasets/configs/tcga_rnaseq.yaml`                            |
| CREATE   | `pyhealth/tasks/tcga_rnaseq_classification.py`                          |
| CREATE   | `tests/test_tcga_rnaseq.py`                                             |
| CREATE   | `docs/api/datasets/pyhealth.datasets.tcga_rnaseq.rst`                   |
| CREATE   | `docs/api/tasks/pyhealth.tasks.tcga_rnaseq_classification.rst`          |
| MODIFY   | `docs/api/datasets.rst` (add entry alphabetically to toctree)           |
| MODIFY   | `docs/api/tasks.rst` (add entry alphabetically to toctree)              |
| MODIFY   | `pyhealth/datasets/__init__.py` (export TCGARNASeqDataset)              |
| MODIFY   | `pyhealth/tasks/__init__.py` (export TCGARNASeqCancerTypeClassification)|

---

## Reference files provided — read these before writing anything

The following files are attached. Read every single one before writing a line of code.
They are the ground truth for how PyHealth expects things to be structured.

- `base_task.py` — the abstract base class all tasks inherit from. Read this to
  understand what task_name, input_schema, output_schema, and __call__ mean.

- `cancer_survival.py` — the single most important reference. It is already a
  TCGA-based task, shows exactly how to call patient.get_events(), how to handle
  missing data, and how to structure the return dict. Model your task class directly
  after this file.

- `chestxray14.py` (the dataset file) — the closest analog to our dataset. It is a
  flat patient-level dataset (not time-series like MIMIC), does preprocessing before
  calling super().__init__, writes a processed CSV, and uses a yaml config. Follow
  this pattern exactly.

- `chestxray14_binary_classification.py` and `chestxray14_multilabel_classification.py`
  — show how a task class pairs with its dataset class.

- `bmd_hs_disease_classification.py` — another clean task example showing
  input_schema and output_schema with multiple fields.

- `covid19_cxr_classification.py` — simplest possible task, good minimal reference.

- `in_hospital_mortality_mimic4.py` — shows how to handle more complex patient event
  filtering if needed.

You also have the BulkRNABert paper PDF. Read Section 2 (Methods) and Section 3.1
(Datasets) to understand the preprocessing pipeline: log10(1+TPM),
max-normalization, 19,042 genes, TCGA cohort structure.

Before writing the yaml config, you must also read the actual PyHealth repo:
clone https://github.com/sunlabuiuc/PyHealth and read:
  - pyhealth/datasets/base_dataset.py (understand BaseDataset __init__ signature)
  - pyhealth/datasets/configs/ (read any existing yaml to understand the format)

---

## Understanding the paper's data pipeline

From Section 3.1 of the paper:
- TCGA: 11,274 tumor samples across 33 cancer cohorts
- All datasets filtered to 19,042 genes common across TCGA, GTEx, and ENCODE
- Preprocessing: apply log10(1 + TPM) then max-normalize per sample
- For classification: the 33 TCGA cohorts are the 33 class labels
- For survival: Cox proportional hazards, target is (survival_time, event) pairs

Our dataset class handles loading and normalization.
Our task class handles turning loaded patient data into ML-ready samples.

---

## Data format agreement with Person 3

Person 3 produces two preprocessed CSV files. Your dataset class reads these.

gene_expression.csv at os.path.join(root, "gene_expression.csv"):
  - patient_id: TCGA patient ID string, e.g. TCGA-A2-A0T2
  - sample_id: TCGA sample barcode (may differ for multi-sample patients)
  - cohort: one of the 33 TCGA cohort strings (BRCA, LUAD, etc.)
  - remaining ~19,042 columns: one per gene, raw TPM values (may or may not be
    pre-normalized — your dataset class must handle both cases)

clinical.csv at os.path.join(root, "clinical.csv"):
  - patient_id: TCGA patient ID
  - cohort: TCGA cohort string
  - survival_time: float, days from diagnosis to event or censoring
  - event: int, 1 = death occurred, 0 = censored

Your tests use synthetic versions of these files. Never use real TCGA data in tests.

---

## The 33 TCGA cohort label mapping

COHORT_TO_LABEL = {
    "ACC": 0,  "BLCA": 1,  "BRCA": 2,  "CESC": 3,  "CHOL": 4,
    "COAD": 5, "DLBC": 6,  "ESCA": 7,  "GBM": 8,   "HNSC": 9,
    "KICH": 10, "KIRC": 11, "KIRP": 12, "LAML": 13, "LGG": 14,
    "LIHC": 15, "LUAD": 16, "LUSC": 17, "MESO": 18, "OV": 19,
    "PAAD": 20, "PCPG": 21, "PRAD": 22, "READ": 23, "SARC": 24,
    "SKCM": 25, "STAD": 26, "TGCT": 27, "THCA": 28, "THYM": 29,
    "UCEC": 30, "UCS": 31,  "UVM": 32,
}

---

## Build order — do these in this exact sequence

### STEP 1: pyhealth/tasks/tcga_rnaseq_classification.py

Start here. Zero dependencies, validate immediately.

Class name: TCGARNASeqCancerTypeClassification
Inherits from: BaseTask (from .base_task)

Class-level attributes:
  task_name: str = "TCGARNASeqCancerTypeClassification"
  input_schema: Dict[str, str] = {"gene_expression": "tensor"}
  output_schema: Dict[str, str] = {"label": "multiclass"}
  COHORT_TO_LABEL: Dict[str, int] = { ... }  # the 33-cohort dict above

__call__(self, patient: Any) -> List[Dict[str, Any]]:
  - Call patient.get_events(event_type="rnaseq") to get events
  - Return [] if no events
  - For each event, read event.cohort and event.gene_expression
  - Skip any event whose cohort is not in COHORT_TO_LABEL
  - Return one dict per event:
    {
        "patient_id": patient.patient_id,
        "gene_expression": event.gene_expression,  # torch.FloatTensor shape (num_genes,)
        "label": self.COHORT_TO_LABEL[event.cohort],  # int 0-32
    }
  - A patient may have multiple RNA-seq samples — return one dict per sample

Primary pattern reference: cancer_survival.py — follow its __call__ structure,
missing data handling, and return format exactly.

Full Google-style docstrings required on class and all methods.

---

### STEP 2: pyhealth/datasets/tcga_rnaseq.py

Class name: TCGARNASeqDataset
Inherits from: BaseDataset (from .base_dataset)

__init__ signature:
  def __init__(self, root: str, config_path: Optional[str] = None, **kwargs) -> None:

__init__ body in this order:
  1. Set default config path if not provided:
       config_path = os.path.join(os.path.dirname(__file__), "configs", "tcga_rnaseq.yaml")
  2. Call self._verify_data(root)
  3. Call self._preprocess(root)
  4. Call super().__init__(root=root, tables=["rnaseq"], dataset_name="tcga_rnaseq",
                           config_path=config_path, **kwargs)

_verify_data(self, root: str) -> None:
  - Check gene_expression.csv exists, raise FileNotFoundError if not
  - Check clinical.csv exists, raise FileNotFoundError if not
  - Use clear, descriptive error messages

_preprocess(self, root: str) -> None:
  - If tcga_rnaseq_pyhealth.csv already exists in root, return early (caching)
  - Read gene_expression.csv with pandas
  - Read clinical.csv with pandas
  - Identify gene columns: all columns except patient_id, sample_id, cohort
  - Call self._normalize(df, gene_cols) to normalize if needed
  - Left-merge on patient_id with clinical data (keep all expression rows)
  - Write merged result to os.path.join(root, "tcga_rnaseq_pyhealth.csv")

_normalize(self, df: pd.DataFrame, gene_cols: List[str]) -> pd.DataFrame:
  - Check if data is already normalized: if df[gene_cols].max().max() <= 1.0, return df unchanged
  - Otherwise apply log10(1 + x) to all gene columns
  - Then max-normalize each row: divide each value by that row's maximum
    (replace row max of 0 with 1 to avoid division by zero)
  - Return normalized df

Primary pattern reference: chestxray14.py — _verify_data and _index_data before
super().__init__, writes a processed CSV, yaml picks it up. Same structure.

CRITICAL: Do NOT import from pyhealth.tasks inside this file. Circular import.

Full Google-style docstrings required on class and all methods.

---

### STEP 3: pyhealth/datasets/configs/tcga_rnaseq.yaml

Before writing this file, clone the real PyHealth repo and read:
  - pyhealth/datasets/base_dataset.py
  - pyhealth/datasets/configs/ (read multiple existing yamls)

The yaml defines a table named "rnaseq" that tells BaseDataset how to load
tcga_rnaseq_pyhealth.csv. It must specify:
  - The CSV filename: tcga_rnaseq_pyhealth.csv
  - patient_id as the patient identifier column
  - sample_id as the visit/event identifier
  - cohort, survival_time, event, and gene columns as event attributes

Use chestxray14.yaml as your primary template since it is also a flat per-sample
CSV (not time-series). Match its structure exactly.

---

### STEP 4: tests/test_tcga_rnaseq.py

Hard rules:
  - NO real data anywhere in this file
  - All tests complete in under 1 second total
  - Use tempfile.TemporaryDirectory() for all file I/O
  - Maximum 3-5 fake patients
  - Use 10 fake genes not 19042

Synthetic data helper to put at the top of the test file:

  import os, tempfile
  import numpy as np
  import pandas as pd

  NUM_FAKE_GENES = 10
  FAKE_GENE_COLS = [f"GENE_{i:04d}" for i in range(NUM_FAKE_GENES)]
  FAKE_COHORTS = ["BRCA", "LUAD", "GBM"]

  def _make_synthetic_csvs(tmpdir: str, n_patients: int = 3) -> None:
      rng = np.random.default_rng(42)
      rows = []
      for i in range(n_patients):
          pid = f"TCGA-TEST-{i:04d}"
          cohort = FAKE_COHORTS[i % len(FAKE_COHORTS)]
          gene_vals = rng.uniform(1.0, 200.0, NUM_FAKE_GENES).tolist()
          rows.append([pid, f"{pid}-01", cohort] + gene_vals)
      ge_df = pd.DataFrame(rows,
          columns=["patient_id", "sample_id", "cohort"] + FAKE_GENE_COLS)
      ge_df.to_csv(os.path.join(tmpdir, "gene_expression.csv"), index=False)

      clin_rows = []
      for i in range(n_patients):
          pid = f"TCGA-TEST-{i:04d}"
          clin_rows.append([pid, FAKE_COHORTS[i % len(FAKE_COHORTS)],
              float(rng.uniform(100, 3000)), int(rng.integers(0, 2))])
      clin_df = pd.DataFrame(clin_rows,
          columns=["patient_id", "cohort", "survival_time", "event"])
      clin_df.to_csv(os.path.join(tmpdir, "clinical.csv"), index=False)

Tests to write:

  test_dataset_loads
    Create synthetic CSVs, instantiate TCGARNASeqDataset(root=tmpdir),
    assert no exception, assert len(dataset.patients) == n_patients

  test_processed_csv_written
    After instantiation, assert tcga_rnaseq_pyhealth.csv exists in tmpdir

  test_normalization_applied
    Read tcga_rnaseq_pyhealth.csv after load, assert all gene values in [0.0, 1.0]

  test_missing_gene_expression_raises
    Only create clinical.csv, assert FileNotFoundError on dataset instantiation

  test_missing_clinical_raises
    Only create gene_expression.csv, assert FileNotFoundError on dataset instantiation

  test_task_output_format
    Load dataset, call dataset.set_task(TCGARNASeqCancerTypeClassification()),
    for each sample assert keys patient_id, gene_expression, label exist,
    assert label is int in 0-32, assert gene_expression is torch.FloatTensor

  test_task_gene_expression_shape
    Assert each sample's gene_expression.shape == (NUM_FAKE_GENES,)

  test_invalid_cohort_skipped
    Add a row with cohort="INVALID_COHORT" to gene_expression.csv,
    assert no sample in task output has label outside 0-32

  test_multi_sample_patient
    Add two rows with the same patient_id but different sample_ids,
    assert task output contains two samples for that patient_id

  test_task_empty_for_no_events
    Directly instantiate TCGARNASeqCancerTypeClassification(),
    create a mock patient whose get_events returns [],
    assert task(mock_patient) == []

---

### STEP 5: RST documentation files

docs/api/datasets/pyhealth.datasets.tcga_rnaseq.rst:

  pyhealth.datasets.tcga\_rnaseq
  ==============================

  .. automodule:: pyhealth.datasets.tcga_rnaseq
     :members:
     :undoc-members:
     :show-inheritance:

docs/api/tasks/pyhealth.tasks.tcga_rnaseq_classification.rst:

  pyhealth.tasks.tcga\_rnaseq\_classification
  ============================================

  .. automodule:: pyhealth.tasks.tcga_rnaseq_classification
     :members:
     :undoc-members:
     :show-inheritance:

---

### STEP 6: Index file updates

In docs/api/datasets.rst, add to the toctree in alphabetical order:
  pyhealth.datasets.tcga_rnaseq

In docs/api/tasks.rst, add to the toctree in alphabetical order:
  pyhealth.tasks.tcga_rnaseq_classification

Read both files before editing to match the exact existing format.

---

### STEP 7: __init__.py exports

In pyhealth/datasets/__init__.py add:
  from .tcga_rnaseq import TCGARNASeqDataset

In pyhealth/tasks/__init__.py add:
  from .tcga_rnaseq_classification import TCGARNASeqCancerTypeClassification

Match the import style of existing entries in each file.

---

## Code style — graded

- PEP8, 88-character line length
- snake_case for variables and functions, PascalCase for classes
- Type hints on every argument and return value
- Google-style docstrings on every class and method

Google-style docstring format:
  def method(self, arg1: str, arg2: Optional[int] = None) -> List[Dict]:
      """One-line summary.

      Longer explanation if needed.

      Args:
          arg1: Description.
          arg2: Description. Defaults to None.

      Returns:
          Description of return value and structure.

      Raises:
          FileNotFoundError: When data file does not exist.

      Examples:
          >>> obj = MyClass(root="/path/to/data")
          >>> result = obj.method("test")
          >>> len(result)
          1
      """

Run black when done:
  black pyhealth/datasets/tcga_rnaseq.py \
        pyhealth/tasks/tcga_rnaseq_classification.py \
        tests/test_tcga_rnaseq.py

---

## Verification commands

  # Task imports
  python -c "from pyhealth.tasks.tcga_rnaseq_classification import \
      TCGARNASeqCancerTypeClassification; print('OK')"

  # Dataset imports
  python -c "from pyhealth.datasets.tcga_rnaseq import \
      TCGARNASeqDataset; print('OK')"

  # Tests — must all pass and be fast
  pytest tests/test_tcga_rnaseq.py -v

  # Style
  black --check pyhealth/datasets/tcga_rnaseq.py \
                pyhealth/tasks/tcga_rnaseq_classification.py \
                tests/test_tcga_rnaseq.py

  # YAML valid
  python -c "import yaml; yaml.safe_load(open(\
      'pyhealth/datasets/configs/tcga_rnaseq.yaml')); print('YAML OK')"

---

## What NOT to do

- Do NOT use real TCGA data in tests — 5 point deduction
- Do NOT make tests slow (over 1 second) — 3 point deduction
- Do NOT skip docstrings — explicitly graded
- Do NOT import pyhealth.tasks inside pyhealth/datasets/tcga_rnaseq.py — circular import
- Do NOT write either file before reading the reference files — you will get it wrong
- Do NOT hardcode 19042 as gene count in tests — use NUM_FAKE_GENES

---

## Common bugs

Circular import: Never import the task class from inside the dataset class.

BaseDataset signature: Read base_dataset.py before calling super().__init__.
Use the exact parameter names it expects.

YAML whitespace: YAML is whitespace-sensitive. Verify with:
  python -c "import yaml; yaml.safe_load(open('tcga_rnaseq.yaml'))"

Missing __init__.py exports: PR reviewers test with
  from pyhealth.datasets import TCGARNASeqDataset
If you skip this step it fails immediately.

Multiple samples per patient: Return one dict per sample in __call__, not one
per patient. Patients in TCGA can have multiple RNA-seq samples.

---

## Paper citation for PR description

Paper: Gélard et al. (2024). BulkRNABert: Cancer prognosis from bulk RNA-seq
based language models.
DOI: https://doi.org/10.1101/2024.06.18.599483
Code: https://github.com/instadeepai/multiomics-open-research
