"""
LLMSYN Validation: Random Forest TSTR Pipeline for Synthetic EHR Evaluation.

This script implements the validation methodology from:
  Hao et al., "LLMSYN: Generating Synthetic Electronic Health Records
  Without Patient-Level Data", MLHC 2024 (PMLR 252).
  https://proceedings.mlr.press/v252/hao24a.html

Key components:
  - Synthetic EHR generation via PyHealth's LLMSYNModel (4-step Markov pipeline)
  - Train-on-Synthetic, Test-on-Real (TSTR) evaluation with Random Forest
  - Mortality prediction (binary) + Phenotype prediction (multi-label, top-10 ICD-9)
  - Metrics: Accuracy, AUROC, F1, K-S statistic, MMD, k-anonymity

================================================================================
PREREQUISITES (full setup, ~10-20 min on a clean machine)
================================================================================

1) PYTHON VERSION
   PyHealth requires Python >=3.12, <3.14. Recommended: Python 3.13.
     macOS:   brew install python@3.13
     Linux:   sudo apt install python3.13 python3.13-venv
     Verify:  python3.13 --version   # -> Python 3.13.x

   Common gotcha (macOS): if `python --version` reports 3.14 even after
   creating a 3.13 venv, you likely have `alias python="..."` in ~/.zshrc.
   Fix in current shell:  unalias python
   Permanent fix:         sed -i '' 's/^alias python=/# alias python=/' ~/.zshrc

2) VIRTUAL ENVIRONMENT (use absolute path to avoid PATH conflicts)
     cd /path/to/PyHealth-master
     /opt/homebrew/bin/python3.13 -m venv examples/path/to/venv
     source examples/path/to/venv/bin/activate
     python --version   # MUST be 3.13.x

3) INSTALL PYHEALTH + ALL DEPENDENCIES (editable install pulls everything)
     pip install --upgrade pip
     pip install -e .
   This installs torch, transformers, polars, pydantic, dask, rdkit, etc.
   ~5-10 min download. Do NOT install packages individually.

4) MIMIC-III DEMO DATA (already in repo, no download needed)
   Located at: test-resources/core/mimic3demo/ (gz CSV files, 100 patients).
   For full MIMIC-III access (paper-quality numbers), credential at PhysioNet:
     https://physionet.org/content/mimiciii/1.4/   (requires CITI training)

5) LLMSYN STATS FILE (already in repo)
   Located at: test-resources/llmsyn/stats.json
   Pre-computed prior statistics from real MIMIC-III (mortality_rate=0.0993,
   100 top diagnosis codes, demographic distributions).

6) LLM PROVIDER (CHOOSE ONE)
   This script defaults to LLM_PROVIDER="mock" (no API call, no cost, but
   produces single-class outputs so AUROC stays at 0.5).

   For real paper-quality numbers, switch to a real LLM:

   Option A - Anthropic Claude (recommended):
     1. Sign up at https://console.anthropic.com/, add >=$5 credit.
     2. Create API key (Settings -> API Keys).
     3. export ANTHROPIC_API_KEY=sk-ant-...
     4. Set LLM_PROVIDER = "claude" below.

   Option B - OpenAI:
     1. Sign up at https://platform.openai.com/, add credit.
     2. Create API key at https://platform.openai.com/api-keys
     3. export OPENAI_API_KEY=sk-...
     4. Set LLM_PROVIDER = "openai" below.

   Option C - Mock backend (default, no cost, no API call):
     Set LLM_PROVIDER = "mock" below. Uses PyHealth's built-in
     _MockLLMBackend (pyhealth/models/llmsyn.py), which exercises the full
     4-step pipeline (prompts, parser, RAG hooks, schema validation) but
     returns deterministic placeholder strings instead of calling an LLM.
     Useful for: smoke-testing the pipeline, CI, debugging code paths.
     Caveat: mock outputs are single-class (Survived="Yes" always), so
     downstream AUROC will be 0.5 by construction. Don't use mock numbers
     for any paper / report.

   COST ESTIMATE: each synthetic record requires 4 LLM calls (4-step Markov
   chain: demographics -> main_dx -> complications -> procedures). Plus RAG
   lookups if enable_rag=True. Rough cost (Claude Haiku):
       N=5    -> ~$0.10   (smoke test)
       N=100  -> ~$2-5    (paper-quality experiment)

================================================================================
USAGE
================================================================================

   cd examples
   python llmsyn_validation.py

Outputs to stdout: dataset stats, per-task metrics, final results dict.

================================================================================
CONFIGURATION (edit constants below)
================================================================================

   N_SYNTHETIC      # how many synthetic records to generate (5 for smoke test, 100+ for paper)
   USE_REAL_DATA    # True -> load real MIMIC-III demo; False -> use mock data
   USE_REAL_LLMSYN  # True -> call LLMSYNModel; False -> use built-in mock generator
   LLM_PROVIDER     # "mock" | "claude" | "openai"

================================================================================
GRACEFUL FALLBACKS (script will not crash if any of these fail)
================================================================================

   - PyHealth import fails              -> use _load_mock_dataset()
   - MIMIC-III demo files missing       -> use _load_mock_dataset()
   - LLMSYN imports fail                -> use _generate_mock_synthetic()
   - API key missing for non-mock LLM   -> use _generate_mock_synthetic()
   - LLMSYNModel.generate() raises      -> use _generate_mock_synthetic()
   - Single-class label edge cases      -> AUROC/F1 reported as NaN (no crash)

================================================================================
FILE LAYOUT
================================================================================

   load_real_dataset()              # MIMIC-III demo loader (PyHealth API)
   _load_mock_dataset()             # numpy-only fallback for real data
   generate_synthetic_data()        # entry point: real LLMSYN or mock fallback
   _generate_mock_synthetic()       # numpy-only fallback for synthetic data
   train_random_forest()            # Mortality task: RF on synthetic
   evaluate_on_real_data()          # Mortality task: TSTR + ACC/AUROC/F1/K-S
   compute_additional_metrics()     # MMD + k-anonymity
   train_phenotype_rf()             # Phenotype task: one RF per top-10 ICD-9
   evaluate_phenotype_on_real()     # Phenotype task: macro ACC/AUROC/F1
   main()                           # orchestrates the 7-step pipeline

================================================================================
"""


import json
import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve

# Allow running from examples/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try to import PyHealth; fall back to mock data if unavailable
try:
    from pyhealth.datasets import MIMIC3Dataset
    PYHEALTH_AVAILABLE = True
except Exception as _e:
    print(f"[warn] PyHealth import failed ({_e}); will use mock data.")
    MIMIC3Dataset = None
    PYHEALTH_AVAILABLE = False

# from pyhealth.models import LLMSYNModel
# from pyhealth.tasks import MortalityPredictionTask
# from pyhealth.metrics import binary_metrics_fn


# Configuration
STATS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test-resources/llmsyn/stats.json"))
MIMIC3_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test-resources/core/mimic3demo"))
N_SYNTHETIC = 100  # Number of synthetic records to generate (small for API smoke test; bump to 100+ for paper)
USE_REAL_DATA = True  # Set False to force mock data even if PyHealth is available
USE_REAL_LLMSYN = True  # Set False to force mock synthetic generator
LLM_PROVIDER = "mock"  # "mock" | "openai" | "claude"
API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")

# Top ICD-9 phenotype codes for multi-label prediction (from MIMIC-III stats)
TOP_PHENOTYPES = ["4019", "4280", "42731", "41401", "5849", "25000", "2724", "51881", "5990", "53081"]


def _load_mock_dataset():
    """Fallback mock real dataset (used when PyHealth or demo files unavailable)."""
    print("Loading mock real MIMIC-III dataset...")
    rng = np.random.default_rng(42)
    base_samples = [
        {"conditions": ["4019", "41401"], "mortality": 0},
        {"conditions": ["4280", "42731"], "mortality": 1},
        {"conditions": ["5849"], "mortality": 1},
        {"conditions": ["25000", "2724"], "mortality": 0},
        {"conditions": ["486", "496"], "mortality": 1},
        {"conditions": ["41071", "41401"], "mortality": 0},
        {"conditions": ["0389", "99591"], "mortality": 1},
        {"conditions": ["40390", "5859"], "mortality": 0},
        {"conditions": ["49121"], "mortality": 1},
        {"conditions": ["4019", "4280", "42731"], "mortality": 1},
    ]
    mock_samples = []
    for s in base_samples * 10:
        enriched = dict(s)
        enriched["age"] = int(rng.integers(40, 90))
        enriched["gender"] = "Female" if rng.random() > 0.5 else "Male"
        enriched["ethnicity"] = str(rng.choice(["WHITE", "BLACK", "HISPANIC", "ASIAN"]))
        enriched["insurance"] = str(rng.choice(["Medicare", "Private", "Medicaid"]))
        mock_samples.append(enriched)
    print(f"Mock real dataset: {len(mock_samples)} samples")
    return mock_samples


def load_real_dataset():
    """Load real MIMIC-III demo dataset via PyHealth's MIMIC3Dataset.

    Returns a list of per-visit dicts with:
      - conditions: list[str] of ICD-9 diagnosis codes
      - mortality: 0/1 (HOSPITAL_EXPIRE_FLAG)
      - age, gender, ethnicity, insurance: demographics

    Falls back to mock data if PyHealth is unavailable, demo files are missing,
    or USE_REAL_DATA is False.
    """
    if not USE_REAL_DATA:
        print("USE_REAL_DATA=False; using mock dataset.")
        return _load_mock_dataset()
    if not PYHEALTH_AVAILABLE:
        print("PyHealth unavailable; falling back to mock dataset.")
        return _load_mock_dataset()
    if not os.path.isdir(MIMIC3_ROOT):
        print(f"MIMIC-III demo files not found at {MIMIC3_ROOT}; falling back to mock dataset.")
        return _load_mock_dataset()

    print(f"Loading real MIMIC-III demo dataset from {MIMIC3_ROOT}...")
    try:
        dataset = MIMIC3Dataset(
            root=MIMIC3_ROOT,
            tables=["diagnoses_icd"],
            dev=True,
        )
    except Exception as e:
        print(f"[warn] MIMIC3Dataset load failed ({e}); falling back to mock dataset.")
        return _load_mock_dataset()

    samples = []
    try:
        for patient in dataset.iter_patients():
            # Patient-level demographics
            gender = "Unknown"
            dob = None
            patient_events = patient.get_events(event_type="patients")
            if patient_events:
                pe = patient_events[0]
                gender = pe.attr_dict.get("gender", "Unknown")
                dob = pe.attr_dict.get("dob", None)

            # One sample per admission
            for adm in patient.get_events(event_type="admissions"):
                hadm_id = adm.attr_dict.get("hadm_id")
                mortality = 1 if int(adm.attr_dict.get("hospital_expire_flag", 0) or 0) == 1 else 0
                ethnicity = adm.attr_dict.get("ethnicity", "WHITE") or "WHITE"
                insurance = adm.attr_dict.get("insurance", "Medicare") or "Medicare"

                # Conditions for this admission
                conditions = []
                for dx in patient.get_events(event_type="diagnoses_icd"):
                    if dx.attr_dict.get("hadm_id") == hadm_id:
                        code = dx.attr_dict.get("icd9_code")
                        if code:
                            conditions.append(str(code))
                if not conditions:
                    continue

                # Age at admission
                age = 65
                if dob is not None and adm.timestamp is not None:
                    try:
                        age = max(0, int((adm.timestamp - dob).days / 365.25))
                        if age > 110:  # MIMIC-III obfuscates ages >89 to ~300
                            age = 90
                    except Exception:
                        age = 65

                samples.append({
                    "conditions": conditions,
                    "mortality": mortality,
                    "age": age,
                    "gender": "Female" if str(gender).upper().startswith("F") else "Male",
                    "ethnicity": str(ethnicity).split()[0].upper(),
                    "insurance": str(insurance),
                })
    except Exception as e:
        print(f"[warn] Failed to iterate MIMIC-III patients ({e}); falling back to mock dataset.")
        import traceback
        traceback.print_exc()
        return _load_mock_dataset()

    if not samples:
        print("[warn] No usable samples extracted from MIMIC-III demo; falling back to mock.")
        return _load_mock_dataset()

    print(f"Loaded {len(samples)} real MIMIC-III demo visits")
    return samples


def _generate_mock_synthetic(n_samples):
    """Mock synthetic generator (used when LLMSYN unavailable / no API key)."""
    print(f"Generating {n_samples} mock synthetic EHR records...")
    mock_records = []
    for i in range(n_samples):
        age = np.random.randint(40, 90)
        gender = "Female" if np.random.rand() > 0.5 else "Male"
        ethnicity = np.random.choice(["WHITE", "BLACK", "HISPANIC", "ASIAN"])
        insurance = np.random.choice(["Medicare", "Private", "Medicaid"])
        survived = "Yes" if np.random.rand() > 0.5 else "No"
        main_dx = np.random.choice(["4019", "4280", "42731", "41401", "5849", "25000"])
        num_comp = np.random.randint(0, 3)
        comps = np.random.choice(["4280", "42731", "41401", "5849", "25000"], num_comp, replace=False)
        has_proc = np.random.rand() > 0.7
        procs = ["CPT:93000"] if has_proc else []
        mock_records.append({
            "Age": str(age),
            "Gender": gender,
            "Ethnicity": ethnicity,
            "Insurance": insurance,
            "Survived": survived,
            "MainDiagnosis": f"ICD9:{main_dx}",
            "Complications": [f"ICD9:{c}" for c in comps],
            "Procedures": procs,
        })
    print(f"Generated {len(mock_records)} mock synthetic records")
    return mock_records


def generate_synthetic_data(n_samples):
    """Generate synthetic EHR records via real LLMSYN if available, else mock."""
    if not USE_REAL_LLMSYN:
        print("USE_REAL_LLMSYN=False; using mock generator.")
        return _generate_mock_synthetic(n_samples)
    if not PYHEALTH_AVAILABLE:
        print("PyHealth unavailable; using mock generator.")
        return _generate_mock_synthetic(n_samples)
    if LLM_PROVIDER != "mock" and not API_KEY:
        print(f"No API key for {LLM_PROVIDER} (set ANTHROPIC_API_KEY or OPENAI_API_KEY); using mock generator.")
        return _generate_mock_synthetic(n_samples)
    if not os.path.isfile(STATS_PATH):
        print(f"Stats file not found at {STATS_PATH}; using mock generator.")
        return _generate_mock_synthetic(n_samples)

    try:
        from pyhealth.models import LLMSYNModel
        from pyhealth.datasets import create_sample_dataset
        from pyhealth.tasks import SyntheticEHRGenerationTask
    except Exception as e:
        print(f"[warn] LLMSYN imports failed ({e}); using mock generator.")
        return _generate_mock_synthetic(n_samples)

    print(f"Generating {n_samples} synthetic EHR records via LLMSYN ({LLM_PROVIDER})...")
    try:
        with open(STATS_PATH) as f:
            stats = json.load(f)

        # Build a tiny seed sample dataset (LLMSYNModel requires it for schema)
        task = SyntheticEHRGenerationTask()
        seed_samples = [
            {
                "patient_id": "P000",
                "visit_id": "V000",
                "conditions": ["4019", "41401"],
                "mortality": 0,
            },
            {
                "patient_id": "P001",
                "visit_id": "V001",
                "conditions": ["4280", "42731"],
                "mortality": 1,
            },
        ]
        seed_dataset = create_sample_dataset(
            samples=seed_samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            task_name=task.task_name,
        )

        model = LLMSYNModel(
            dataset=seed_dataset,
            llm_provider=LLM_PROVIDER,
            api_key=API_KEY,
            stats=stats,
            prior_mode="full",
            enable_rag=True,
            noise_scale=0.05,
            seed=42,
        )
        records = model.generate(n=n_samples)
        print(f"Generated {len(records)} LLMSYN synthetic records")
        return records
    except Exception as e:
        print(f"[warn] LLMSYN generation failed ({e}); falling back to mock generator.")
        import traceback
        traceback.print_exc()
        return _generate_mock_synthetic(n_samples)


def preprocess_synthetic_for_training(synthetic_records):
    """Convert synthetic records to feature vectors for Random Forest.
    
    Features (to match real data):
    - Num_diagnoses (int): total number of diagnosis codes
    - Has_procedures (0/1): whether procedures are present
    - Age (float): patient age
    
    Label: Mortality (0=deceased, 1=survived)
    """
    print("Preprocessing synthetic data for training...")
    
    X = []
    y = []
    
    for record in synthetic_records:
        # Extract diagnosis codes
        diagnoses = []
        main_dx = record.get("MainDiagnosis", "")
        if main_dx:
            diagnoses.append(main_dx.replace("ICD9:", ""))
        
        complications = record.get("Complications", [])
        if isinstance(complications, str):
            complications = [c.strip().replace("ICD9:", "") for c in complications.split(",") if c.strip()]
        diagnoses.extend(complications)
        
        num_diagnoses = len(diagnoses)
        
        # Procedures
        procedures = record.get("Procedures", [])
        if isinstance(procedures, str):
            procedures = [p.strip() for p in procedures.split(",") if p.strip()]
        has_procedures = 1 if procedures else 0
        
        # Age
        age = float(record.get("Age", 65))
        
        # Feature vector
        features = [num_diagnoses, has_procedures, age]
        X.append(features)
        
        # Label: Survived (1=Yes/survived, 0=No/deceased)
        survived = 1 if record.get("Survived") == "Yes" else 0
        y.append(survived)
    
    X = np.array(X)
    y = np.array(y)
    print(f"Preprocessed {len(X)} synthetic samples with {X.shape[1]} features")
    return X, y


def train_random_forest(X_train, y_train):
    """Train Random Forest on synthetic data.

    If only a single class is present in y_train, RandomForest still fits but
    predict_proba will return shape (n, 1). The downstream evaluator handles this.
    """
    print("Training Random Forest on synthetic data...")
    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        print(f"  [warn] Synthetic labels are single-class (only {n_classes}); RF will degenerate.")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    return rf


def evaluate_on_real_data(rf_model, real_dataset):
    """Test trained model on real MIMIC-III data and compute metrics."""
    print("Evaluating on real data...")
    
    X_real = []
    y_real = []
    
    for sample in real_dataset:
        conditions = sample.get("conditions", [])
        num_diagnoses = len(conditions)
        has_procedures = 0  # No procedures in this task
        age = float(sample.get("age", 65))
        
        features = [num_diagnoses, has_procedures, age]
        X_real.append(features)
        
        mortality = sample.get("mortality", 0)
        y_real.append(mortality)
    
    X_real = np.array(X_real)
    y_real = np.array(y_real)
    
    print(f"Real test set: {len(X_real)} samples")
    
    # Predict
    y_pred = rf_model.predict(X_real)
    proba = rf_model.predict_proba(X_real)
    if proba.shape[1] >= 2:
        y_pred_prob = proba[:, 1]
    else:
        # Single-class training set: positive-class prob = the only class iff it's 1
        only_class = int(rf_model.classes_[0])
        y_pred_prob = np.full(len(X_real), float(only_class))
    
    # Compute metrics (guard against single-class real labels)
    acc = accuracy_score(y_real, y_pred)
    if len(np.unique(y_real)) < 2:
        print("  [warn] Real labels are single-class; AUROC/F1/K-S undefined.")
        auc, f1, ks = float("nan"), float("nan"), float("nan")
    else:
        auc = roc_auc_score(y_real, y_pred_prob)
        f1 = f1_score(y_real, y_pred, zero_division=0)
        fpr, tpr, _ = roc_curve(y_real, y_pred_prob)
        ks = float(max(tpr - fpr))
    
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  AUROC:        {auc:.4f}")
    print(f"  F1 Score:     {f1:.4f}")
    print(f"  K-S Stat:     {ks:.4f}")
    
    return {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "ks": ks,
    }


def compute_additional_metrics(synthetic_records, real_dataset, X_synth, X_real):
    """Compute MMD, k-anonymity, and other metrics."""
    print("Computing additional metrics...")
    
    # MMD (Maximum Mean Discrepancy) using RBF kernel
    def rbf_kernel(X, Y, sigma=1.0):
        """Compute RBF kernel between two sets."""
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        dist = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-dist / (2 * sigma ** 2))
    
    K_xx = rbf_kernel(X_synth, X_synth)
    K_yy = rbf_kernel(X_real, X_real)
    K_xy = rbf_kernel(X_synth, X_real)
    
    mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
    
    # k-anonymity: simple check - number of unique synthetic records
    # In reality, k-anonymity checks if quasi-identifiers can identify <k individuals
    # Here, approximate as uniqueness
    synth_features = [tuple(x) for x in X_synth]
    unique_synth = len(set(synth_features))
    k_anonymity = len(synth_features) / unique_synth if unique_synth > 0 else float('inf')
    
    print(f"  MMD:          {mmd:.4f}")
    print(f"  k-Anonymity:  {k_anonymity:.2f}")
    
    return {
        "mmd": mmd,
        "k_anonymity": k_anonymity,
    }


def _phenotype_features_from_synth(record):
    """Extract demographic-only features for phenotype prediction (synthetic)."""
    age = float(record.get("Age", 65))
    gender = 1 if record.get("Gender") == "Female" else 0
    eth = record.get("Ethnicity", "WHITE")
    eth_idx = ["WHITE", "BLACK", "HISPANIC", "ASIAN"].index(eth) if eth in ["WHITE", "BLACK", "HISPANIC", "ASIAN"] else 0
    ins = record.get("Insurance", "Medicare")
    ins_idx = ["Medicare", "Private", "Medicaid"].index(ins) if ins in ["Medicare", "Private", "Medicaid"] else 0
    return [age, gender, eth_idx, ins_idx]


def _phenotype_features_from_real(sample):
    """Extract demographic-only features for phenotype prediction (real)."""
    age = float(sample.get("age", 65))
    gender = 1 if sample.get("gender") == "Female" else 0
    eth = sample.get("ethnicity", "WHITE")
    eth_idx = ["WHITE", "BLACK", "HISPANIC", "ASIAN"].index(eth) if eth in ["WHITE", "BLACK", "HISPANIC", "ASIAN"] else 0
    ins = sample.get("insurance", "Medicare")
    ins_idx = ["Medicare", "Private", "Medicaid"].index(ins) if ins in ["Medicare", "Private", "Medicaid"] else 0
    return [age, gender, eth_idx, ins_idx]


def _phenotype_labels(diagnoses):
    """Build multi-hot label vector over TOP_PHENOTYPES."""
    return [1 if code in diagnoses else 0 for code in TOP_PHENOTYPES]


def preprocess_phenotype_synthetic(synthetic_records):
    """Build (X, Y) for multi-label phenotype prediction from synthetic data."""
    print("Preprocessing synthetic data for phenotype task...")
    X, Y = [], []
    for r in synthetic_records:
        diagnoses = set()
        main_dx = r.get("MainDiagnosis", "").replace("ICD9:", "")
        if main_dx:
            diagnoses.add(main_dx)
        for c in r.get("Complications", []):
            diagnoses.add(c.replace("ICD9:", "") if isinstance(c, str) else c)
        X.append(_phenotype_features_from_synth(r))
        Y.append(_phenotype_labels(diagnoses))
    X, Y = np.array(X), np.array(Y)
    print(f"  Synthetic phenotype set: {X.shape[0]} samples, {Y.shape[1]} phenotype labels")
    return X, Y


def preprocess_phenotype_real(real_dataset):
    """Build (X, Y) for multi-label phenotype prediction from real data."""
    X, Y = [], []
    for s in real_dataset:
        diagnoses = set(s.get("conditions", []))
        X.append(_phenotype_features_from_real(s))
        Y.append(_phenotype_labels(diagnoses))
    return np.array(X), np.array(Y)


def train_phenotype_rf(X_train, Y_train):
    """Train one RandomForest per phenotype (multi-label one-vs-rest)."""
    print("Training Random Forests for phenotype prediction...")
    models = []
    for j in range(Y_train.shape[1]):
        y = Y_train[:, j]
        if len(np.unique(y)) < 2:
            models.append(None)  # Not enough class diversity to train
            continue
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y)
        models.append(rf)
    n_trained = sum(m is not None for m in models)
    print(f"  Trained {n_trained}/{len(models)} phenotype classifiers")
    return models


def evaluate_phenotype_on_real(models, X_real, Y_real):
    """Evaluate per-phenotype RFs on real data; report macro-averaged metrics."""
    print("Evaluating phenotype models on real data...")
    accs, aucs, f1s = [], [], []
    for j, rf in enumerate(models):
        if rf is None:
            continue
        y_true = Y_real[:, j]
        if len(np.unique(y_true)) < 2:
            continue  # AUROC undefined
        y_pred = rf.predict(X_real)
        proba = rf.predict_proba(X_real)
        if proba.shape[1] >= 2:
            y_prob = proba[:, 1]
        else:
            only_class = int(rf.classes_[0])
            y_prob = np.full(len(X_real), float(only_class))
        accs.append(accuracy_score(y_true, y_pred))
        try:
            aucs.append(roc_auc_score(y_true, y_prob))
        except ValueError:
            pass
        f1s.append(f1_score(y_true, y_pred, zero_division=0))

    macro_acc = float(np.mean(accs)) if accs else 0.0
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    print(f"  Phenotype Macro-Accuracy: {macro_acc:.4f}")
    print(f"  Phenotype Macro-AUROC:    {macro_auc:.4f}")
    print(f"  Phenotype Macro-F1:       {macro_f1:.4f}")
    print(f"  Evaluable phenotypes:     {len(accs)}/{len(models)}")

    return {
        "phenotype_acc": macro_acc,
        "phenotype_auc": macro_auc,
        "phenotype_f1": macro_f1,
    }


def main():
    try:
        print("=" * 60)
        print("LLMSYN VALIDATION: RANDOM FOREST TSTR PIPELINE")
        print("=" * 60)

        # Step 1: Load real dataset
        real_dataset = load_real_dataset()

        # Step 2: Generate synthetic data
        synthetic_records = generate_synthetic_data(N_SYNTHETIC)

        # Step 3: Preprocess synthetic data
        X_synth, y_synth = preprocess_synthetic_for_training(synthetic_records)

        # Step 4: Train Random Forest
        rf_model = train_random_forest(X_synth, y_synth)

        # Step 5: Evaluate on real data
        metrics = evaluate_on_real_data(rf_model, real_dataset)

        # Step 6: Compute additional metrics
        # Get X_real
        X_real = []
        for sample in real_dataset:
            conditions = sample.get("conditions", [])
            num_diagnoses = len(conditions)
            has_procedures = 0
            age = float(sample.get("age", 65))
            features = [num_diagnoses, has_procedures, age]
            X_real.append(features)
        X_real = np.array(X_real)
        
        additional_metrics = compute_additional_metrics(synthetic_records, real_dataset, X_synth, X_real)

        # Step 7: Phenotype prediction (multi-label TSTR)
        print("\n" + "-" * 60)
        print("PHENOTYPE PREDICTION TASK (multi-label)")
        print("-" * 60)
        Xp_synth, Yp_synth = preprocess_phenotype_synthetic(synthetic_records)
        Xp_real, Yp_real = preprocess_phenotype_real(real_dataset)
        phenotype_models = train_phenotype_rf(Xp_synth, Yp_synth)
        phenotype_metrics = evaluate_phenotype_on_real(phenotype_models, Xp_real, Yp_real)

        # Combine results
        all_metrics = {**metrics, **additional_metrics, **phenotype_metrics}
        print("\nFinal Results:")
        for key, value in all_metrics.items():
            try:
                print(f"  {key}: {float(value):.4f}")
            except (TypeError, ValueError):
                print(f"  {key}: {value}")

    except ImportError as e:
        print(f"Missing dependency: {e}. Please install numpy and scikit-learn.")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
