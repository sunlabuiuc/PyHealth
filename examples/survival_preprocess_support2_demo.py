"""
Demo script for survival prediction preprocessing using SUPPORT2 dataset.

This example demonstrates how to:
1. Load the SUPPORT2 dataset (using test data with 3 patients)
2. Apply the preprocessing task to extract features and labels
3. Examine preprocessed samples ready for model training

The preprocessing task extracts:
- Features from raw patient data (demographics, vitals, labs, scores, etc.)
- Ground truth survival probabilities from the dataset (surv2m/surv6m fields)
- Structures data into samples ready for training a prediction model

Note: The survival probabilities shown are ground truth labels extracted from the
dataset (surv2m/surv6m columns). These are the target variables that a model
would learn to predict from the extracted features.

This example uses the synthetic test dataset from test-resources (3 patients).
For real usage, replace the path with your actual SUPPORT2 dataset.
"""

import warnings
import logging
from pathlib import Path

# Suppress warnings and reduce logging verbosity
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logging.getLogger("pyhealth").setLevel(logging.WARNING)
logging.getLogger("pyhealth.datasets").setLevel(logging.WARNING)
logging.getLogger("pyhealth.datasets.support2").setLevel(logging.WARNING)
logging.getLogger("pyhealth.datasets.base_dataset").setLevel(logging.WARNING)

# Import pyhealth modules
from pyhealth.datasets import Support2Dataset
from pyhealth.tasks import SurvivalPreprocessSupport2

# Suppress tqdm progress bars for cleaner output
try:
    def noop_tqdm(iterable, *args, **kwargs):
        return iterable
    from pyhealth.datasets import base_dataset, sample_dataset
    base_dataset.tqdm = noop_tqdm
    sample_dataset.tqdm = noop_tqdm
    import tqdm
    tqdm.tqdm = noop_tqdm
except (ImportError, AttributeError):
    pass

# Step 1: Load dataset using test data
print("=" * 70)
print("Step 1: Load SUPPORT2 Dataset")
print("=" * 70)
script_dir = Path(__file__).parent
test_data_path = script_dir.parent / "test-resources" / "core" / "support2"

dataset = Support2Dataset(
    root=str(test_data_path),
    tables=["support2"],
)

print(f"Loaded dataset with {len(dataset.unique_patient_ids)} patients\n")

# Step 2: Apply preprocessing task to extract features and labels (2-month horizon)
print("=" * 70)
print("Step 2: Apply Survival Preprocessing Task")
print("=" * 70)
task = SurvivalPreprocessSupport2(time_horizon="2m")
sample_dataset = dataset.set_task(task=task)

print(f"Generated {len(sample_dataset)} samples")
print(f"Input schema: {sample_dataset.input_schema}")
print(f"Output schema: {sample_dataset.output_schema}\n")

# Helper function to decode tensor indices to feature strings
def decode_features(tensor, processor):
    """Decode tensor indices back to original feature strings."""
    if processor is None or not hasattr(processor, 'code_vocab'):
        return [str(idx.item()) for idx in tensor]
    reverse_vocab = {idx: token for token, idx in processor.code_vocab.items()}
    return [reverse_vocab.get(idx.item(), f"<unk:{idx.item()}>") for idx in tensor]

# Step 3: Display features for all samples
print("=" * 70)
print("Step 3: Examine Preprocessed Samples")
print("=" * 70)
# Sort samples by patient_id to ensure consistent order
samples = sorted(sample_dataset, key=lambda x: int(x['patient_id']))
for sample in samples:
    # Decode features for this sample
    demographics = decode_features(
        sample['demographics'],
        sample_dataset.input_processors.get('demographics')
    )
    disease_codes = decode_features(
        sample['disease_codes'],
        sample_dataset.input_processors.get('disease_codes')
    )
    vitals = decode_features(
        sample['vitals'],
        sample_dataset.input_processors.get('vitals')
    )
    labs = decode_features(
        sample['labs'],
        sample_dataset.input_processors.get('labs')
    )
    scores = decode_features(
        sample['scores'],
        sample_dataset.input_processors.get('scores')
    )
    comorbidities = decode_features(
        sample['comorbidities'],
        sample_dataset.input_processors.get('comorbidities')
    )
    
    # Display this patient's features
    print(f"\nPatient {sample['patient_id']}:")
    print(f"  Demographics: {', '.join(demographics)}")
    print(f"  Disease Codes: {', '.join(disease_codes)}")
    print(f"  Vitals: {', '.join(vitals)}")
    print(f"  Labs: {', '.join(labs)}")
    print(f"  Scores: {', '.join(scores)}")
    print(f"  Comorbidities: {', '.join(comorbidities)}")
    print(f"  Survival Probability (2m): {sample['survival_probability'].item():.4f}")

print("\n")
print("=" * 70)
print("Preprocessing Complete!")
print("=" * 70)
print("The samples are ready for model training.")
