import sys
from pathlib import Path

# Add fork to Python path
fork_path = "."
if fork_path not in sys.path:
    sys.path.insert(0, fork_path)

from pyhealth.datasets.mimic_cxr_reports import MIMICCXRReportsDataset

from pyhealth.datasets.configs.config import load_yaml_config
yaml_path = Path(fork_path) / "pyhealth" / "datasets" / "configs" / "mimic_cxr_reports.yaml"
print("YAML path used:", yaml_path)
with open(yaml_path, "r") as f:
    print(f.read())

# Path to MIMIC-CXR reports extracted data
root_path = "/Users/lokanathdas/Downloads/"  # <-- provide the path which contains the mimic-cxr-reports.zip file

# Check contents of root_path for verification
print("Contents of root_path:", list(Path(root_path).iterdir()))

# Initialize dataset (minimal fix: remove tables argument)
ds = MIMICCXRReportsDataset(
    root=root_path,
    patients=None,  # test subset
    dev_mode=True,
    limit=10
)

# Load samples and print
samples = ds.get_samples()
print(f"\nLoaded {len(samples)} samples from MIMIC-CXR Reports\n")
for i, s in enumerate(samples):
    print(f"Sample {i+1}:")
    print(f"Patient ID: {s['patient_id']}")
    print(f"Study ID: {s['study_id']}")
    print(f"Path: {s['path']}")
    print(f"Findings: {s['findings']}")
    print(f"Impression: {s['impression']}\n")