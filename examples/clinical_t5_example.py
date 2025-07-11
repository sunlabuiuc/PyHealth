# examples/clinical_t5_example.py
"""
ClinicalT5 for Radiology Report Generation
"""
from pyhealth.datasets import MIMICCXRDataset
from pyhealth.models import ClinicalT5
from pyhealth.datasets.utils import get_dataset_root

# 1. Load dataset
dataset = MIMICCXRDataset(
    root=get_dataset_root("mimic-cxr"),
    dev=True,  # use small subset for demo
)

# 2. Define task (using PyHealth's built-in function)
task_dataset = dataset.set_task(
    task_fn="radreport_generation_mimiccxr_fn",
    task_kwargs={"max_len": 256}
)

# 3. Initialize model
model = ClinicalT5(
    dataset=task_dataset,
    model_name="clinical-t5-large",
    mode="generation"
)

# 4. Inference
sample_findings = "The heart size is normal. The lungs are clear."
report = model.predict(sample_findings)
print(f"Generated Report:\n{report}")