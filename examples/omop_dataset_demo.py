"""
Demo script for loading and exploring OMOP CDM dataset.

This example demonstrates how to:
1. Load an OMOP CDM dataset
2. Explore dataset statistics
3. Access patient data
4. Examine events from different clinical tables
5. Create task-specific samples for mortality prediction

Dataset path: MIMIC-IV Demo OMOP CDM version
"""

from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import MortalityPredictionOMOP

# Initialize OMOP dataset
# This dataset includes MIMIC-IV demo data in OMOP CDM 5.3 format
dataset = OMOPDataset(
    root=(
        "/home/johnwu3/projects/PyHealth_Branch_Testing/datasets/"
        "physionet.org/files/mimic-iv-demo-omop/0.9/1_omop_data_csv"
    ),
    tables=[
        "condition_occurrence",  # Diagnoses
        "procedure_occurrence",  # Procedures
        "drug_exposure",  # Medications
        "measurement",  # Lab measurements
    ],
    dev=False,  # Set to True to limit to 1000 patients for faster testing
)

# Display dataset statistics
print("=" * 70)
print("OMOP Dataset Statistics")
print("=" * 70)
dataset.stats()
print()

# Get list of unique patient IDs
patient_ids = dataset.unique_patient_ids
print(f"Total number of patients: {len(patient_ids)}")
print(f"Sample patient IDs: {patient_ids[:5]}")
print()

# Access individual patient data
print("=" * 70)
print("Patient Data Example")
print("=" * 70)
first_patient_id = patient_ids[0]
patient = dataset.get_patient(first_patient_id)

print(f"Patient ID: {patient.patient_id}")
print(f"Number of events: {len(patient.data_source)}")
print()

# Display first few events
print("First 10 events for this patient:")
print(patient.data_source.head(10))
print()

# Examine events by type
print("=" * 70)
print("Events by Type")
print("=" * 70)
for event_type in patient.event_type_partitions.keys():
    event_type_str = event_type[0]  # Keys are tuples
    count = len(patient.get_events(event_type=event_type_str))
    print(f"  {event_type_str}: {count} events")
print()

# Examine condition occurrences (diagnoses)
condition_events = patient.get_events(event_type="condition_occurrence")
if len(condition_events) > 0:
    print("=" * 70)
    print("Condition Occurrences (Diagnoses)")
    print("=" * 70)
    print(f"Total condition events: {len(condition_events)}")
    print("First 5 condition events:")
    for i, event in enumerate(condition_events[:5]):
        print(f"  {i+1}. {event.timestamp}: {event.attr_dict}")
    print()

# Examine drug exposures (medications)
drug_events = patient.get_events(event_type="drug_exposure")
if len(drug_events) > 0:
    print("=" * 70)
    print("Drug Exposures (Medications)")
    print("=" * 70)
    print(f"Total drug events: {len(drug_events)}")
    print("First 5 drug events:")
    for i, event in enumerate(drug_events[:5]):
        print(f"  {i+1}. {event.timestamp}: {event.attr_dict}")
    print()

# Examine measurements (labs)
measurement_events = patient.get_events(event_type="measurement")
if len(measurement_events) > 0:
    print("=" * 70)
    print("Measurements (Labs)")
    print("=" * 70)
    print(f"Total measurement events: {len(measurement_events)}")
    print("First 5 measurement events:")
    for i, event in enumerate(measurement_events[:5]):
        print(f"  {i+1}. {event.timestamp}: {event.attr_dict}")
    print()

# Iterate through multiple patients
print("=" * 70)
print("Iterating Through Patients")
print("=" * 70)
for i, patient in enumerate(dataset.iter_patients()):
    if i >= 3:  # Only show first 3 patients
        break
    print(
        f"Patient {i+1}: ID={patient.patient_id}, " f"Events={len(patient.data_source)}"
    )
print()

print("=" * 70)
print("Mortality Prediction Task Demo")
print("=" * 70)
print("\nCreating mortality prediction samples...")

# Create mortality prediction task
mortality_task = MortalityPredictionOMOP()

# Generate samples (using a small subset for demo)
sample_dataset = dataset.set_task(task=mortality_task)

print(f"\nGenerated {len(sample_dataset)} samples")
print(f"Input schema: {sample_dataset.input_schema}")
print(f"Output schema: {sample_dataset.output_schema}")

# Show mortality label distribution
print("\n" + "=" * 70)
print("Mortality Label Distribution")
print("=" * 70)
mortality_counts = {}
for sample in sample_dataset.samples:
    label = sample["mortality"]
    # Extract the actual value from tensor if needed
    if hasattr(label, "item"):
        label = int(label.item())
    elif isinstance(label, (list, tuple)):
        label = int(label[0])
    else:
        label = int(label)
    mortality_counts[label] = mortality_counts.get(label, 0) + 1

total_samples = len(sample_dataset.samples)
alive_count = mortality_counts.get(0, 0)
death_count = mortality_counts.get(1, 0)
alive_pct = (alive_count / total_samples * 100) if total_samples > 0 else 0
death_pct = (death_count / total_samples * 100) if total_samples > 0 else 0

print(f"Total samples: {total_samples}\n")
print(f"Alive: {alive_count} samples ({alive_pct:.1f}%)")
print(f"Death: {death_count} samples ({death_pct:.1f}%)")
print()

print("=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nNext steps:")
print("- Use dataset.set_task() to create task-specific samples")
print("- Apply feature processors to transform the data")
print("- Train machine learning models on the processed data")
print("\nFor more information, see:")
print("https://pyhealth.readthedocs.io/")
