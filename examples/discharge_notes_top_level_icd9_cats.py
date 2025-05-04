"""
Author: Fabricio Brigagao (fb8)

Purpose
-------
Creates a KeyClass-ready dataset from MIMIC-III discharge summaries, following the
procedure described in *“Classifying Unstructured Clinical Notes via Automatic
Weak Supervision”* (https://arxiv.org/abs/2206.12088).

Highlights
----------
* Uses the custom task also created for this: ``MIMIC3DischargeNotesICD9Coding``.
* **No TF-IDF filtering** is applied to the discharge-note text since 
  the paper is unclear on this aspect (possible future improvement).
* Each note is mapped to zero or more of the 19 top-level ICD-9 categories
  based on the ICD-9 codes recorded in MIMIC-III.
* Generates train/test splits and writes files in the exact layout expected by KeyClass:

    ┌────────────────────┬─────────────────────────────────────────────────────┐
    │ File               │ Contents                                            │
    ├────────────────────┼─────────────────────────────────────────────────────┤
    │ labels.txt         │ Names of the 19 ICD-9 top-level categories          │
    │ train.txt          │ One discharge note per line (training)              │
    │ train_labels.txt   │ 19-dim binary vectors for the training set          │
    │ test.txt           │ One discharge note per line (test)                  │
    │ test_labels.txt    │ 19-dim binary vectors for the test set              │
    └────────────────────┴─────────────────────────────────────────────────────┘

ICD-9 Top-Level Category Mappings
-----------------------------------------------------------------------------------------
Range (3-digit) | Code   | Description
--------------- | ------ | --------------------------------------------------------------
001 – 139       | cat:01 | Infectious and parasitic diseases
140 – 239       | cat:02 | Neoplasms
240 – 279       | cat:03 | Endocrine, nutritional and metabolic disorders
280 – 289       | cat:04 | Diseases of the blood and blood-forming organs
290 – 319       | cat:05 | Mental disorders
320 – 359       | cat:06 | Diseases of the nervous system
360 – 389       | cat:07 | Diseases of the sense organs
390 – 459       | cat:08 | Diseases of the circulatory system
460 – 519       | cat:09 | Diseases of the respiratory system
520 – 579       | cat:10 | Diseases of the digestive system
580 – 629       | cat:11 | Diseases of the genitourinary system
630 – 679       | cat:12 | Pregnancy, childbirth and puerperium
680 – 709       | cat:13 | Diseases of the skin and subcutaneous tissue
710 – 739       | cat:14 | Diseases of the musculoskeletal system and connective tissue
740 – 759       | cat:15 | Congenital anomalies
760 – 779       | cat:16 | Certain conditions originating in the perinatal period
800 – 999       | cat:17 | Injury and poisoning
E-codes         | cat:18 | External causes of injury
V-codes         | cat:19 | Supplementary factors influencing health status

Before you run
--------------
Set the constant ``OUTPUT_DIR`` below to the directory where you want the generated files to be written.
Set the constant ``MIMIC_DB_LOCATION`` with the location of the MIMIC database files.
"""
import sys
sys.path.append("../PyHealth")
from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import MIMIC3DischargeNotesICD9Coding
from pyhealth.datasets import split_by_sample
import numpy as np
import os
import torch 

# Directory to save the final dataset files
OUTPUT_DIR = "example_output_dev"
# Location of the MIMIC database files
MIMIC_DB_LOCATION = "../mimicdatabase"

def write_labels_file(label_file):
    """ Writes the label.txt file with the names of the 19 top-level ICD-9 categories."""
    with open(label_file, 'w') as f:
        labels = [
            "Infectious & parasitic",
            "Neoplasms",
            "Endocrine, nutritional and metabolic",
            "Blood & blood-forming organs",
            "Mental disorders",
            "Nervous system",
            "Sense organs",
            "Circulatory system",
            "Respiratory system",
            "Digestive system",
            "Genitourinary system",
            "Pregnancy & childbirth complications",
            "Skin & subcutaneous tissue",
            "Musculoskeletal system & connective tissue",
            "Congenital anomalies",
            "Perinatal period conditions",
            "Injury and poisoning",
            "External causes of injury",
            "Supplementary"
        ]
        for label in labels:
            f.write(f"{label}\n")
        print()

def write_output_files(dataset, feature_filename, labels_filename):
    """ Writes the train/test sets and respective labels to file."""

    with open(feature_filename, 'w', encoding='utf-8') as f_text, \
        open(labels_filename, 'w', encoding='utf-8') as f_labels:

        for i, sample_dict in enumerate(dataset):

            # 1. Extract and clean the discharge note text
            text_feature = sample_dict.get('text', '')
            # Replace newlines within the text with spaces for single-line output
            text_line = text_feature.replace('\n', ' ').replace('\r', ' ').strip()

            # 2. Process the label tensor (already computed by PyHealth)
            label_tensor = sample_dict.get('top_level_icd9_cats')

            if label_tensor is None:
                print(f"Warning: Missing 'top_level_icd9_cats' tensor in sample {i}. Skipping label.")
                return 1
            elif not isinstance(label_tensor, torch.Tensor):
                 print(f"Warning: 'top_level_icd9_cats' in sample {i} is not a tensor (type: {type(label_tensor)}). Skipping label.")
                 label_string = ""
                 return 1
            else:
                # Convert tensor elements (e.g., 1.0, 0.0) to integers (1, 0) and then to strings ('1', '0') and join them.
                label_string = "".join(map(str, label_tensor.int().tolist()))

            # 3. Write to the respective files
            f_text.write(text_line + '\n')
            f_labels.write(label_string + '\n')

    print(f"Successfully wrote:")
    print(f"- Features to: {feature_filename}")
    print(f"- Labels to: {labels_filename}")


def main():
    """ Writes the train/test sets and respective labels to file."""

    dataset = MIMIC3Dataset(
        root=MIMIC_DB_LOCATION,
        dataset_name="mimic3",
        tables=[ "diagnoses_icd", "noteevents" ],
        dev=True
    )

    #dataset.stats()

    # Creates the new task for generating the MIMIC-derived KeyClass dataset
    mimic3_discharge_notes_coding = MIMIC3DischargeNotesICD9Coding()
    samples = dataset.set_task(mimic3_discharge_notes_coding)

    # Print sample information
    print(f"Total samples generated: {len(samples)}")
    if len(samples) > 0:
        print("First sample:")
        print(f"  - Patient ID: {samples[0]['patient_id']}")
        print(f"  - Text length: {len(samples[0]['text'])} characters")
        print(f"  - Number of Top Level ICD9 Categories: {len(samples[0]['top_level_icd9_cats'])}")
        if len(samples[0]['top_level_icd9_cats']) > 0:
            print(f"  - Top-Level ICD9 Categories: {samples[0]['top_level_icd9_cats']}")

    # KeyClass uses a 75%/25% with only training and test sets (no validation)
    train_dataset, val_dataset, test_dataset = split_by_sample(
        dataset=samples,
        ratios=[0.75, 0, 0.25]
    )

    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    print(train_dataset)

    # Writing the dataset files in the format required by KeyClass

    # Create output directory if it does not exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define destination filenames
    train_filename = os.path.join(OUTPUT_DIR, "train.txt")
    train_labels_filename = os.path.join(OUTPUT_DIR, "train_labels.txt")
    test_filename = os.path.join(OUTPUT_DIR, "test.txt")
    test_labels_filename = os.path.join(OUTPUT_DIR, "test_labels.txt")
    label_filename = os.path.join(OUTPUT_DIR, "labels.txt")

    # Write the files
    write_output_files(train_dataset, train_filename, train_labels_filename)
    write_output_files(test_dataset, test_filename, test_labels_filename)
    write_labels_file(label_filename)

if __name__ == "__main__":
    main()