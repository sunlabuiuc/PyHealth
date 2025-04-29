# MIMIC-III Database Generator

This repository contains tools for processing the MIMIC-III Clinical Database and generating SQLite databases for use in medical research and machine learning applications. This is a crucial step in preparing data for the "Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records" (CHIL 2022) dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pyhealth.git
   cd pyhealth
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate   
   ```

## Citation

If you use this code in your research, please cite the following papers:

```bibtex
@inproceedings{wang2020text,
  title={Text-to-SQL Generation for Question Answering on Electronic Medical Records},
  author={Wang, Ping and Shi, Tian and Reddy, Chandan K},
  booktitle={Proceedings of The Web Conference 2020},
  pages={350--361},
  year={2020},
  doi={10.1145/3366423.3380128}
}
```

This work is based on the TREQS repository: [https://github.com/wangpinggl/TREQS]


## Dataset Access

The MIMIC-III Clinical Database is a large, freely-available database comprising deidentified health-related data associated with over forty thousand patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012.

To access the MIMIC-III database:
1. Register on [PhysioNet](https://physionet.org/content/mimiciii/1.4/)
2. Complete the required training
3. Sign the data use agreement
4. Download the database

## Features

- Processes MIMIC-III CSV files into SQLite databases
- Creates two database versions:
  - Full database (`mimic_all.db`)
  - Sampled database (`mimic.db`) with 100 randomly selected admissions
- Generates five main tables:
  - DEMOGRAPHIC: Patient demographic information
  - DIAGNOSES: Patient diagnoses with ICD-9 codes
  - PROCEDURES: Medical procedures with ICD-9 codes
  - PRESCRIPTIONS: Medication prescriptions
  - LAB: Laboratory test results

## Key Files

### id2name.csv
This file serves as a mapping between patient IDs and virtual names in the generated database. It is used to:
- Map MIMIC-III patient IDs to virtual names for better readability
- Maintain patient privacy while providing meaningful identifiers
- Enable easier tracking and analysis of patient data
- Support the generation of the DEMOGRAPHIC table with virtual patient names

## Usage

1. Place your MIMIC-III CSV files in the `data/mimic_db_generator` directory
2. Run the database generator:
   ```bash
   cd pyhealth/data/mimic_db_generator
   python process_mimic_db.py
   ```
3. Test the generated databases:
   ```bash
   python test_db.py
   ```

## Database Applications

The generated SQLite databases can be used for:

1. **Research**
   - Medical data analysis
   - Clinical studies
   - Healthcare research

2. **Development**
   - Local data access
   - SQL query testing
   - Application development

## Integration with Uncertainty-Aware Text-to-Program

This database generator is a crucial component in preparing data for the "Uncertainty-Aware Text-to-Program for Question Answering on Structured Electronic Health Records" (CHIL 2022) dataset. The generated SQLite databases serve as the foundation for:

1. Structured EHR data representation
2. Question answering system development
3. Uncertainty modeling in medical decision-making
4. Transfer prediction tasks

## Requirements

- Python 3.6+
- Required Python packages:
  - pandas
  - numpy
  - sqlite3
  - gzip

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Required CSV Files
The following CSV files are required after unzipping:
- PATIENTS.csv
- ADMISSIONS.csv
- DIAGNOSES_ICD.csv
- D_ICD_DIAGNOSES.csv
- PROCEDURES_ICD.csv
- D_ICD_PROCEDURES.csv
- PRESCRIPTIONS.csv
- LABEVENTS.csv
- D_LABITEMS.csv 