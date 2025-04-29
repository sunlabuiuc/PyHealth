"""Script to process MIMIC-III Clinical Database and generate SQLite databases.

This script performs the following tasks:
1. Unzips the MIMIC-III CSV files
2. Processes the data into a SQLite database
3. Creates both a full database (mimic_all.db) and a sampled database (mimic.db)
   containing 100 randomly selected admissions.

The script generates five main tables:
- DEMOGRAPHIC: Patient demographic information
- DIAGNOSES: Patient diagnoses with ICD-9 codes
- PROCEDURES: Medical procedures with ICD-9 codes
- PRESCRIPTIONS: Medication prescriptions
- LAB: Laboratory test results
"""

import os
import csv
import shutil
import sqlite3
import gzip
import pandas as pd
import numpy as np
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

def get_patient_name(data_dir: str) -> Dict[str, str]:
    """Get patient ID to name mapping from id2name.csv.

    This function reads the id2name.csv file which contains mappings between
    patient IDs and virtual names for privacy protection.

    Args:
        data_dir: Directory containing the id2name.csv file.

    Returns:
        Dictionary mapping patient IDs to virtual names.

    Raises:
        FileNotFoundError: If id2name.csv cannot be found.
    """
    pat_id2name = {}
    file_ = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'id2name.csv')
    with open(file_, 'r') as fp:
        for line in csv.reader(fp, delimiter=','):
            pat_id2name[line[0]] = line[1]
    return pat_id2name

def read_table(data_dir: str, data_file: str) -> List[Dict[str, str]]:
    """Read a CSV table from the specified directory.

    This function reads a CSV file and converts it into a list of dictionaries,
    where each dictionary represents a row with column names as keys.

    Args:
        data_dir: Directory containing the CSV file.
        data_file: Name of the CSV file to read.

    Returns:
        List of dictionaries containing the table data, where each dictionary
        represents a row with column names as keys.

    Raises:
        FileNotFoundError: If the specified CSV file cannot be found.
    """
    out_info = []
    file_ = os.path.join(data_dir, data_file)
    with open(file_, 'r') as fp:
        reader = csv.reader(fp, delimiter=',')
        for line in reader:
            header = line
            break
        for line in reader:
            arr = {}
            for k in range(len(header)):
                arr[header[k]] = line[k]
            out_info.append(arr)
    return out_info

def show_progress(current: int, total: int) -> None:
    """Display a progress bar in the console.

    This function displays a progress bar showing the percentage of completion
    for a long-running operation.

    Args:
        current: Current progress value.
        total: Total value to reach (100%).
    """
    progress = int(round(100.0 * float(current) / float(total)))
    progress_bar = '[' + '>' * progress + ' ' * (100 - progress) + ']'
    sys.stdout.write(progress_bar + str(progress) + '%' + '\r')
    sys.stdout.flush()

def build_demographic_table(
    data_dir: str,
    out_dir: str,
    conn: sqlite3.Connection
) -> None:
    """Build the demographic table from patient and admission data.

    This function processes the PATIENTS and ADMISSIONS tables to create a
    comprehensive demographic table containing patient information.

    Args:
        data_dir: Directory containing the input CSV files.
        out_dir: Directory to save the output files.
        conn: SQLite database connection.
    """
    print('Build demographic_table')
    pat_id2name = get_patient_name('process_mimic_db')
    pat_info = read_table(data_dir, 'PATIENTS.csv')
    adm_info = read_table(data_dir, 'ADMISSIONS.csv')
    print('-- Process PATIENTS')
    cnt = 0
    for itm in pat_info:
        cnt += 1
        show_progress(cnt, len(pat_info))
        itm['NAME'] = pat_id2name[itm['SUBJECT_ID']]
        
        dob = datetime.strptime(itm['DOB'], '%Y-%m-%d %H:%M:%S')
        itm['DOB_YEAR'] = str(dob.year)
        
        if len(itm['DOD']) > 0:
            dod = datetime.strptime(itm['DOD'], '%Y-%m-%d %H:%M:%S')
            itm['DOD_YEAR'] = str(dod.year)
        else:
            itm['DOD_YEAR'] = ''
            
    pat_dic = {ky['SUBJECT_ID']: ky for ky in pat_info}
    print()
    print('-- Process ADMISSIONS')
    cnt = 0
    for itm in adm_info:
        cnt += 1
        show_progress(cnt, len(adm_info))
        # patients.csv
        for ss in pat_dic[itm['SUBJECT_ID']]:
            if ss == 'ROW_ID' or ss == 'SUBJECT_ID':
                continue
            itm[ss] = pat_dic[itm['SUBJECT_ID']][ss]
        # admissions.csv
        admtime = datetime.strptime(itm['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        itm['ADMITYEAR'] = str(admtime.year)
        dctime = datetime.strptime(itm['DISCHTIME'], '%Y-%m-%d %H:%M:%S')
        itm['DAYS_STAY'] = str((dctime - admtime).days)
        itm['AGE'] = str(int(itm['ADMITYEAR']) - int(itm['DOB_YEAR']))
        if int(itm['AGE']) > 89:
            itm['AGE'] = str(89 + int(itm['AGE']) - 300)
    print()
    print('-- write table')
    header = [
        'SUBJECT_ID', 'HADM_ID', 'NAME', 'MARITAL_STATUS', 'AGE', 'DOB',
        'GENDER', 'LANGUAGE', 'RELIGION', 'ADMISSION_TYPE', 'DAYS_STAY',
        'INSURANCE', 'ETHNICITY', 'EXPIRE_FLAG', 'ADMISSION_LOCATION',
        'DISCHARGE_LOCATION', 'DIAGNOSIS', 'DOD', 'DOB_YEAR', 'DOD_YEAR',
        'ADMITTIME', 'DISCHTIME', 'ADMITYEAR'
    ]
            
    with open(os.path.join(out_dir, 'DEMOGRAPHIC.csv'), 'w') as fout:
        fout.write('\"' + '\",\"'.join(header) + '\"\n')
        for itm in adm_info:
            arr = []
            for wd in header:
                arr.append(itm[wd])
            fout.write('\"' + '\",\"'.join(arr) + '\"\n')
    print('-- write sql')
    data = pd.read_csv(
        os.path.join(out_dir, 'DEMOGRAPHIC.csv'),
        dtype={'HADM_ID': str, "DOD_YEAR": float, "SUBJECT_ID": str})
    data.to_sql('DEMOGRAPHIC', conn, if_exists='replace', index=False)

def build_diagnoses_table(data_dir, out_dir, conn):
    """Build the diagnoses table from ICD codes and descriptions.

    Args:
        data_dir (str): Directory containing the input CSV files.
        out_dir (str): Directory to save the output files.
        conn: SQLite database connection.
    """
    print('Build diagnoses_table')
    left = pd.read_csv(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'), dtype=str)
    right = pd.read_csv(os.path.join(data_dir, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    left = left.drop(columns=['ROW_ID', 'SEQ_NUM'])
    right = right.drop(columns=['ROW_ID'])
    out = pd.merge(left, right, on='ICD9_CODE')
    out = out.sort_values(by='HADM_ID')
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'DIAGNOSES.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('DIAGNOSES', conn, if_exists='replace', index=False)
    
def build_procedures_table(data_dir, out_dir, conn):
    """Build the procedures table from ICD codes and descriptions.

    Args:
        data_dir (str): Directory containing the input CSV files.
        out_dir (str): Directory to save the output files.
        conn: SQLite database connection.
    """
    print('Build procedures_table')
    left = pd.read_csv(os.path.join(data_dir, 'PROCEDURES_ICD.csv'), dtype=str)
    right = pd.read_csv(os.path.join(data_dir, 'D_ICD_PROCEDURES.csv'), dtype=str)
    left = left.drop(columns=['ROW_ID', 'SEQ_NUM'])
    right = right.drop(columns=['ROW_ID'])
    out = pd.merge(left, right, on='ICD9_CODE')
    out = out.sort_values(by='HADM_ID')
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'PROCEDURES.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('PROCEDURES', conn, if_exists='replace', index=False)
    
def build_prescriptions_table(data_dir, out_dir, conn):
    """Build the prescriptions table from medication data.

    Args:
        data_dir (str): Directory containing the input CSV files.
        out_dir (str): Directory to save the output files.
        conn: SQLite database connection.
    """
    print('Build prescriptions_table')
    data = pd.read_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), dtype=str)
    data = data.drop(columns=[
        'ROW_ID', 'GSN', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'NDC',
        'PROD_STRENGTH', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'STARTDATE',
        'ENDDATE'
    ])
    data = data.dropna(subset=['DOSE_VAL_RX', 'DOSE_UNIT_RX'])
    data['DRUG_DOSE'] = data[['DOSE_VAL_RX', 'DOSE_UNIT_RX']].apply(
        lambda x: ''.join(x), axis=1)
    data = data.drop(columns=['DOSE_VAL_RX', 'DOSE_UNIT_RX'])
    print('-- write table')
    data.to_csv(os.path.join(out_dir, 'PRESCRIPTIONS.csv'), sep=',', index=False)
    print('-- write sql')
    data.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False)
    
def build_lab_table(data_dir, out_dir, conn):
    """Build the lab table from laboratory test results.

    Args:
        data_dir (str): Directory containing the input CSV files.
        out_dir (str): Directory to save the output files.
        conn: SQLite database connection.
    """
    print('Build lab_table')
    cnt = 0
    show_progress(cnt, 4)
    left = pd.read_csv(os.path.join(data_dir, 'LABEVENTS.csv'), dtype=str)
    cnt += 1
    show_progress(cnt, 4)
    right = pd.read_csv(os.path.join(data_dir, 'D_LABITEMS.csv'), dtype=str)
    cnt += 1
    show_progress(cnt, 4)
    left = left.dropna(subset=['HADM_ID', 'VALUE', 'VALUEUOM'])
    left = left.drop(columns=['ROW_ID', 'VALUENUM'])
    left['VALUE_UNIT'] = left[['VALUE', 'VALUEUOM']].apply(
        lambda x: ''.join(x), axis=1)
    left = left.drop(columns=['VALUE', 'VALUEUOM'])
    right = right.drop(columns=['ROW_ID', 'LOINC_CODE'])
    cnt += 1
    show_progress(cnt, 4)
    out = pd.merge(left, right, on='ITEMID')
    cnt += 1
    show_progress(cnt, 4)
    print()
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'LAB.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('LAB', conn, if_exists='replace', index=False)

def unzip_files():
    """Unzip all .gz files in the data directory.

    Creates an 'unzipped' directory and extracts all .gz files into it.
    """
    # Create unzipped directory if it doesn't exist
    unzipped_dir = os.path.join(data_dir, 'unzipped')
    if not os.path.exists(unzipped_dir):
        os.makedirs(unzipped_dir)
    
    # Get all .gz files
    gz_files = [f for f in os.listdir(data_dir) if f.endswith('.gz')]
    
    # Unzip each file
    for gz_file in gz_files:
        print(f"Unzipping {gz_file}...")
        gz_path = os.path.join(data_dir, gz_file)
        csv_path = os.path.join(unzipped_dir, gz_file[:-3])  # Remove .gz extension
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(csv_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        print(f"Done unzipping {gz_file}")

# Specify the path to the downloaded MIMIC III data
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mimic-iii-clinical-database-1.4')
# Path to the generated mimic.db
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# First, unzip all files
print("Starting to unzip files...")
unzip_files()
print("All files unzipped successfully!")

# Generate five tables and the database with all admissions
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.mkdir(out_dir)
conn = sqlite3.connect(os.path.join(out_dir, 'mimic_all.db'))

# Update data_dir to point to unzipped files
data_dir = os.path.join(data_dir, 'unzipped')

build_demographic_table(data_dir, out_dir, conn)
build_diagnoses_table(data_dir, out_dir, conn)
build_procedures_table(data_dir, out_dir, conn)
build_prescriptions_table(data_dir, out_dir, conn)
build_lab_table(data_dir, out_dir, conn)

'''
1. We did not emumerate all possible questions about MIMIC III.
MIMICSQL data is generated based on the patient information 
related to 100 randomly selected admissions.
2. The following codes are used for sampling the admissions 
from the large database. 
3. The parameter 'random_state=0' in line 41 will provide you 
the same set of sampled admissions and the same database as we used.
'''

print('Begin sampling ...')
# DEMOGRAPHIC
print('Processing DEMOGRAPHIC')
conn = sqlite3.connect(os.path.join(out_dir, 'mimic.db'))
data_demo = pd.read_csv(os.path.join(out_dir, "DEMOGRAPHIC.csv"))
data_demo_sample = data_demo.sample(100, random_state=0)
data_demo_sample.to_sql('DEMOGRAPHIC', conn, if_exists='replace', index=False)
sampled_id = data_demo_sample['HADM_ID'].values

# DIAGNOSES
print('Processing DIAGNOSES')
data_input = pd.read_csv(os.path.join(out_dir, "DIAGNOSES.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID==' + str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pd.concat(data_filter, ignore_index=True)
data_out.to_sql('DIAGNOSES', conn, if_exists='replace', index=False)

# PROCEDURES
print('Processing PROCEDURES')
data_input = pd.read_csv(os.path.join(out_dir, "PROCEDURES.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID==' + str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pd.concat(data_filter, ignore_index=True)
data_out.to_sql('PROCEDURES', conn, if_exists='replace', index=False)

# PRESCRIPTIONS
print('Processing PRESCRIPTIONS')
data_input = pd.read_csv(os.path.join(out_dir, "PRESCRIPTIONS.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID==' + str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pd.concat(data_filter, ignore_index=True)
data_out.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False)

# LAB
print('Processing LAB')
data_input = pd.read_csv(os.path.join(out_dir, "LAB.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID==' + str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pd.concat(data_filter, ignore_index=True)
data_out.to_sql('LAB', conn, if_exists='replace', index=False)
print('Done!')