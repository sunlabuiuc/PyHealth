'''
The following is a helper script for finding good test cases for the MIMIC3 readmission task.
'''
import sqlite3

import pandas as pd

admissions_df = pd.read_csv("ADMISSIONS.csv.gz")
diagnoses_df = pd.read_csv("DIAGNOSES_ICD.csv.gz")
patients_df = pd.read_csv("PATIENTS.csv.gz")
prescriptions_df = pd.read_csv("PRESCRIPTIONS.csv.gz")
procedures_df = pd.read_csv("PROCEDURES_ICD.csv.gz")

con = sqlite3.connect(":memory:")
cur = con.cursor()

admissions_df.to_sql("Admissions", con, if_exists='replace', index=False, method='multi', chunksize=10_000)
diagnoses_df.to_sql("Diagnoses", con, if_exists='replace', index=False, method='multi', chunksize=10_000)
patients_df.to_sql("Patients", con, if_exists='replace', index=False, method='multi', chunksize=10_000)
prescriptions_df.to_sql("Prescriptions", con, if_exists='replace', index=False, method='multi', chunksize=10_000)
procedures_df.to_sql("Procedures", con, if_exists='replace', index=False, method='multi', chunksize=10_000)

missing_diagnoses = """
SELECT a.hadm_id
FROM admissions AS a
WHERE
    NOT EXISTS (
        SELECT 1 FROM diagnoses d WHERE d.hadm_id = a.hadm_id
    )
AND EXISTS (
        SELECT 1 FROM prescriptions p WHERE p.hadm_id = a.hadm_id
    )
AND EXISTS (
        SELECT 1 FROM procedures pr WHERE pr.hadm_id = a.hadm_id
    )
AND a.subject_id IN (
        SELECT subject_id
        FROM admissions
        GROUP BY subject_id
        HAVING COUNT(*) > 1
    );
"""

results = cur.execute(missing_diagnoses).fetchall()

print("Visits with no diagnosis codes")
for result in results:
    print(result)
print()

missing_prescriptions = """
SELECT a.hadm_id
FROM admissions AS a
WHERE
    EXISTS (
        SELECT 1 FROM diagnoses d WHERE d.hadm_id = a.hadm_id
    )
AND NOT EXISTS (
        SELECT 1 FROM prescriptions p WHERE p.hadm_id = a.hadm_id
    )
AND EXISTS (
        SELECT 1 FROM procedures pr WHERE pr.hadm_id = a.hadm_id
    )
AND a.subject_id IN (
        SELECT subject_id
        FROM admissions
        GROUP BY subject_id
        HAVING COUNT(*) > 1
    );
"""

results = cur.execute(missing_prescriptions).fetchall()

print("Visits with no prescriptions")
for result in results:
    print(result)
print()

missing_procedures = """
SELECT a.hadm_id
FROM admissions AS a
WHERE
    EXISTS (
        SELECT 1 FROM diagnoses d WHERE d.hadm_id = a.hadm_id
    )
AND EXISTS (
        SELECT 1 FROM prescriptions p WHERE p.hadm_id = a.hadm_id
    )
AND NOT EXISTS (
        SELECT 1 FROM procedures pr WHERE pr.hadm_id = a.hadm_id
    )
AND a.subject_id IN (
        SELECT subject_id
        FROM admissions
        GROUP BY subject_id
        HAVING COUNT(*) > 1
    );
"""

results = cur.execute(missing_procedures).fetchall()

print("Visits with no procedure codes")
for result in results:
    print(result)
print()

is_minor = """
SELECT a.hadm_id
FROM admissions a
JOIN patients p
  ON p.subject_id = a.subject_id
WHERE
    (
        CAST(strftime('%Y', a.admittime) AS INTEGER)
      - CAST(strftime('%Y', p.dob) AS INTEGER)
      - (
            strftime('%m-%d', a.admittime)
          < strftime('%m-%d', p.dob)
        )
    ) < 18
AND a.subject_id IN (
        SELECT subject_id
        FROM admissions
        GROUP BY subject_id
        HAVING COUNT(*) > 1
    );
"""

results = cur.execute(is_minor).fetchall()

print("Visits where the patient is under 18 years old")
for result in results:
    print(result)
print()

con.close()


