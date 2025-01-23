# import pandas as pd
# from pyhealth.datasets import MIMIC3Dataset

# #converted the sample MIMIC-III data to usable format for PyHealth MIMIC3Dataset function
# files = ['PATIENTS', 'ADMISSIONS','DIAGNOSES_ICD', 'PROCEDURES_ICD', 'LABEVENTS']

# #script to change lowercase col names to uppercase
# for f in files:
#   df = pd.read_csv(f'MIMIC-III/{f}.csv')
#   cols = df.columns.tolist()
#   for c in cols:
#     df[c.upper()] = df[c]
#     df = df.drop(columns=[c])
#   #save the processed data in csv file
#   df.to_csv(f'Final_MIMIC-III/{f}.csv', index=False)