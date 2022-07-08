# PyHealth

## Run Interactive Web
```python
python pyhealth-web/app.py # you can set the port in the script
```
## Environment
- pytorch: 1.12.0
- pytorch-lightning: 1.6.4

## Dataset
- MIMIC-III
- MIMIC-IV
- eICU
- OMOP CDM

## Input
- Condition code
- Drug code
- Procedure code

## Output
- Mortality prediction
- Length-of-stay estimation
- Drug recommendation
- Phenotyping

## Model






### datasets.py
- provide process for MIMIC-III, eICU and MIMIC-IV
- datasets.df gives the clean training data, which can be input into the task Class objects, such as tasks.DrugRec.
