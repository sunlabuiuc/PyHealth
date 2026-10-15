"""
PyHealth task for studying how imputation affects algorithmic fairness on
MIMIC-III, when which lab tests are missing reflects clinical presence: how
patients interact with the healthcare system.

The cohort is each adult's last hospital admission, kept if it lasts at least
24 hours. Features are the latest value of each of 67 lab tests drawn in the
first 24 hours, with missing values imputed at the cohort level (median, MICE,
group MICE or MissForest-style, fit on training patients only). The label is
death within 8 days of admission. Black, female and public-insurance flags
travel with each sample for subgroup evaluation, but are never features.

Dataset link:
    https://physionet.org/content/mimiciii/1.4/

Task paper: (please cite if you use this task)
    Vincent Jeanselme, Maria De-Arteaga, Zhe Zhang, Jessica Barrett and Brian
    Tom. "Imputation Strategies Under Clinical Presence: Impact on Algorithmic
    Fairness." Machine Learning for Health (ML4H) 2022, PMLR 193:12-34.

Paper link:
    https://proceedings.mlr.press/v193/jeanselme22a.html

Reference code:
    https://github.com/Jeanselme/ClinicalPresenceFairness

Authors:
    Chunyi Yu (cyu410@gatech.edu)
    John Nguyen (jnguyen461@gatech.edu)
"""

# Copy mapping from Jeanselme's preprocessing.ipynb
LAB_ITEMIDS: dict[str, tuple[int, ...]] = {
    "Alanine aminotransferase": (50861,),
    "Albumin": (50862,),
    "Albumin (ascites)": (50835,),
    "Albumin (pleural)": (51046,),
    "Albumin (urine)": (51069,),
    "Alkaline phosphate": (50863,),
    "Anion gap": (50868,),
    "Asparate aminotransferase": (50878,),
    "Basophils": (51146,),
    "Bicarbonate": (50882,),
    "Bicarbonate (whole blood)": (50803,),
    "Bilirubin (conjugated)": (50883,),
    "Bilirubin (total)": (50885,),
    "Bilirubin (unconjugated)": (50884,),
    "Blood urea nitrogen": (51006,),
    "Calcium (total)": (50893,),
    "Calcium (urine)": (51077,),
    "Calcium ionized": (50808,),
    "Chloride": (50902,),
    "Chloride (urine)": (51078,),
    "Chloride (whole blood)": (50806,),
    "Cholesterol (HDL)": (50904,),
    "Cholesterol (LDL)": (50905,),
    "Cholesterol (total)": (50907,),
    "CO2 (ETCO2, PCO2, etc.)": (50804,),
    "Creatinine": (50912,),
    "Creatinine (ascites)": (50841,),
    "Creatinine (body fluid)": (51032,),
    "Creatinine (pleural)": (51052,),
    "Creatinine (urine)": (51082,),
    "Eosinophils": (51114, 51347, 51419, 51444),
    "Fibrinogen": (51214,),
    "Glucose": (50809, 50931),
    "Hematocrit": (50810, 51221),
    "Hemoglobin": (50811, 51222),
    "Lactate": (50813,),
    "Lactate dehydrogenase": (50954, 51054),
    "Lymphocytes": (51116, 51244, 51375, 51427, 51446),
    "Lymphocytes (atypical)": (51143, 51343),
    "Lymphocytes (percent)": (51245,),
    "Magnesium": (50960,),
    "Mean corpuscular hemoglobin": (51248,),
    "Mean corpuscular hemoglobin concentration": (51249,),
    "Mean corpuscular volume": (51250,),
    "Monocytes": (51254, 51355),
    "Neutrophils": (51256,),
    "Oxygen saturation": (50817,),
    "Partial pressure of carbon dioxide": (50818,),
    "Partial thromboplastin time": (51275,),
    "pH": (50820, 50831, 51094, 51491),
    "Phosphate": (50970,),
    "Platelets": (51265,),
    "Positive end-expiratory pressure": (50819,),
    "Potassium": (50971,),
    "Potassium (whole blood)": (50822,),
    "Prothrombin time": (51237, 51274),
    "Red blood cell count": (51279, 51493),
    "Red blood cell count (ascites)": (51127,),
    "Red blood cell count (CSF)": (51362,),
    "Red blood cell count (pleural)": (51457,),
    "Sodium": (50983,),
    "Sodium (whole blood)": (50824,),
    "Tidal Volume Observed": (50826,),
    "Total Protein Urine": (51102,),
    "Troponin-I": (51002,),
    "Troponin-T": (51003,),
    "White blood cell count": (51300, 51301, 51516),
}

# LABEVENTS itemid -> lab name
ITEM_TO_LAB: dict[str, str] = {
    str(itemid): lab for lab, itemids in LAB_ITEMIDS.items() for itemid in itemids
}

LAB_NAMES: list[str] = sorted(LAB_ITEMIDS)

# Group flags, where 1 marks the marginalised group
GROUP_FLAGS: tuple[str, ...] = ("black", "female", "public_insurance")
