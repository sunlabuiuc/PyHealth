pyhealth.tasks.length_of_stay_tpc_mimic4
===========================================

Remaining ICU length-of-stay prediction task for MIMIC-IV with TPC-compatible preprocessing.

Overview
--------

The RemainingLOSMIMIC4 task generates samples for predicting remaining ICU length of stay at 
hourly intervals. Unlike traditional length-of-stay tasks that predict total stay duration at 
admission, this task formulates the problem as a time-series regression where the model predicts 
remaining hours at each timestep throughout the ICU stay.

**Input Features:**

- **Timeseries** ``(2F+2, T)``: Hourly clinical measurements with:
  
  - Elapsed time channel (1)
  - Feature values from chartevents and labevents (F channels)
  - Decay indicators showing time since last measurement (F channels)
  - Hour of day (1 channel)

- **Static** ``(2,)``: Patient demographics (age, sex)

- **Conditions**: ICD diagnosis codes from admission

**Output:**

- **Remaining LoS** ``(T,)``: Remaining hours in ICU at each timestep


**Default Configuration:**

- Prediction step size: 1 hour
- Minimum history: 5 hours before predictions start
- Minimum remaining stay: 1 hour
- Maximum history window: 366 hours (15.25 days)
- Clinical features: 17 chartevents + 17 labevents = 34 features

**Clinical Features:**

*Vital Signs (chartevents):*

- Heart rate, blood pressure (systolic/diastolic/mean)
- Respiratory rate, SpO2, temperature
- Glasgow Coma Scale components
- Urine output, weight

*Laboratory Values (labevents):*

- Hematology: WBC, platelets, hemoglobin, hematocrit
- Chemistry: sodium, potassium, chloride, bicarbonate
- Renal: BUN, creatinine
- Metabolic: glucose, lactate
- Liver: bilirubin, ALT
API Reference
-------------

.. autoclass:: pyhealth.tasks.length_of_stay_tpc_mimic4.RemainingLOSMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.tasks.length_of_stay_tpc_mimic4.RemainingLOSConfig
    :members:
    :undoc-members:
    :show-inheritance:
