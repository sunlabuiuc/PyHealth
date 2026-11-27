CDC COVID-19 Case Surveillance
================================

The CDC `COVID-19 Case Surveillance Public Use Data with Geography
<https://data.cdc.gov/Case-Surveillance/COVID-19-Case-Surveillance-Public-Use-Data-with-Ge/n8mc-b4w4>`_
contains deidentified case rows rather than longitudinal patients. Download a
CSV export as ``n8mc-b4w4.csv``. The loader assigns one generated identifier to
each physical row and never links records into people.

Unknown, missing and privacy-suppressed values are retained by the dataset.
The associated mortality task decides which outcomes are labeled and excludes
hospitalization and ICU admission from its features.

.. autoclass:: pyhealth.datasets.CDCCOVIDDataset
    :members:
    :undoc-members:
    :show-inheritance:
