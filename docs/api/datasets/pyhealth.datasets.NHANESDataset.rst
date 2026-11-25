pyhealth.datasets.NHANESDataset
===============================

The National Health and Nutrition Examination Survey (NHANES) provides
population-level health, demographic, examination, and physical activity
data collected in the United States.

The NHANESDataset class supports the 2003-2004 and 2005-2006 survey
cycles, combining minute-level accelerometer measurements with clinical
covariates and linked mortality records.

Activity preprocessing includes Choi-style non-wear detection,
wear-aware five-minute epoch aggregation, and participant-level
Parquet storage.

Data are publicly available from the
`NHANES website <https://wwwn.cdc.gov/nchs/nhanes/>`_.


Data Preparation
----------------

Download the following files for the 2003-2004 and 2005-2006
NHANES survey cycles:

- Demographics (DEMO)
- Body Measures (BMX)
- Blood Pressure (BPX)
- Physical Activity Monitor minute-level data (PAXRAW)
- 2019 public-use linked mortality files

The linked mortality files are available from the
`NCHS Data Linkage website
<https://www.cdc.gov/nchs/data-linkage/mortality-public.htm>`_.

Extract the activity archives and organize the files as follows:

.. code-block:: text

    nhanes/
        2003-2004/
            DEMO_C.XPT
            BMX_C.XPT
            BPX_C.XPT
            paxraw_c.xpt
        2005-2006/
            DEMO_D.XPT
            BMX_D.XPT
            BPX_D.XPT
            paxraw_d.xpt
        mortality/
            NHANES_2003_2004_MORT_2019_PUBLIC.dat
            NHANES_2005_2006_MORT_2019_PUBLIC.dat

The dataset automatically prepares participant-level activity
Parquet files and metadata when processed metadata are unavailable.

The generated files are stored under ``processed/pyhealth/``
and reused on subsequent initialization.

Preprocessing the raw accelerometer files may require substantial
disk space and processing time.

Supported Tasks
---------------

- :doc:`../tasks/pyhealth.tasks.NHANESMortalityPrediction`

Example
-------

.. code-block:: python

    from pyhealth.datasets import NHANESDataset
    from pyhealth.tasks import NHANESMortalityPrediction

    dataset = NHANESDataset(root="/path/to/nhanes")

    task = NHANESMortalityPrediction(
        root="/path/to/nhanes",
        feature_format="motionage",
        wear_representation="binary",
    )

    samples = dataset.set_task(task)

.. autoclass:: pyhealth.datasets.NHANESDataset
    :members:
    :undoc-members:
    :show-inheritance:
