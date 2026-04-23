Tasks
===============

We support various real-world healthcare predictive tasks defined by **function calls**. The following example tasks are collected from top AI/Medical venues, such as:

(i) Drug Recommendation [Yang et al. IJCAI 2021a, Yang et al. IJCAI 2021b, Shang et al. AAAI 2020]

(ii) Readmission Prediction [Choi et al. AAAI 2021]

(iii) Mortality Prediction [Choi et al. AAAI 2021]

(iv) Length of Stay Prediction

(v) Sleep Staging [Yang et al. ArXix 2021]

Getting Started
---------------

New to PyHealth tasks? Start here:

- **Tutorial**: `Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1kKkkBVS_GclHoYTbnOtjyYnSee79hsyT?usp=sharing>`_ - Learn the basics of defining and using tasks
- **Code Examples**: Browse all examples online at https://github.com/sunlabuiuc/PyHealth/tree/master/examples
- **Pipeline Examples**: Check out our :doc:`../tutorials` page for complete end-to-end examples including:

  - Mortality Prediction Pipeline
  - Readmission Prediction Pipeline
  - Medical Coding Pipeline
  - Chest X-ray Classification Pipeline

These tutorials demonstrate how to load datasets, apply tasks, train models, and evaluate results.

Understanding Tasks and Processors
-----------------------------------

Tasks define **what** data to extract (via ``input_schema`` and ``output_schema``), while **processors** define **how** to transform that data into tensors for model training.

After you define a task:

1. **Task execution**: The task function extracts relevant features from patient records and generates samples
2. **Processor application**: Processors automatically transform these samples into model-ready tensors based on the schemas

**Example workflow:**

.. code-block:: python

    # 1. Define a task with input/output schemas
    task = MortalityPredictionMIMIC4()
    # input_schema = {"conditions": "sequence", "procedures": "sequence"}
    # output_schema = {"mortality": "binary"}

    # 2. Apply task to dataset
    sample_dataset = base_dataset.set_task(task)

    # 3. Processors automatically transform samples:
    #    - "sequence" -> SequenceProcessor (converts codes to indices)
    #    - "binary" -> BinaryLabelProcessor (converts labels to tensors)

    # 4. Get model-ready tensors
    sample = sample_dataset[0]
    # sample["conditions"] is now a tensor of token indices
    # sample["mortality"] is now a binary tensor [0] or [1]

**Learn more about processors:**

- See the :doc:`processors` documentation for details on all available processors
- Learn about string keys (``"sequence"``, ``"binary"``, etc.) that map to specific processors
- Discover how to customize processor behavior with kwargs tuples
- Understand processor types for different data modalities (text, images, signals, etc.)

Writing a Custom Task
----------------------

When a built-in task doesn't match your cohort or prediction target, you can
define your own by subclassing :class:`~pyhealth.tasks.BaseTask`. The class
needs three things: a name, input and output schemas, and a ``__call__``
method that processes one patient at a time.

.. code-block:: python

    from pyhealth.tasks import BaseTask
    from pyhealth.data import Patient
    from typing import List, Dict, Any

    class MyMortalityTask(BaseTask):
        task_name: str = "MyMortalityTask"

        input_schema: Dict[str, str] = {
            "conditions": "sequence",   # maps to SequenceProcessor
            "procedures": "sequence",
        }
        output_schema: Dict[str, str] = {
            "label": "binary"           # maps to BinaryLabelProcessor
        }

        def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
            samples = []
            for adm in patient.get_events("admissions"):
                label = 1 if adm.hospital_expire_flag == "1" else 0

                # Fetch historical diagnoses up to this admission
                conditions = patient.get_events("diagnoses_icd", end=adm.timestamp)
                cond_codes = [e.icd_code for e in conditions]

                if not cond_codes:
                    continue

                samples.append({
                    "conditions": [cond_codes],  # wrapped in a list for the sequence processor
                    "procedures": [[]],
                    "label": label,
                })
            return samples

The ``__call__`` method receives one ``Patient`` and returns a list of sample
dictionaries. Each dictionary's keys should match the schemas you declared.
Returning an empty list is fine — PyHealth simply skips that patient. Note
that event attribute names are always lowercase (e.g. ``e.icd_code`` rather
than ``e.ICD_CODE``) because PyHealth lowercases all column names at ingest
time. Timestamps are accessed through ``event.timestamp`` rather than the
original column name like ``charttime``, since PyHealth normalises them into
a single property.

Processor String Keys
----------------------

The string values in your schemas map to specific processor classes. Here is
a quick reference:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - String key
     - Processor
     - Typical use
   * - ``"sequence"``
     - ``SequenceProcessor``
     - Diagnosis codes, procedure codes, drug names
   * - ``"nested_sequence"``
     - ``NestedSequenceProcessor``
     - Cumulative visit history (drug recommendation, readmission)
   * - ``"tensor"``
     - ``TensorProcessor``
     - Aggregated numeric values (e.g. last lab value per item)
   * - ``"timeseries"``
     - ``TimeseriesProcessor``
     - Irregular time-series measurements
   * - ``"multi_hot"``
     - ``MultiHotProcessor``
     - Demographics, comorbidity flags
   * - ``"text"``
     - ``TextProcessor``
     - Clinical notes
   * - ``"binary"``
     - ``BinaryLabelProcessor``
     - Binary classification label (0 / 1)
   * - ``"multiclass"``
     - ``MultiClassLabelProcessor``
     - Multi-class label
   * - ``"multilabel"``
     - ``MultiLabelProcessor``
     - Multi-label classification
   * - ``"regression"``
     - ``RegressionLabelProcessor``
     - Continuous regression target

How set_task() Works
---------------------

Calling ``dataset.set_task(task)`` iterates over every patient in the
dataset, runs your ``__call__`` method on each one, fits all the processors
on the collected samples, then serialises everything to disk as LitData
``.ld`` files. The result is a :class:`~pyhealth.datasets.SampleDataset` that
supports ``len()`` and index access, ready for a DataLoader.

.. code-block:: python

    sample_ds = dataset.set_task(MyMortalityTask(), num_workers=4)
    len(sample_ds)   # total ML samples across all patients
    sample_ds[0]     # a single sample dict with tensor values

If you re-run ``set_task()`` with the same task and processor configuration,
PyHealth detects the existing cache and skips reprocessing. During
development it is useful to set ``dev=True`` on the dataset, which limits
processing to 1 000 patients so iterations are fast.

.. note::

   **A note on multiprocessing.** ``set_task()`` can spawn worker processes
   when ``num_workers > 1``. On macOS and Linux this requires the standard
   Python multiprocessing guard around your top-level script:

   .. code-block:: python

       if __name__ == '__main__':
           sample_ds = dataset.set_task(task, num_workers=4)

   Without this guard, Python may try to re-import and re-run the script in
   each worker process, leading to infinite recursion. This is a general
   Python multiprocessing requirement, not specific to PyHealth.

Available Tasks
---------------

.. toctree::
    :maxdepth: 3

    Base Task <tasks/pyhealth.tasks.BaseTask>
    In-Hospital Mortality (MIMIC-IV) <tasks/pyhealth.tasks.InHospitalMortalityMIMIC4>
    MIMIC-III ICD-9 Coding <tasks/pyhealth.tasks.MIMIC3ICD9Coding>
    Cardiology Detection <tasks/pyhealth.tasks.cardiology_detect>
    COVID-19 CXR Classification <tasks/pyhealth.tasks.COVID19CXRClassification>
    DKA Prediction (MIMIC-IV) <tasks/pyhealth.tasks.dka>
    Drug Recommendation <tasks/pyhealth.tasks.drug_recommendation>
    Length of Stay Prediction <tasks/pyhealth.tasks.length_of_stay_prediction>
    Medical Transcriptions Classification <tasks/pyhealth.tasks.MedicalTranscriptionsClassification>
    Mortality Prediction (Next Visit) <tasks/pyhealth.tasks.mortality_prediction>
    Mortality Prediction (StageNet MIMIC-IV) <tasks/pyhealth.tasks.mortality_prediction_stagenet_mimic4>
    Patient Linkage (MIMIC-III) <tasks/pyhealth.tasks.patient_linkage_mimic3_fn>
    Readmission Prediction <tasks/pyhealth.tasks.readmission_prediction>
    Sleep Staging <tasks/pyhealth.tasks.sleep_staging>
    Sleep Staging (SleepEDF) <tasks/pyhealth.tasks.SleepStagingSleepEDF>
    Temple University EEG Tasks <tasks/pyhealth.tasks.temple_university_EEG_tasks>
    Sleep Staging v2 <tasks/pyhealth.tasks.sleep_staging_v2>
    Benchmark EHRShot <tasks/pyhealth.tasks.benchmark_ehrshot>
    ChestX-ray14 Binary Classification <tasks/pyhealth.tasks.ChestXray14BinaryClassification>
    ChestX-ray14 Multilabel Classification <tasks/pyhealth.tasks.ChestXray14MultilabelClassification>
    Variant Classification (ClinVar) <tasks/pyhealth.tasks.VariantClassificationClinVar>
    Mutation Pathogenicity (COSMIC) <tasks/pyhealth.tasks.MutationPathogenicityPrediction>
    Cancer Survival Prediction (TCGA) <tasks/pyhealth.tasks.CancerSurvivalPrediction>
    Cancer Mutation Burden (TCGA) <tasks/pyhealth.tasks.CancerMutationBurden>
    Remaining Length of Stay (TPC, MIMIC-IV) <tasks/pyhealth.tasks.RemainingLengthOfStayTPC_MIMIC4>
