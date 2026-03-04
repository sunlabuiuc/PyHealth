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
    EEG Abnormal <tasks/pyhealth.tasks.EEG_abnormal>
    EEG Events <tasks/pyhealth.tasks.EEG_events>
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
    Multimodal Mortality Prediction (MIMIC-IV) <tasks/pyhealth.tasks.multimodal_mimic4>
