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

- **Browse tasks**: `pyhealth.dev/tasks <https://pyhealth.dev/tasks.html>`_ lists every task with the dataset it runs on. Each task page below also links to its dataset, and each dataset page lists its tasks
- **Tutorial**: `Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1QB0acnGb-wOuK53UNSgHxjCW74QeYjUl?usp=sharing>`_ - Learn the basics of defining and using tasks
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

Task Views
----------

A task defines who is in the cohort, when the prediction is made, what the
label is and which measurements a model may use. How those measurements are
turned into tensors is a separate choice, and different models need
different ones. A task can therefore offer several *views*: named input
schemas over the same samples.

.. code-block:: python

    from pyhealth.datasets import PatientSplit, SplitPlan
    from pyhealth.models import RNN_V2
    from pyhealth.tasks import YAIBMortalityMIMIC4

    task = YAIBMortalityMIMIC4()               # the default view, "jagged"
    task = YAIBMortalityMIMIC4(view="dense")   # a view by name
    task = YAIBMortalityMIMIC4(model=RNN_V2)   # the view RNN_V2 reads

    plan = SplitPlan.load_or_draw(dataset, "seed0", PatientSplit((0.8, 0.1, 0.1), seed=0))
    train, val, test = dataset.set_task(task, split=plan)
    model = RNN_V2(train, ...)

What stays the same across the views of a task, and what may differ:

- **Same:** the samples ``__call__`` returns (computed once and cached once),
  the labels, and the split. A :class:`~pyhealth.datasets.SplitPlan` is a
  saved assignment of patients to parts, so every view, and every task on the
  same dataset, puts a patient in the same part.
- **May differ:** imputation, scaling, encoding and tensor layout. Anything a
  view learns from data is learned on the training part, which is why
  ``set_task`` requires ``split=`` for such a view.

A model class lists the views it reads in ``accepted_views``. ``model=``
selects the first of them the task offers, and building a model on a dataset
of another view raises an error naming the view to use. ``view=`` also takes
an input schema of your own.

The YAIB tasks offer three views. ``"jagged"`` is read by
:class:`~pyhealth.models.RNN_V2`, which also handles the per-hour labels of
the AKI, sepsis and length-of-stay tasks. ``"dense"`` gives padded tensors
with an explicit mask and is read by RNN, MultimodalRNN, TCN, Transformer,
RETAIN, MultimodalRETAIN, MultimodalAdaCare, GRASP, Agent, JambaEHR, StageNet,
MLP, LogisticRegression, KnowRare, BEHRT, BiteNet and UDC, for the tasks with
one label per stay (mortality, kidney function). Both views hold the same
values: they differ only in layout. ``"events"`` writes the same
measurements as a timeline of coded events, one per measured value and with
nothing imputed or scaled, for the EHR foundation models
(:class:`~pyhealth.models.HFEHRFoundationModel`,
:class:`~pyhealth.models.CLMBR`). Each feature's code defaults to
``"YAIB/<feature>"``; override ``codes`` (below) to write the codes a
pretrained model's vocabulary uses, and check ``model.coverage_report()`` to
see how many events it recognised.

**Your own version of a view.** The task fixes the samples, labels and
split; how a view processes them is yours to change. Change settings of a
view field by field with ``view_overrides``:

.. code-block:: python

    # no standardization of the measurements
    task = YAIBMortalityMIMIC4(
        view="dense", view_overrides={"dynamic": {"normalize": False}}
    )
    # the codes a foundation model's vocabulary uses
    task = YAIBMortalityMIMIC4(
        view="events",
        view_overrides={"dynamic": {"codes": {"crea": "LOINC/2160-0"}}},
    )

The given settings replace those of the field's processor (a nested setting,
such as the ``levels`` of ``"jagged"``, is replaced whole). A processor spec
in place of the settings, e.g. ``{"static": "raw"}``, swaps the processor.
For anything further, pass a whole input schema as ``view``; it is named
``"custom"`` and any model may be built on it. Every version has its own
processed cache and reads the task's one cache of samples.

To give your own task views, set ``views`` and ``default_view`` instead of
``input_schema``. Each task defines its views itself, as plain input schemas:

.. code-block:: python

    class MyICUTask(BaseTask):
        task_name = "MyICUTask"
        views = {
            "dense": {
                "dynamic": ("dense_series", {"forward_fill": True}),
                "static": ("dense_static", {}),
            },
            "events": {
                "dynamic": ("event_stream", {"names": ["hr", "crea"]}),
            },
        }
        default_view = "dense"
        output_schema = {"mortality": "binary"}

        def __call__(self, patient):
            ...  # emit "dynamic" and "static" with NaN where not measured

To add a named view to an existing task, subclass it and extend ``views``
(``views = {**YAIBMortalityMIMIC4.views, "mine": {...}}``). Use the view
names of the built-in tasks (``"jagged"``, ``"dense"``, ``"events"``) for
the same layouts, since models declare the names they read.

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
   * - ``"nested_multihot"``
     - ``NestedMultiHotProcessor``
     - Per-visit code sets (HALO; see ``EHRGenerationMIMIC3``)
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
PyHealth detects the existing cache and skips reprocessing. The cache key holds
the task's arguments and schemas but not its code, so after you edit a task's
``__call__``, run ``dataset.clear_task_cache(task)`` (see *Task caches and when
to clear them* in :doc:`datasets`). During
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
    Task Card <tasks/pyhealth.tasks.TaskCard>
    PBC Competing Risks <tasks/pyhealth.tasks.competing_risks_pbc>
    Competing-Risk Preprocessing <tasks/pyhealth.tasks.competing_risks>
    In-Hospital Mortality (MIMIC-IV) <tasks/pyhealth.tasks.InHospitalMortalityMIMIC4>
    In-Hospital Mortality (MEDS) <tasks/pyhealth.tasks.InHospitalMortalityMEDS>
    DearLLM Mortality (MIMIC-III/IV) <tasks/pyhealth.tasks.dearllm_mortality>
    ICU Admission (MEDS) <tasks/pyhealth.tasks.ICUAdmissionMEDS>
    Hospital Readmission (MEDS) <tasks/pyhealth.tasks.HospitalReadmissionMEDS>
    Prolonged Length of Stay (MEDS) <tasks/pyhealth.tasks.ProlongedLengthOfStayMEDS>
    ETHOS Timelines (MEDS) <tasks/pyhealth.tasks.EthosTimelineMEDS>
    ETHOS Zero-Shot Outcomes (MEDS) <tasks/pyhealth.tasks.EthosOutcomeMEDS>
    MIMIC-III ICD-9 Coding <tasks/pyhealth.tasks.MIMIC3ICD9Coding>
    BMD-HS Disease Classification <tasks/pyhealth.tasks.bmd_hs_disease_classification>
    Cardiology Detection <tasks/pyhealth.tasks.cardiology_detect>
    COVID-19 CXR Classification <tasks/pyhealth.tasks.COVID19CXRClassification>
    COVID-19 CXR Image Generation <tasks/pyhealth.tasks.COVID19CXRImageGeneration>
    PTB-XL Multilabel Classification <tasks/pyhealth.tasks.PTBXLMultilabelClassification>
    PTB-XL HeartLang Classification <tasks/pyhealth.tasks.PTBXLHeartLangClassification>
    Chapman-Shaoxing ECG Classification <tasks/pyhealth.tasks.ChapmanShaoxingSuperclassClassification>
    PhysioNet 2012 Forecasting <tasks/pyhealth.tasks.PhysioNet2012Forecasting>
    Disease Progression (MIMIC-IV + Chest ImaGenome) <tasks/pyhealth.tasks.DiseaseProgressionMIMIC4>
    DKA Prediction (MIMIC-IV) <tasks/pyhealth.tasks.dka>
    Drug Recommendation <tasks/pyhealth.tasks.drug_recommendation>
    EHR + CXR In-Hospital Mortality (MIMIC-IV) <tasks/pyhealth.tasks.EHRCXRMortalityMIMIC4>
    EHR + CXR Phenotyping (MIMIC-IV) <tasks/pyhealth.tasks.EHRCXRPhenotypingMIMIC4>
    EHR + CXR Decompensation (MIMIC-IV, MedMod) <tasks/pyhealth.tasks.MedModDecompensationMIMIC4>
    EHR Generation <tasks/pyhealth.tasks.generate_ehr>
    Length of Stay Prediction <tasks/pyhealth.tasks.length_of_stay_prediction>
    Length of Stay Prediction (StageNet MIMIC-IV) <tasks/pyhealth.tasks.length_of_stay_stagenet_mimic4>
    Medical Transcriptions Classification <tasks/pyhealth.tasks.MedicalTranscriptionsClassification>
    MPF Clinical Prediction (FHIR) <tasks/pyhealth.tasks.mpf_clinical_prediction>
    Mortality Prediction (Next Visit) <tasks/pyhealth.tasks.mortality_prediction>
    Mortality Prediction (StageNet MIMIC-IV) <tasks/pyhealth.tasks.mortality_prediction_stagenet_mimic4>
    Next-Visit Diagnosis Prediction <tasks/pyhealth.tasks.next_visit_diagnosis_prediction>
    Patient Linkage (MIMIC-III, legacy) <tasks/pyhealth.tasks.patient_linkage_mimic3_fn>
    Patient Linkage (MIMIC-III) <tasks/pyhealth.tasks.patient_linkage_mimic3>
    Readmission Prediction <tasks/pyhealth.tasks.readmission_prediction>
    Sleep Staging <tasks/pyhealth.tasks.sleep_staging>
    Sleep Staging (SleepEDF) <tasks/pyhealth.tasks.SleepStagingSleepEDF>
    Sleep Staging Sequences (SleepEDF) <tasks/pyhealth.tasks.SleepStagingSequenceSleepEDF>
    Sleep Staging (ISRUC) <tasks/pyhealth.tasks.SleepStagingISRUC>
    Sleep Staging Sequences (ISRUC) <tasks/pyhealth.tasks.SleepStagingSequenceISRUC>
    Seizure Onset Zone Localization (HUP iEEG) <tasks/pyhealth.tasks.SOZLocalizationHUP>
    Temple University EEG Tasks <tasks/pyhealth.tasks.temple_university_EEG_tasks>
    CHB-MIT Seizure Detection <tasks/pyhealth.tasks.SeizureDetectionCHBMIT>
    Helsinki Neonatal Seizure Detection <tasks/pyhealth.tasks.SeizureDetectionHelsinki>
    Ear-EEG Sleep Staging (EESM23) <tasks/pyhealth.tasks.SleepStagingEarEEG>
    IIIC Seizure Classification <tasks/pyhealth.tasks.iiic_seizure_classification>
    EEGBCI Tasks <tasks/pyhealth.tasks.eegbci>
    BCI Competition IV 2a Motor Imagery <tasks/pyhealth.tasks.BCIIV2aMotorImageryClassification>
    BCI Competition IV 2a Left vs Right Hand <tasks/pyhealth.tasks.BCIIV2aLeftRightImagery>
    PTB-XL Tasks <tasks/pyhealth.tasks.ptbxl>
    Sleep Staging v2 <tasks/pyhealth.tasks.sleep_staging_v2>
    Benchmark EHRShot <tasks/pyhealth.tasks.benchmark_ehrshot>
    EHR Text Serialization (LLM encoders) <tasks/pyhealth.tasks.ehr_text_serialization>
    EHR Foundation Model Timelines <tasks/pyhealth.tasks.ehr_foundation_model>
    ED Disposition and Outcomes (MIMIC-IV-ED) <tasks/pyhealth.tasks.ed_disposition_mimic4>
    ChestX-ray14 Binary Classification <tasks/pyhealth.tasks.ChestXray14BinaryClassification>
    ISIC 2019 Skin Lesion Classification <tasks/pyhealth.tasks.ISIC2019Classification>
    De-Identification NER <tasks/pyhealth.tasks.DeIDNERTask>
    ChestX-ray14 Multilabel Classification <tasks/pyhealth.tasks.ChestXray14MultilabelClassification>
    HAM10000 Skin Lesion Classification <tasks/pyhealth.tasks.HAM10000Classification>
    MSD Segmentation <tasks/pyhealth.tasks.MSDSegmentationTask>
    CHAOS Segmentation Tasks <tasks/pyhealth.tasks.CHAOSSegmentation>
    RSNA Pneumonia Classification <tasks/pyhealth.tasks.RSNAPneumoniaClassification>
    PAD-UFES-20 Skin Lesion Classification <tasks/pyhealth.tasks.PADUFES20Classification>
    ISIC 2020 Melanoma Classification <tasks/pyhealth.tasks.ISIC2020MelanomaClassification>
    CBIS-DDSM Mammography Classification <tasks/pyhealth.tasks.cbis_ddsm_classification>
    CBIS-DDSM Abnormality Type Classification <tasks/pyhealth.tasks.CBISDDSMAbnormalityClassification>
    CBIS-DDSM Breast Density Classification <tasks/pyhealth.tasks.CBISDDSMDensityClassification>
    Variant Classification (ClinVar) <tasks/pyhealth.tasks.VariantClassificationClinVar>
    Jichi Davis DR Grading <tasks/pyhealth.tasks.JichiDavisGrading>
    Mutation Pathogenicity (COSMIC) <tasks/pyhealth.tasks.MutationPathogenicityPrediction>
    LNDb Finding Classification <tasks/pyhealth.tasks.LNDbFindingClassification>
    LNDb Texture Classification <tasks/pyhealth.tasks.LNDbTextureClassification>
    Cancer Survival Prediction (TCGA) <tasks/pyhealth.tasks.CancerSurvivalPrediction>
    In-Hospital Survival (MIMIC-IV) <tasks/pyhealth.tasks.InHospitalSurvivalMIMIC4>
    BCSS Tile Classification <tasks/pyhealth.tasks.BCSSTileClassification>
    BCSS Tissue Segmentation <tasks/pyhealth.tasks.BCSSTissueSegmentation>
    METABRIC Survival Prediction <tasks/pyhealth.tasks.MetabricSurvivalPrediction>
    ABIDE Autism Classification <tasks/pyhealth.tasks.ABIDEAutismClassification>
    Rare-Condition ICU Mortality (KnowRare, MIMIC-IV) <tasks/pyhealth.tasks.KnowRareICUMortalityMIMIC4>
    KiTS23 Malignancy Classification <tasks/pyhealth.tasks.KiTS23MalignancyClassification>
    KiTS23 Segmentation <tasks/pyhealth.tasks.KiTS23Segmentation>
    Lab Nowcasting (MIMIC-III/IV) <tasks/pyhealth.tasks.LabNowcasting>
    COVID-BLUES Frame Classification <tasks/pyhealth.tasks.COVIDBLUESFrameClassification>
    COVID-BLUES Severity Classification <tasks/pyhealth.tasks.COVIDBLUESSeverityClassification>
    ICU Acuity and Therapies (MIMIC-IV) <tasks/pyhealth.tasks.ICUAcuityMIMIC4>
    Brain Tumor MRI Classification <tasks/pyhealth.tasks.BrainTumorMRIClassification>
    Cancer Mutation Burden (TCGA) <tasks/pyhealth.tasks.CancerMutationBurden>
    Survival Preprocessing (SUPPORT2) <tasks/pyhealth.tasks.SurvivalPreprocessSupport2>
    YAIB Tasks (MIMIC-IV) <tasks/pyhealth.tasks.yaib_mimic4>
    YAIB Tasks (MIMIC-III) <tasks/pyhealth.tasks.yaib_mimic3>
    YAIB Tasks (eICU) <tasks/pyhealth.tasks.yaib_eicu>
    Value Set Completion (VSAC) <tasks/pyhealth.tasks.ValueSetCompletion>
    Time-to-event survival (GBSG and SUPPORT) <tasks/pyhealth.tasks.survival_time_to_event>
    TCGA RNA-seq Survival Prediction <tasks/pyhealth.tasks.TCGASurvivalPrediction>
