Welcome to PyHealth!
====================================

.. note::

   **Documentation:** The official documentation site will be released when it is no longer double blind.


-----------------------------------------------------------------


PyHealth is a comprehensive deep learning toolkit for supporting clinical predictive modeling, which is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to develop, test, and deploy—more flexible and more customizable.

**Key Features**

- Modular 5-stage pipeline for healthcare ML
- Healthcare-first: medical codes and clinical datasets (MIMIC, eICU, OMOP)
- 33+ pre-built models and production-ready trainer/metrics
- 10+ supported healthcare tasks and datasets
- Fast (~3x faster than pandas) data processing for quick experimentation

 **[News!]** We are continuously implementing good papers and benchmarks into PyHealth. Welcome to send us a PR with influential and new papers.

.. image:: figure/poster.png
   :width: 810

..

1. Installation :rocket:
----------------------------

**Python Version Requirement**

PyHealth requires **Python 3.12 or 3.13** (``>=3.12,<3.14``). This version requirement enables optimal parallel processing, memory management, and compatibility with our modern dependencies.

**Installation**

Install PyHealth from PyPI:

.. code-block:: sh

    pip install pyhealth

**For Contributors and Developers**

If you are contributing to PyHealth or need the latest development features, install from GitHub source:

.. code-block:: sh

    git clone REDACTED_GITHUB_URL
    cd PyHealth
    pip install -e .

**Note:** PyHealth automatically installs PyTorch and other deep learning dependencies. However, users may need to reinstall PyTorch based on their CUDA versions for optimal GPU support.


2. Introduction :book:
--------------------------
``pyhealth`` provides these functionalities (we are still enriching some modules):

.. image:: figure/overview.png
   :width: 770

You can use the following functions independently:

- **Dataset**: ``MIMIC-III``, ``MIMIC-IV``, ``eICU``, ``OMOP-CDM``, ``EHRShot``, ``COVID19-CXR``, ``SleepEDF``, ``SHHS``, ``ISRUC``, ``customized EHR datasets``, etc.
- **Tasks**: ``diagnosis-based drug recommendation``, ``patient hospitalization and mortality prediction``, ``readmission prediction``, ``length of stay forecasting``, ``sleep staging``, etc.
- **ML models**: ``RNN``, ``LSTM``, ``GRU``, ``Transformer``, ``RETAIN``, ``SafeDrug``, ``GAMENet``, ``MoleRec``, ``AdaCare``, ``ConCare``, ``StageNet``, ``GRASP``, ``SparcNet``, ``ContraWR``, ``Deepr``, ``TCN``, ``Dr. Agent``, etc.

*Building a healthcare AI pipeline can be as short as 10 lines of code in PyHealth*.


3. Build ML Pipelines :trophy:
---------------------------------

All healthcare tasks in our package follow a **five-stage pipeline**:

.. image:: figure/five-stage-pipeline.png
   :width: 640

..

 We try hard to make sure each stage is as separate as possible, so that people can customize their own pipeline by only using our data processing steps or the ML models.

Module 1: <pyhealth.datasets>
""""""""""""""""""""""""""""""""""""

``pyhealth.datasets`` provides a clean structure for the dataset, independent from the tasks. We support `MIMIC-III`, `MIMIC-IV`, `eICU`, `OMOP-CDM`, and more. The output (mimic3base) is a multi-level dictionary structure (see illustration below).

.. code-block:: python

    from pyhealth.datasets import MIMIC3Dataset

    mimic3base = MIMIC3Dataset(
        # root directory of the dataset
        root="REDACTED_STORAGE_URL/Synthetic_MIMIC-III/",
        # raw CSV table name
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        # map all NDC codes to CCS codes in these tables
        code_mapping={"NDC": "CCSCM"},
    )

.. image:: figure/structured-dataset.png
   :width: 400

..

Module 2: <pyhealth.tasks>
""""""""""""""""""""""""""""""""""""

``pyhealth.tasks`` defines how to process each patient's data into a set of samples for the tasks. In the package, we provide several task examples, such as ``drug recommendation``, ``mortality prediction``, and ``readmission prediction``. **It is easy to customize your own tasks following our template** (documentation will be released when no longer double blind).

.. code-block:: python

    from pyhealth.tasks import ReadmissionPredictionMIMIC3

    mimic3sample = mimic3base.set_task(ReadmissionPredictionMIMIC3())
    mimic3sample[0] # show the information of the first sample

    from pyhealth.datasets import split_by_patient, get_dataloader

    train_ds, val_ds, test_ds = split_by_patient(mimic3sample, [0.8, 0.1, 0.1])
    train_loader = get_dataloader(train_ds, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_ds, batch_size=32, shuffle=False)

Module 3: <pyhealth.models>
""""""""""""""""""""""""""""""""""""

``pyhealth.models`` provides different ML models with very similar argument configs.

.. code-block:: python

    from pyhealth.models import Transformer

    model = Transformer(
        dataset=mimic3sample,
    )

Module 4: <pyhealth.trainer>
""""""""""""""""""""""""""""""""""""

``pyhealth.trainer`` can specify training arguments, such as epochs, optimizer, learning rate, etc. The trainer will automatically save the best model and output the path in the end.

.. code-block:: python

    from pyhealth.trainer import Trainer

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=50,
        monitor="pr_auc_samples",
    )

Module 5: <pyhealth.metrics>
""""""""""""""""""""""""""""""""""""

``pyhealth.metrics`` provides several **common evaluation metrics** (documentation will be released when no longer double blind).

.. code-block:: python

    # method 1
    trainer.evaluate(test_loader)

    # method 2
    from pyhealth.metrics.binary import binary_metrics_fn

    y_true, y_prob, loss = trainer.inference(test_loader)
    binary_metrics_fn(y_true, y_prob, metrics=["pr_auc", "roc_auc"])

4. Medical Code Map :hospital:
---------------------------------

``pyhealth.codemap`` provides two core functionalities. **This module can be used independently.**

* For code ontology lookup within one medical coding system (e.g., name, category, sub-concept);

.. code-block:: python

    from pyhealth.medcode import InnerMap

    icd9cm = InnerMap.load("ICD9CM")
    icd9cm.lookup("428.0")
    # `Congestive heart failure, unspecified`
    icd9cm.get_ancestors("428.0")
    # ['428', '420-429.99', '390-459.99', '001-999.99']

    atc = InnerMap.load("ATC")
    atc.lookup("M01AE51")
    # `ibuprofen, combinations`
    atc.lookup("M01AE51", "drugbank_id")
    # `DB01050`
    atc.lookup("M01AE51", "description")
    # Ibuprofen is a non-steroidal anti-inflammatory drug (NSAID) derived ...
    atc.lookup("M01AE51", "indication")
    # Ibuprofen is the most commonly used and prescribed NSAID. It is very common over the ...

* For code mapping between two coding systems (e.g., ICD9CM to CCSCM).

.. code-block:: python

    from pyhealth.medcode import CrossMap

    codemap = CrossMap.load("ICD9CM", "CCSCM")
    codemap.map("428.0")
    # ['108']

    codemap = CrossMap.load("NDC", "RxNorm")
    codemap.map("50580049698")
    # ['209387']

    codemap = CrossMap.load("NDC", "ATC")
    codemap.map("50090539100")
    # ['A10AC04', 'A10AD04', 'A10AB04']

5. Medical Code Tokenizer :speech_balloon:
---------------------------------------------

``pyhealth.tokenizer`` is used for transformations between string-based tokens and integer-based indices, based on the overall token space. We provide flexible functions to tokenize 1D, 2D and 3D lists. **This module can be used independently.**

.. code-block:: python

    from pyhealth.tokenizer import Tokenizer

    # Example: we use a list of ATC3 code as the token
    token_space = ['A01A', 'A02A', 'A02B', 'A02X', 'A03A', 'A03B', 'A03C', 'A03D', \
            'A03F', 'A04A', 'A05A', 'A05B', 'A05C', 'A06A', 'A07A', 'A07B', 'A07C', \
            'A12B', 'A12C', 'A13A', 'A14A', 'A14B', 'A16A']
    tokenizer = Tokenizer(tokens=token_space, special_tokens=["<pad>", "<unk>"])

    # 2d encode
    tokens = [['A03C', 'A03D', 'A03E', 'A03F'], ['A04A', 'B035', 'C129']]
    indices = tokenizer.batch_encode_2d(tokens)
    # [[8, 9, 10, 11], [12, 1, 1, 0]]

    # 2d decode
    indices = [[8, 9, 10, 11], [12, 1, 1, 0]]
    tokens = tokenizer.batch_decode_2d(indices)
    # [['A03C', 'A03D', 'A03E', 'A03F'], ['A04A', '<unk>', '<unk>']]

    # 3d encode
    tokens = [[['A03C', 'A03D', 'A03E', 'A03F'], ['A08A', 'A09A']], \
        [['A04A', 'B035', 'C129']]]
    indices = tokenizer.batch_encode_3d(tokens)
    # [[[8, 9, 10, 11], [24, 25, 0, 0]], [[12, 1, 1, 0], [0, 0, 0, 0]]]

    # 3d decode
    indices = [[[8, 9, 10, 11], [24, 25, 0, 0]], \
        [[12, 1, 1, 0], [0, 0, 0, 0]]]
    tokens = tokenizer.batch_decode_3d(indices)
    # [[['A03C', 'A03D', 'A03E', 'A03F'], ['A08A', 'A09A']], [['A04A', '<unk>', '<unk>']]]
..

7. Datasets :mountain_snow:
-----------------------------
We provide the processing files for the following open EHR datasets:

===================  =======================================  ========================================  ========================================================================================================
MIMIC-III            ``pyhealth.datasets.MIMIC3Dataset``      2016                                      `MIMIC-III Clinical Database <https://physionet.org/content/mimiciii/1.4//>`_
MIMIC-IV             ``pyhealth.datasets.MIMIC4Dataset``      2020                                      `MIMIC-IV Clinical Database <https://physionet.org/content/mimiciv/0.4/>`_
eICU                 ``pyhealth.datasets.eICUDataset``        2018                                      `eICU Collaborative Research Database <https://eicu-crd.mit.edu//>`_
OMOP                 ``pyhealth.datasets.OMOPDataset``                                                  `OMOP-CDM schema based dataset <https://www.ohdsi.org/data-standardization/the-common-data-model/>`_
EHRShot              ``pyhealth.datasets.EHRShotDataset``     2023                                      `Few-shot EHR benchmarking dataset <https://github.com/som-shahlab/ehrshot-benchmark>`_
COVID19-CXR          ``pyhealth.datasets.COVID19CXRDataset``  2020                                      `COVID-19 chest X-ray image dataset`
SleepEDF             ``pyhealth.datasets.SleepEDFDataset``    2018                                      `Sleep-EDF dataset <https://physionet.org/content/sleep-edfx/1.0.0/>`_
SHHS                 ``pyhealth.datasets.SHHSDataset``        2016                                      `Sleep Heart Health Study dataset <https://sleepdata.org/datasets/shhs>`_
ISRUC                ``pyhealth.datasets.ISRUCDataset``       2016                                      `ISRUC-SLEEP dataset <https://sleeptight.isr.uc.pt/?page_id=48>`_
===================  =======================================  ========================================  ========================================================================================================


8. Machine/Deep Learning Models :airplane:
------------------------------------------------------------

**Deep Learning Models**

==================================    ======  ============================================================================================================
Model                                 Year    Key Innovation
==================================    ======  ============================================================================================================
**RETAIN**                            2016    Interpretable attention for clinical decisions
**GAMENet**                           2019    Memory networks for drug recommendation
**SafeDrug**                          2021    Molecular graphs for safe drug combinations
**MoleRec**                           2023    Substructure-aware drug recommendation
**AdaCare**                           2020    Scale-adaptive feature extraction
**ConCare**                           2020    Transformer-based patient modeling
**StageNet**                          2020    Disease progression stage modeling
**GRASP**                             2021    Graph neural networks for patient clustering
**MICRON**                            2021    Medication change prediction with recurrent residual networks
==================================    ======  ============================================================================================================

**Foundation Models**

==================================    ======  ============================================================================================================
Model                                 Year    Description
==================================    ======  ============================================================================================================
**Transformer**                       2017    Attention-based sequence modeling
**RNN/LSTM/GRU**                      2011    Recurrent neural networks for sequences
**CNN**                               1989    Convolutional networks for structured data
**TCN**                               2018    Temporal convolutional networks
**MLP**                               1986    Multi-layer perceptrons for tabular data
==================================    ======  ============================================================================================================

**Specialized Models**

==================================    ======  ============================================================================================================
Model                                 Year    Specialization
==================================    ======  ============================================================================================================
**ContraWR**                          2021    Biosignal analysis (EEG, ECG)
**SparcNet**                          2023    Seizure detection and sleep staging
**Deepr**                             2017    Electronic health records
**Dr. Agent**                         2020    Reinforcement learning for clinical decisions
==================================    ======  ============================================================================================================

* Check the interactive map on benchmark EHR predictive tasks (documentation will be released when no longer double blind).


9. Research Initiative :microscope:
--------------------------------------------

The **PyHealth Research Initiative** is a year-round, open research program that brings together talented individuals from diverse backgrounds to conduct cutting-edge research in healthcare AI.

**How to participate:**

1. Submit a high-quality PR to the PyHealth repository
2. Check the documentation for more details (will be released when no longer double blind)


10. About Us :busts_in_silhouette:
--------------------------------------------

We are the REDACTED_LAB healthcare research team at REDACTED_INSTITUTION.

**Current Maintainers:**

- REDACTED_CONTRIBUTOR_1 (REDACTED_ROLE @ REDACTED_INSTITUTION)
- REDACTED_CONTRIBUTOR_2 (REDACTED_ROLE @ REDACTED_INSTITUTION)
- REDACTED_CONTRIBUTOR_3 (REDACTED_ROLE @ REDACTED_INSTITUTION_2)
- REDACTED_CONTRIBUTOR_4 (REDACTED_ROLE @ REDACTED_INSTITUTION)

**Get in Touch:**

- GitHub Issues (will be released when no longer double blind)

