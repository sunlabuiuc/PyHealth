Welcome to PyHealth!
====================================

.. note::

   **This README may be out of date.** For the most up-to-date documentation, tutorials, and API reference, please visit our official documentation site at `pyhealth.readthedocs.io <https://pyhealth.readthedocs.io/en/latest/>`_.

.. important::

   * **Join our PyHealth Discord Community!** We are actively looking for contributors and want to get to know our users better! `Click here to join Discord <https://discord.gg/mpb835EHaX>`_
   * **Signup for our mailing list!** We will email any significant PyHealth changes that are soon to come! `Click here to subscribe <https://docs.google.com/forms/d/e/1FAIpQLSfpJB5tdkI7BccTCReoszV9cyyX2rF99SgznzwlOepi5v-xLw/viewform?usp=header>`_

.. image:: https://img.shields.io/readthedocs/pyhealth?logo=readthedocs&label=docs&version=latest
   :target: https://pyhealth.readthedocs.io/en/latest/
   :alt: Docs

.. image:: https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white
   :target: https://discord.gg/mpb835EHaX
   :alt: Discord

.. image:: https://img.shields.io/badge/Mailing%20List-Subscribe-blue?logo=gmail&logoColor=white
   :target: https://docs.google.com/forms/d/e/1FAIpQLSfpJB5tdkI7BccTCReoszV9cyyX2rF99SgznzwlOepi5v-xLw/viewform?usp=header
   :alt: Mailing list

.. image:: https://img.shields.io/pypi/v/pyhealth.svg?color=brightgreen
   :target: https://pypi.org/project/pyhealth/
   :alt: PyPI version


.. image:: https://img.shields.io/github/stars/sunlabuiuc/pyhealth.svg
   :target: https://github.com/sunlabuiuc/pyhealth/stargazers
   :alt: GitHub stars


.. image:: https://img.shields.io/github/forks/sunlabuiuc/pyhealth.svg?color=blue
   :target: https://github.com/sunlabuiuc/pyhealth/network
   :alt: GitHub forks


.. image:: https://static.pepy.tech/badge/pyhealth
   :target: https://pepy.tech/project/pyhealth
   :alt: Downloads


.. image:: https://img.shields.io/badge/Tutorials-Google%20Colab-red
   :target: https://pyhealth.readthedocs.io/en/latest/tutorials.html
   :alt: Tutorials


.. image:: https://img.shields.io/badge/YouTube-16%20Videos-red
   :target: https://www.youtube.com/playlist?list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV
   :alt: YouTube

.. image:: https://github.com/sunlabuiuc/PyHealth/workflows/CI/badge.svg
   :target: https://github.com/sunlabuiuc/PyHealth/actions
   :alt: CI status


Citing PyHealth :handshake:
----------------------------------
 Yang, Chaoqi, Zhenbang Wu, Patrick Jiang, Zhen Lin, Junyi Gao, Benjamin P. Danek, and Jimeng Sun. 2023. "PyHealth: A Deep Learning Toolkit for Healthcare Applications." In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 5788–89. KDD '23. New York, NY, USA: Association for Computing Machinery.

.. code-block:: bibtex

    @inproceedings{pyhealth2023yang,
        author = {Yang, Chaoqi and Wu, Zhenbang and Jiang, Patrick and Lin, Zhen and Gao, Junyi and Danek, Benjamin and Sun, Jimeng},
        title = {{PyHealth}: A Deep Learning Toolkit for Healthcare Predictive Modeling},
        url = {https://github.com/sunlabuiuc/PyHealth},
        booktitle = {Proceedings of the 27th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD) 2023},
        year = {2023}
    }

-----------------------------------------------------------------


PyHealth is a comprehensive deep learning toolkit for supporting clinical predictive modeling, which is designed for both **ML researchers and medical practitioners**. We can make your **healthcare AI applications** easier to develop, test, and deploy—more flexible and more customizable. `[Tutorials] <https://pyhealth.readthedocs.io/>`_

**Key Features**

- Modular 5-stage pipeline for healthcare ML
- Healthcare-first: medical codes and clinical datasets (MIMIC, eICU, OMOP)
- 33+ pre-built models and production-ready trainer/metrics
- 10+ supported healthcare tasks and datasets
- Fast (~3x faster than pandas) data processing for quick experimentation

 **[News!]** We are continuously implementing good papers and benchmarks into PyHealth, checkout the `[Planned List] <https://docs.google.com/spreadsheets/d/1PNMgDe-llOm1SM5ZyGLkmPysjC4wwaVblPLAHLxejTw/edit#gid=159213380>`_. Welcome to pick one from the list and send us a PR or add more influential and new papers into the plan list.

.. image:: figure/poster.png
   :width: 810

..

1. Installation :rocket:
----------------------------

**Python Version Requirement**

PyHealth 2.0 requires **Python 3.12 or 3.13** (``>=3.12,<3.14``). This version requirement enables optimal parallel processing, memory management, and compatibility with our modern dependencies.

**Recommended Installation (Latest Release)**

Install the latest PyHealth 2.0 release from PyPI:

.. code-block:: sh

    pip install pyhealth

This version includes significant performance improvements, dynamic memory support, parallelized processing, multimodal dataloaders, and many new features.

**Legacy Version**

The older stable version (1.16) is still available for backward compatibility and supports Python 3.9+:

.. code-block:: sh

    pip install pyhealth==1.16

**For Contributors and Developers**

If you are contributing to PyHealth or need the latest development features, install from GitHub source:

.. code-block:: sh

    git clone https://github.com/sunlabuiuc/PyHealth.git
    cd PyHealth
    pip install -e .

**Note:** PyHealth 2.0 automatically installs PyTorch and other deep learning dependencies. The alpha version includes all required libraries for neural network-based models.


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
        root="https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III/",
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

``pyhealth.tasks`` defines how to process each patient's data into a set of samples for the tasks. In the package, we provide several task examples, such as ``drug recommendation``, ``mortality prediction``, and ``readmission prediction``. **It is easy to customize your own tasks following our** `template <https://colab.research.google.com/drive/1r7MYQR_5yCJGpK_9I9-A10HmpupZuIN-?usp=sharing>`_.

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

``pyhealth.metrics`` provides several **common evaluation metrics** (refer to `Doc <https://pyhealth.readthedocs.io/en/latest/api/metrics.html>`_ and see what are available).

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

6. Tutorials :teacher:
----------------------------

.. image:: https://colab.research.google.com/assets/colab-badge.svg
    :target: https://pyhealth.readthedocs.io/en/latest/tutorials.html

..

 We provide the following tutorials to help users get started with our pyhealth. Please bear with us as we update the documentation on how to use PyHealth 2.0.

`Tutorial 0: Introduction to pyhealth.data <https://colab.research.google.com/drive/1y9PawgSbyMbSSMw1dpfwtooH7qzOEYdN?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=Nk1itBoLOX8&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=2>`__

`Tutorial 1: Introduction to pyhealth.datasets <https://colab.research.google.com/drive/1voSx7wEfzXfEf2sIfW6b-8p1KqMyuWxK?usp=sharing>`_  `[Video (PyHealth 1.16)] <https://www.youtube.com/watch?v=c1InKqFJbsI&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=3>`__

`Tutorial 2: Introduction to pyhealth.tasks <https://colab.research.google.com/drive/1kKkkBVS_GclHoYTbnOtjyYnSee79hsyT?usp=sharing>`_  `[Video (PyHealth 1.16)] <https://www.youtube.com/watch?v=CxESe1gYWU4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=4>`__

`Tutorial 3: Introduction to pyhealth.models <https://colab.research.google.com/drive/1LcXZlu7ZUuqepf269X3FhXuhHeRvaJX5?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=fRc0ncbTgZA&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=6>`__

`Tutorial 4: Introduction to pyhealth.trainer <https://colab.research.google.com/drive/1L1Nz76cRNB7wTp5Pz_4Vp4N2eRZ9R6xl?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=5Hyw3of5pO4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=7>`__

`Tutorial 5: Introduction to pyhealth.metrics <https://colab.research.google.com/drive/1Mrs77EJ92HwMgDaElJ_CBXbi4iABZBeo?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=d-Kx_xCwre4&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=8>`__


`Tutorial 6: Introduction to pyhealth.tokenizer <https://colab.research.google.com/drive/1bDOb0A5g0umBjtz8NIp4wqye7taJ03D0?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=CeXJtf0lfs0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=10>`__


`Tutorial 7: Introduction to pyhealth.medcode <https://colab.research.google.com/drive/1xrp_ACM2_Hg5Wxzj0SKKKgZfMY0WwEj3?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=MmmfU6_xkYg&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=9>`__


 The following tutorials will help users build their own task pipelines.

`Pipeline 1: Chest Xray Classification <https://colab.research.google.com/drive/18vK23gyI1LjWbTgkq4f99yDZA3A7Pxp9?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=GGP3Dhfyisc&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=12>`__

`Pipeline 2: Medical Coding <https://colab.research.google.com/drive/1ThYP_5ng5xPQwscv5XztefkkoTruhjeK?usp=sharing>`_

`Pipeline 3: Medical Transcription Classification <https://colab.research.google.com/drive/1bjk_IArc2ZmXGR6u6Qzyf7kh70RdiY9c?usp=sharing>`_

`Pipeline 4: Mortality Prediction <https://colab.research.google.com/drive/1b9xRbxUz-HLzxsrvxdsdJ868ajGQCY6U?usp=sharing>`_

`Pipeline 5: Readmission Prediction <https://colab.research.google.com/drive/1h0pAymUlPQfkLFryI9QI37-HAW1tRxGZ?usp=sharing>`_


 We provide advanced tutorials for supporting various needs.

`Advanced Tutorial 1: Fit your dataset into our pipeline <https://colab.research.google.com/drive/1UurxwAAov1bL_5OO3gQJ4gAa_paeJwJp?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=xw2hGLEQ4Y0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=13>`__

`Advanced Tutorial 2: Define your own healthcare task <https://colab.research.google.com/drive/1gK6zPXvfFGBM1uNaLP32BOKrnnJdqRq2?usp=sharing>`_

`Advanced Tutorial 3: Adopt customized model into pyhealth <https://colab.research.google.com/drive/1F_NJ90GC8_Eq-vKTf7Tyziew4gWjjKoH?usp=sharing>`_  `[Video] <https://www.youtube.com/watch?v=lADFlcmLtdE&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=14>`__

`Advanced Tutorial 4: Load your own processed data into pyhealth and try out our ML models <https://colab.research.google.com/drive/1ZRnKch2EyJLrI3G5AvDXVpeE2wwgBWfw?usp=sharing>`_ `[Video] <https://www.youtube.com/watch?v=xw2hGLEQ4Y0&list=PLR3CNIF8DDHJUl8RLhyOVpX_kT4bxulEV&index=13>`__


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

* Check the `interactive map on benchmark EHR predictive tasks <https://pyhealth.readthedocs.io/en/latest/index.html#benchmark-on-healthcare-tasks>`_.


9. Research Initiative :microscope:
--------------------------------------------

The **PyHealth Research Initiative** is a year-round, open research program that brings together talented individuals from diverse backgrounds to conduct cutting-edge research in healthcare AI.

**How to participate:**

1. Join our `Discord server <https://discord.gg/mpb835EHaX>`_
2. Submit a high-quality PR to the `PyHealth repository <https://github.com/sunlabuiuc/PyHealth>`_
3. Check the `documentation <https://pyhealth.readthedocs.io/en/latest/research_initiative.html>`_ for more details

Recent research from the initiative has been published at venues including **ML4H 2025** and other top conferences.


10. About Us :busts_in_silhouette:
--------------------------------------------

We are the `SunLab <http://sunlab.org/>`_ healthcare research team at UIUC.

**Current Maintainers:**

- `Zhenbang Wu <https://zzachw.github.io/>`_ (Ph.D. Student @ UIUC)
- `John Wu <https://jhnwu3.github.io/>`_ (Ph.D. Student @ UIUC)
- `Junyi Gao <http://aboutme.vixerunt.org/>`_ (Ph.D. Student @ University of Edinburgh)
- `Jimeng Sun <http://sunlab.org/>`_ (Professor @ UIUC)

**Get in Touch:**

- `Discord Community <https://discord.gg/mpb835EHaX>`_ (fastest response)
- `GitHub Issues <https://github.com/sunlabuiuc/PyHealth/issues>`_
- `Mailing List <https://docs.google.com/forms/d/e/1FAIpQLSfpJB5tdkI7BccTCReoszV9cyyX2rF99SgznzwlOepi5v-xLw/viewform?usp=header>`_

